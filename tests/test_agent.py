"""Behavioral tests for the voice-realism / phone-call pacing instructions
in DefaultAgent (see src/agent.py). These cover the "Output rules" section
added to make the agent sound like a real phone conversation instead of a
written assistant: short replies, one question at a time, no reading out
markdown or lists.
"""

import asyncio

import pytest
from livekit.agents import AgentSession, inference
from livekit.agents.evals import JudgeGroup, conciseness_judge, relevancy_judge

from agent import DefaultAgent

JUDGE_MODEL = "google/gemma-4-31b-it"


def _assistant_messages(chat_ctx):
    return [
        item
        for item in chat_ctx.items
        if item.type == "message" and item.role == "assistant"
    ]


def _sentence_count(text: str) -> int:
    return len(
        [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    )


async def _wait_for_greeting(session, timeout: float = 10.0):
    # on_enter's generate_reply() runs as a background task, so the greeting
    # isn't necessarily in session.history yet when session.start() returns.
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        messages = _assistant_messages(session.history)
        if messages:
            return messages
        await asyncio.sleep(0.05)
    raise TimeoutError("on_enter never produced a greeting message")


@pytest.mark.asyncio
async def test_greeting_is_short_and_plain_text() -> None:
    async with (
        inference.LLM(model=JUDGE_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(DefaultAgent())

        greetings = await _wait_for_greeting(session)
        greeting_text = greetings[-1].text_content

        # Output rules: short reply, no written-text formatting. Allow a
        # short "Hi there!" opener plus up to two more sentences.
        assert _sentence_count(greeting_text) <= 3, greeting_text
        for marker in ("*", "#", "```", "- "):
            assert marker not in greeting_text, greeting_text

        result = await JudgeGroup(
            llm=llm, judges=[conciseness_judge(), relevancy_judge()]
        ).evaluate(session.history)
        assert result.all_passed, result.judgments


@pytest.mark.asyncio
async def test_reply_stays_brief_and_asks_one_question() -> None:
    async with (
        inference.LLM(model=JUDGE_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(DefaultAgent())
        # Let the on_enter greeting finish first so it doesn't race with the
        # run() call below and leak an extra event into its result.
        await _wait_for_greeting(session)

        result = await session.run(
            user_input=(
                "Oh, today's been alright I suppose. I had some toast for "
                "breakfast, went out to water the garden a bit, and my "
                "daughter called earlier which was nice."
            )
        )

        reply = result.expect.next_event().is_message(role="assistant")
        reply_text = reply.event().item.text_content

        for marker in ("*", "#", "```", "- "):
            assert marker not in reply_text, reply_text

        await reply.judge(
            llm,
            intent=(
                "Responds warmly to what the person shared and asks at "
                "most one short, natural follow-up question, without "
                "listing multiple questions."
            ),
        )
