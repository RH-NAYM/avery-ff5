"""Behavioral tests for the voice-realism / phone-call pacing instructions
in DefaultAgent (see src/agent.py). These cover the "Output rules" section
added to make the agent sound like a real phone conversation instead of a
written assistant: short replies, one question at a time, no reading out
markdown or lists.
"""

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from livekit.agents import AgentSession, ChatContext, inference
from livekit.agents.evals import JudgeGroup, conciseness_judge, relevancy_judge

import agent as agent_module
from agent import ClosingAgent, DefaultAgent, on_session_end, summarize_session
from languages import LANGUAGES, is_supported_language, resolve_language

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


# --- Closing-phase handoff (#17 / #33) -------------------------------------
# These run fully offline (no LLM/API key needed) since they exercise the
# handoff mechanism itself, not conversational quality.


def test_default_agent_registers_begin_wrap_up_tool() -> None:
    tool_names = {t.info.name for t in DefaultAgent().tools}
    assert "begin_wrap_up" in tool_names


def test_custom_prompt_persona_also_registers_begin_wrap_up_tool() -> None:
    # The closing-phase trigger has to apply regardless of which persona
    # branch built the instructions (the default elder-companion persona,
    # or a caller-supplied `prompt` via the outbound call API) -- it's
    # defined on DefaultAgent itself, not tucked inside persona text.
    custom = DefaultAgent(prompt="You are a pizza-ordering assistant.")
    tool_names = {t.info.name for t in custom.tools}
    assert "begin_wrap_up" in tool_names


@pytest.mark.asyncio
async def test_begin_wrap_up_hands_off_to_a_dedicated_closing_agent() -> None:
    default_agent = DefaultAgent(language="es")
    result = await default_agent.begin_wrap_up()

    assert isinstance(result, ClosingAgent)


# --- Structured call summary (#19) ------------------------------------------
# Fakes just enough of the LLM surface (chat().collect() -> an object with
# .tool_calls/.text) to test summarize_session()'s parsing logic without a
# live model.


class _FakeToolCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeCollectedResponse:
    def __init__(self, tool_calls=(), text: str = "") -> None:
        self.tool_calls = list(tool_calls)
        self.text = text


class _FakeLLMStream:
    def __init__(self, response: _FakeCollectedResponse) -> None:
        self._response = response

    async def collect(self) -> _FakeCollectedResponse:
        return self._response


class _FakeSummarizer:
    def __init__(self, response: _FakeCollectedResponse) -> None:
        self._response = response
        self.last_kwargs: dict | None = None

    def chat(self, **kwargs) -> _FakeLLMStream:
        self.last_kwargs = kwargs
        return _FakeLLMStream(self._response)


def _chat_ctx_with_turns() -> ChatContext:
    ctx = ChatContext()
    ctx.add_message(role="user", content="I had a good day, went for a walk.")
    ctx.add_message(role="assistant", content="That's wonderful to hear!")
    return ctx


@pytest.mark.asyncio
async def test_summarize_session_parses_structured_tool_call() -> None:
    tool_call = _FakeToolCall(
        name="record_call_summary",
        arguments=json.dumps(
            {
                "summary": "Caller had a good day and went for a walk.",
                "concern_level": "none",
                "flagged_topics": [],
            }
        ),
    )
    summarizer = _FakeSummarizer(_FakeCollectedResponse(tool_calls=[tool_call]))

    result = await summarize_session(summarizer, _chat_ctx_with_turns())

    assert result.text == "Caller had a good day and went for a walk."
    assert result.concern_level == "none"
    assert result.flagged_topics == []
    # The tool call must be forced, not left optional -- an optional tool
    # call lets the model fall back to free text and lose concern_level.
    assert summarizer.last_kwargs["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_summarize_session_flags_urgent_concern() -> None:
    tool_call = _FakeToolCall(
        name="record_call_summary",
        arguments=json.dumps(
            {
                "summary": "Caller mentioned chest pain and felt dizzy.",
                "concern_level": "urgent",
                "flagged_topics": ["chest pain", "dizziness"],
            }
        ),
    )
    summarizer = _FakeSummarizer(_FakeCollectedResponse(tool_calls=[tool_call]))

    result = await summarize_session(summarizer, _chat_ctx_with_turns())

    assert result.concern_level == "urgent"
    assert "chest pain" in result.flagged_topics


@pytest.mark.asyncio
async def test_summarize_session_falls_back_to_text_if_tool_not_called() -> None:
    summarizer = _FakeSummarizer(
        _FakeCollectedResponse(tool_calls=[], text="A pleasant chat about gardening.")
    )

    result = await summarize_session(summarizer, _chat_ctx_with_turns())

    assert result.text == "A pleasant chat about gardening."
    assert result.concern_level == "none"


@pytest.mark.asyncio
async def test_summarize_session_returns_none_for_empty_conversation() -> None:
    summarizer = _FakeSummarizer(_FakeCollectedResponse())

    result = await summarize_session(summarizer, ChatContext())

    assert result is None


# --- Immediate greeting on answer -------------------------------------------
# The opening line is spoken straight to TTS instead of being generated, so
# these check which line gets chosen for which persona. Fully offline.


def test_default_persona_greets_with_a_fixed_line() -> None:
    # No LLM round trip needed: the line is known before the call connects.
    assert DefaultAgent().resolve_greeting() == agent_module.DEFAULT_GREETING


def test_explicit_greeting_wins_over_the_default() -> None:
    custom = DefaultAgent(greeting="Good morning, Janie, it's Avery.")
    assert custom.resolve_greeting() == "Good morning, Janie, it's Avery."


def test_caller_supplied_persona_without_a_greeting_generates_one() -> None:
    # A fixed "calling to see how you're doing" line would be wrong coming
    # from an arbitrary caller-supplied persona, so None means "let the LLM
    # open" rather than "say nothing".
    pizza_bot = DefaultAgent(prompt="You are a pizza shop confirming an order.")
    assert pizza_bot.resolve_greeting() is None


def test_caller_supplied_persona_can_still_supply_its_own_greeting() -> None:
    pizza_bot = DefaultAgent(
        prompt="You are a pizza shop confirming an order.",
        greeting="Hi, it's Tony's Pizza calling about your order.",
    )
    assert (
        pizza_bot.resolve_greeting()
        == "Hi, it's Tony's Pizza calling about your order."
    )


# --- Environment tuning knobs ------------------------------------------------
# The VAD/interruption values are all env-overridable so a deployment can
# retune against real call recordings; a typo in one of them must not take
# the worker down.


def test_env_float_reads_an_override(monkeypatch) -> None:
    monkeypatch.setenv("VAD_ACTIVATION_THRESHOLD", "0.72")
    assert agent_module._env_float("VAD_ACTIVATION_THRESHOLD", 0.6) == 0.72


def test_env_float_falls_back_on_a_bad_value(monkeypatch) -> None:
    monkeypatch.setenv("VAD_ACTIVATION_THRESHOLD", "very-high")
    assert agent_module._env_float("VAD_ACTIVATION_THRESHOLD", 0.6) == 0.6


def test_env_int_rejects_an_unsupported_sample_rate(monkeypatch) -> None:
    # Silero only supports 8kHz and 16kHz; anything else would raise inside
    # VAD.load() and kill the worker at startup.
    monkeypatch.setenv("VAD_SAMPLE_RATE", "44100")
    assert (
        agent_module._env_int("VAD_SAMPLE_RATE", 16000, allowed=(8000, 16000)) == 16000
    )


def test_env_bool_parses_common_spellings(monkeypatch) -> None:
    monkeypatch.setenv("PREEMPTIVE_TTS", "0")
    assert agent_module._env_bool("PREEMPTIVE_TTS", True) is False
    monkeypatch.setenv("PREEMPTIVE_TTS", "yes")
    assert agent_module._env_bool("PREEMPTIVE_TTS", False) is True


# --- End-of-call reporting --------------------------------------------------
# The failure these cover is a quiet one: when on_session_end doesn't manage to
# post a result, nothing errors anywhere the caller can see. The outbound call
# API simply keeps saying "pending" until CALL_TIMEOUT_SECONDS (900s) runs out
# and it reports a timeout with no summary. So each of these asserts that
# *something* is always posted back, and that a failure is reported as one.


class _FakeJob:
    def __init__(self, metadata: str) -> None:
        self.metadata = metadata


class _FakeSession:
    def __init__(self, llm, history) -> None:
        self.llm = llm
        self.history = history


class _FakeJobContext:
    """Stands in for JobContext, including the detail that matters most here:
    `primary_session` raises rather than returning None when no session was
    ever started (livekit-agents 1.7)."""

    def __init__(self, metadata: str, session=None, session_error=None) -> None:
        self.job = _FakeJob(metadata)
        self._session = session
        self._session_error = session_error

    @property
    def primary_session(self):
        if self._session_error is not None:
            raise self._session_error
        return self._session


DIAL_METADATA = json.dumps(
    {
        "call_id": "call-123",
        "phone_number": "+15105550100",
        "callback_url": "http://api.test/internal/calls/call-123/completed",
    }
)


def _captured_callback():
    posted: list[dict] = []

    async def _fake_post(callback_url: str, payload: dict) -> bool:
        posted.append(payload)
        return True

    return posted, _fake_post


@pytest.mark.asyncio
async def test_on_session_end_reports_when_no_session_was_started() -> None:
    # ctx.primary_session raises instead of returning None. Before this was
    # handled, the RuntimeError propagated out of on_session_end, the worker
    # swallowed it, and the API was left waiting on a callback that never came.
    ctx = _FakeJobContext(
        DIAL_METADATA,
        session_error=RuntimeError("No AgentSession was started for this job"),
    )
    posted, fake_post = _captured_callback()

    with patch.object(agent_module, "_post_callback", fake_post):
        await on_session_end(ctx)

    assert len(posted) == 1
    assert posted[0]["call_id"] == "call-123"
    assert posted[0]["error"] == "session never started"


@pytest.mark.asyncio
async def test_on_session_end_reports_error_when_summary_times_out() -> None:
    class _HangingSummarizer:
        def chat(self, **kwargs):
            class _Stream:
                async def collect(self):
                    await asyncio.sleep(30)

            return _Stream()

    ctx = _FakeJobContext(
        DIAL_METADATA,
        session=_FakeSession(_HangingSummarizer(), _chat_ctx_with_turns()),
    )
    posted, fake_post = _captured_callback()

    with (
        patch.object(agent_module, "_post_callback", fake_post),
        patch.object(agent_module, "SUMMARY_TIMEOUT_SECONDS", 0.05),
    ):
        await on_session_end(ctx)

    assert len(posted) == 1
    assert "timed out" in posted[0]["error"]
    # Crucially not reported as a clean call: on a wellbeing check-in,
    # concern_level "none" means "nothing came up", which is a different
    # claim from "the summary never generated".
    assert "concern_level" not in posted[0]


@pytest.mark.asyncio
async def test_on_session_end_reports_error_when_summary_raises() -> None:
    class _BrokenSummarizer:
        def chat(self, **kwargs):
            raise RuntimeError("vertex ai is unreachable")

    ctx = _FakeJobContext(
        DIAL_METADATA,
        session=_FakeSession(_BrokenSummarizer(), _chat_ctx_with_turns()),
    )
    posted, fake_post = _captured_callback()

    with patch.object(agent_module, "_post_callback", fake_post):
        await on_session_end(ctx)

    assert len(posted) == 1
    assert "vertex ai is unreachable" in posted[0]["error"]
    assert "concern_level" not in posted[0]


@pytest.mark.asyncio
async def test_on_session_end_posts_structured_summary_on_success() -> None:
    tool_call = _FakeToolCall(
        name="record_call_summary",
        arguments=json.dumps(
            {
                "summary": "Caller mentioned a fall last night.",
                "concern_level": "urgent",
                "flagged_topics": ["fall"],
            }
        ),
    )
    summarizer = _FakeSummarizer(_FakeCollectedResponse(tool_calls=[tool_call]))
    ctx = _FakeJobContext(
        DIAL_METADATA, session=_FakeSession(summarizer, _chat_ctx_with_turns())
    )
    posted, fake_post = _captured_callback()

    with patch.object(agent_module, "_post_callback", fake_post):
        await on_session_end(ctx)

    assert len(posted) == 1
    assert posted[0]["concern_level"] == "urgent"
    assert posted[0]["flagged_topics"] == ["fall"]
    assert "error" not in posted[0]


def _client_factory(outcomes: list, calls: list):
    """Fake httpx.AsyncClient whose POST returns/raises the next outcome."""

    class _Client:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, url, json=None, headers=None):
            calls.append(url)
            outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
            if isinstance(outcome, Exception):
                raise outcome
            return httpx.Response(outcome, request=httpx.Request("POST", url))

    return lambda **kwargs: _Client(**kwargs)


@pytest.mark.asyncio
async def test_post_callback_retries_until_the_result_lands() -> None:
    # The summary only exists in this process, and the worker is already
    # shutting down -- a single blip on the tunnel between worker and API
    # would otherwise lose it permanently.
    calls: list[str] = []
    outcomes = [httpx.ConnectError("tunnel down"), 503, 200]

    with (
        patch.object(
            agent_module.httpx, "AsyncClient", _client_factory(outcomes, calls)
        ),
        patch.object(agent_module, "CALLBACK_MAX_ATTEMPTS", 3),
        patch.object(agent_module, "CALLBACK_BACKOFF_SECONDS", 0.01),
    ):
        ok = await agent_module._post_callback(
            "http://api.test/cb", {"call_id": "call-123"}
        )

    assert ok is True
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_post_callback_gives_up_and_says_so() -> None:
    calls: list[str] = []

    with (
        patch.object(agent_module.httpx, "AsyncClient", _client_factory([503], calls)),
        patch.object(agent_module, "CALLBACK_MAX_ATTEMPTS", 2),
        patch.object(agent_module, "CALLBACK_BACKOFF_SECONDS", 0.01),
    ):
        ok = await agent_module._post_callback(
            "http://api.test/cb", {"call_id": "call-123"}
        )

    assert ok is False
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_post_callback_does_not_retry_a_rejected_payload() -> None:
    # A 4xx means the API understood us and said no (unknown call_id, bad
    # body). Retrying can't change that and only delays job shutdown.
    calls: list[str] = []

    with (
        patch.object(agent_module.httpx, "AsyncClient", _client_factory([404], calls)),
        patch.object(agent_module, "CALLBACK_MAX_ATTEMPTS", 4),
        patch.object(agent_module, "CALLBACK_BACKOFF_SECONDS", 0.01),
    ):
        ok = await agent_module._post_callback(
            "http://api.test/cb", {"call_id": "call-123"}
        )

    assert ok is False
    assert len(calls) == 1


def test_non_object_job_metadata_is_ignored() -> None:
    # Valid JSON of the wrong shape used to reach .get() as a str/list and
    # blow up mid-call with an AttributeError.
    assert agent_module._parse_dial_info(_FakeJobContext('"just a string"')) == {}
    assert agent_module._parse_dial_info(_FakeJobContext("[1, 2, 3]")) == {}
    assert agent_module._parse_dial_info(_FakeJobContext("not json at all")) == {}


# --- Multilingual support ---------------------------------------------------
# en / bn / es / ar / hi. The failure mode these guard against is silent: the
# language used to be threaded through the whole pipeline and then dropped
# before it ever reached STT, so a Bengali call was transcribed as en-US and
# nobody found out until they read the transcript.


@pytest.mark.parametrize("code", ["en", "bn", "es", "ar", "hi"])
def test_every_supported_language_is_fully_configured(code: str) -> None:
    profile = LANGUAGES[code]
    assert profile.code == code
    assert profile.bcp47.startswith(code)
    assert profile.greeting.strip()
    assert profile.google_stt_model
    # The greeting is spoken straight to TTS, and Gemini TTS infers the
    # language from the text it is handed -- so every language needs its own
    # line, not the English one. Bengali, Arabic and Hindi are non-Latin
    # scripts, so they can be checked directly; Spanish shares the alphabet
    # with English, so the check is just that it isn't the English line.
    if code in ("bn", "ar", "hi"):
        assert not profile.greeting.isascii(), (
            f"{code} greeting is not in its own script: {profile.greeting!r}"
        )
    if code != "en":
        assert profile.greeting != LANGUAGES["en"].greeting


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("en", "en"),
        ("bn", "bn"),
        ("bn-BD", "bn"),
        ("bengali", "bn"),
        ("ENGLISH", "en"),
        ("es-MX", "es"),
        ("hi", "hi"),
        ("ar", "ar"),
    ],
)
def test_language_resolution_accepts_codes_locales_and_names(
    value: str, expected: str
) -> None:
    assert resolve_language(value).code == expected


def test_unknown_language_falls_back_but_is_not_reported_as_supported() -> None:
    # resolve_language() can't raise -- by the time it runs the phone is
    # already ringing -- so the rejection has to happen at the API boundary,
    # which is what is_supported_language() is for.
    assert resolve_language("klingon").code == "en"
    assert resolve_language(None).code == "en"
    assert is_supported_language("klingon") is False
    assert is_supported_language("bn-BD") is True


def _captured_stt(monkeypatch_target: str):
    captured: dict = {}

    class _Stub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    return captured, _Stub


def test_google_stt_receives_the_call_language() -> None:
    captured, stub = _captured_stt("google")
    with (
        patch.dict("os.environ", {"STT_PROVIDER": "google"}),
        patch.object(agent_module.google, "STT", stub),
    ):
        agent_module._build_stt(LANGUAGES["bn"])

    # The bug: this used to be absent entirely, so the plugin fell back to
    # its own "en-US" default on every single call.
    assert captured["languages"] == ["bn-BD", "en-US"]
    # Detection is on only to choose between those two -- the call's language
    # is still fixed, English is just allowed alongside it.
    assert captured["detect_language"] is True
    # Google's official v1 table lists bn-BD under latest_long -- confirmed
    # 2026-09-09, reversing an earlier, never-actually-tested assumption.
    assert captured["model"] == "latest_long"
    assert captured["use_streaming"] is True


def test_elevenlabs_stt_receives_the_call_language_and_a_streaming_model() -> None:
    captured, stub = _captured_stt("elevenlabs")
    with (
        patch.dict(
            "os.environ", {"STT_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "k"}
        ),
        patch.object(agent_module.elevenlabs, "STT", stub),
    ):
        agent_module._build_stt(LANGUAGES["hi"])

    assert captured["language_code"] == "hi"
    # scribe_v1 is batch-only; on a live call that means no text until the
    # utterance has already ended.
    assert captured["model"] == "scribe_v2_realtime"
    # Regression test: without server_vad, the plugin connects with
    # commit_strategy=manual, which never finalizes a transcript on its own
    # (confirmed on a live Bengali call -- partial_transcript kept arriving
    # and drifting for 40+ seconds, committed_transcript never arrived, the
    # agent never replied). server_vad must always be passed.
    # These values are a latency decision, not a style one: no turn can
    # commit and no LLM call can start until ElevenLabs' server finalizes,
    # so this wait is a hard floor under every single reply. Lowered from
    # 1.5s/800ms on 2026-09-09.
    assert captured["server_vad"] == {
        "vad_silence_threshold_secs": 0.7,
        "min_silence_duration_ms": 400,
    }


def test_elevenlabs_stt_vad_silence_is_overridable() -> None:
    captured, stub = _captured_stt("elevenlabs")
    with (
        patch.dict(
            "os.environ",
            {
                "STT_PROVIDER": "elevenlabs",
                "ELEVENLABS_API_KEY": "k",
                "ELEVENLABS_STT_VAD_SILENCE_SECS": "2.0",
                "ELEVENLABS_STT_VAD_MIN_SILENCE_MS": "1200",
            },
        ),
        patch.object(agent_module.elevenlabs, "STT", stub),
    ):
        agent_module._build_stt(LANGUAGES["en"])

    assert captured["server_vad"] == {
        "vad_silence_threshold_secs": 2.0,
        "min_silence_duration_ms": 1200,
    }


def test_google_stt_model_is_overridable_per_language() -> None:
    with patch.dict("os.environ", {"GOOGLE_STT_MODEL_BN": "chirp_2"}):
        assert agent_module._google_stt_model(LANGUAGES["bn"]) == "chirp_2"
        # ...without disturbing the others
        assert agent_module._google_stt_model(LANGUAGES["en"]) == "latest_long"


@pytest.mark.parametrize("code", ["bn", "es", "ar", "hi"])
def test_default_persona_greets_in_the_call_language(code: str) -> None:
    assert DefaultAgent(language=code).resolve_greeting() == LANGUAGES[code].greeting


def test_persona_is_told_the_language_by_name_not_by_code() -> None:
    # "Speak only in bn" is a much weaker instruction to a model than
    # "Speak only in Bengali (Bangla)".
    instructions = DefaultAgent(language="bn").instructions
    assert "Bengali (Bangla)" in instructions
    assert "Speak only in bn" not in instructions


def test_english_persona_gets_no_redundant_language_instruction() -> None:
    assert "Speak only in English" not in DefaultAgent(language="en").instructions


def test_caller_supplied_persona_also_gets_the_language_instruction() -> None:
    # A custom `prompt` replaces the built-in persona entirely, so the
    # language rule has to be re-applied on top of it or a Spanish call with
    # a custom persona answers in English.
    custom = DefaultAgent(prompt="You are a pizza-ordering assistant.", language="es")
    assert "Spanish" in custom.instructions


def test_closing_agent_stays_in_the_call_language() -> None:
    assert "Arabic" in ClosingAgent(language="ar").instructions


@pytest.mark.asyncio
async def test_wrap_up_handoff_preserves_the_language() -> None:
    closing = await DefaultAgent(language="hi").begin_wrap_up()
    assert "Hindi" in closing.instructions


@pytest.mark.parametrize("code", ["en", "es", "ar", "hi"])
def test_turn_detector_is_used_for_supported_languages(code: str) -> None:
    assert agent_module.resolve_turn_detection(LANGUAGES[code]) is not None


@pytest.mark.parametrize("code", ["bn"])
def test_turn_detector_falls_back_to_vad_for_unsupported_languages(code: str) -> None:
    # Neither the old text model nor the audio model scores these. Returning
    # None is the honest answer -- the session then commits turns on the
    # endpointing delay instead of pretending it has a signal.
    assert agent_module.resolve_turn_detection(LANGUAGES[code]) is None


# --- STT provider selection (ElevenLabs is the default) --------------------


def test_elevenlabs_is_the_default_stt_for_every_language() -> None:
    # ElevenLabs (scribe_v2_realtime) covers all five languages in one
    # streaming model and sidesteps a Google v1 streaming failure that was
    # never root-caused (worked for one exchange on a live Bengali call, then
    # went silent with no error for the rest of the call). This is the code
    # default even with no ELEVENLABS_API_KEY in the environment -- the key's
    # absence only breaks _build_stt (via _require_env), not provider
    # selection, so that failure is loud and immediate rather than a silent
    # fallback to a different provider.
    with patch.dict("os.environ", {}, clear=False):
        os_env = agent_module.os.environ
        os_env.pop("STT_PROVIDER", None)
        os_env.pop("ELEVENLABS_API_KEY", None)
        for code in LANGUAGES:
            assert agent_module._resolve_stt_provider(LANGUAGES[code]) == "elevenlabs"


def test_gemini_still_works_when_explicitly_selected() -> None:
    # The Vertex path is broken on this deployment right now, but the code
    # path itself stays fully wired -- setting GEMINI_API_KEY and
    # STT_PROVIDER=gemini switches onto the (working) non-Vertex API-key path.
    with patch.dict("os.environ", {"STT_PROVIDER": "gemini"}, clear=False):
        for code in LANGUAGES:
            assert agent_module._resolve_stt_provider(LANGUAGES[code]) == "gemini"


def test_gemini_stt_gets_the_language_and_allows_english_alongside_it() -> None:
    captured: dict = {}

    class _Stub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with (
        patch.dict(
            "os.environ",
            {
                "STT_PROVIDER": "gemini",
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/key.json",
            },
        ),
        patch.object(agent_module, "GeminiSTT", _Stub),
    ):
        agent_module._build_stt(LANGUAGES["bn"])

    assert captured["language"] == "bn-BD"
    # Bangla speakers routinely drop English words mid-sentence; a recognizer
    # locked to one language turns those into native-sounding nonsense.
    assert captured["language_codes"] == ["bn-BD", "en-US"]
    assert captured["model"] == "gemini-3.5-transcribe-live"


def test_gemini_stt_uses_vertex_when_no_api_key_is_set() -> None:
    captured: dict = {}

    class _Stub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    env = {
        "STT_PROVIDER": "gemini",
        "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/key.json",
    }
    with (
        patch.dict("os.environ", env),
        patch.object(agent_module, "GeminiSTT", _Stub),
    ):
        agent_module.os.environ.pop("GEMINI_API_KEY", None)
        agent_module._build_stt(LANGUAGES["hi"])

    # Same service account the LLM and TTS already authenticate with, so this
    # needs no new key and no new IAM grant.
    assert captured["vertexai"] is True
    assert captured["credentials_path"] == "/tmp/key.json"
    assert "api_key" not in captured


def test_english_is_not_listed_twice_for_an_english_call() -> None:
    assert LANGUAGES["en"].stt_language_codes == ("en-US",)


@pytest.mark.parametrize("code", ["bn", "es", "ar", "hi"])
def test_every_non_english_language_allows_english_alongside_it(code: str) -> None:
    codes = LANGUAGES[code].stt_language_codes
    assert codes[0].startswith(code)
    assert "en-US" in codes


def test_a_per_language_override_still_wins_over_the_default() -> None:
    with patch.dict(
        "os.environ",
        {"STT_PROVIDER": "google", "STT_PROVIDER_BN": "elevenlabs"},
        clear=False,
    ):
        assert agent_module._resolve_stt_provider(LANGUAGES["bn"]) == "elevenlabs"
        # No override for hi, so it falls through to the global default --
        # forced to "google" here specifically to prove the override beats
        # it, not because google is the code default any more.
        assert agent_module._resolve_stt_provider(LANGUAGES["hi"]) == "google"


# --- TTS voice / locale (Google Chirp3-HD) -----------------------------------
# TTS_PROVIDER=google switched from TTS_PROVIDER=gemini (which is hardcoded
# non-streaming and was the root cause of "TTS buffering" reported on live
# calls). Chirp3-HD, the voice set that gives named, same-character voices
# across languages (e.g. "Leda"), only covers a subset of locales -- bn-IN
# and ar-XA, not this project's STT-side bn-BD/ar-EG -- so _build_tts must
# send Google's TTS a different language code than STT gets for those two.


def _captured_tts():
    captured: dict = {}

    class _Stub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    return captured, _Stub


@pytest.mark.parametrize(
    ("code", "expected_language"),
    [
        ("en", "en-US"),
        ("bn", "bn-IN"),  # not bn-BD -- see google_tts_bcp47
        ("es", "es-US"),
        ("ar", "ar-XA"),  # not ar-EG -- see google_tts_bcp47
        ("hi", "hi-IN"),
    ],
)
def test_google_tts_uses_a_chirp3_hd_compatible_locale(
    code: str, expected_language: str
) -> None:
    captured, stub = _captured_tts()
    with (
        patch.dict("os.environ", {"TTS_PROVIDER": "google"}, clear=False),
        patch.object(agent_module.google, "TTS", stub),
    ):
        agent_module._build_tts(LANGUAGES[code])

    assert captured["language"] == expected_language


def test_google_tts_receives_the_per_language_voice_override(monkeypatch) -> None:
    captured, stub = _captured_tts()
    monkeypatch.setenv("TTS_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_TTS_VOICE_BN", "bn-IN-Chirp3-HD-Leda")
    with patch.object(agent_module.google, "TTS", stub):
        agent_module._build_tts(LANGUAGES["bn"])

    assert captured["voice_name"] == "bn-IN-Chirp3-HD-Leda"


def test_google_tts_falls_back_to_the_unsuffixed_voice_override(monkeypatch) -> None:
    # A language with no GOOGLE_TTS_VOICE_<CODE> of its own still gets the
    # unsuffixed GOOGLE_TTS_VOICE, same fallback pattern as ELEVENLABS_VOICE_ID.
    captured, stub = _captured_tts()
    monkeypatch.setenv("TTS_PROVIDER", "google")
    monkeypatch.delenv("GOOGLE_TTS_VOICE_HI", raising=False)
    monkeypatch.setenv("GOOGLE_TTS_VOICE", "en-US-Chirp3-HD-Leda")
    with patch.object(agent_module.google, "TTS", stub):
        agent_module._build_tts(LANGUAGES["hi"])

    assert captured["voice_name"] == "en-US-Chirp3-HD-Leda"


# ---------------------------------------------------------------------------
# Latency: LLM thinking budget and Vertex region
# ---------------------------------------------------------------------------


def _captured_llm():
    captured: dict = {}

    def stub(**kwargs):
        captured.update(kwargs)
        return object()

    return captured, stub


def test_gemini_thinking_is_disabled_by_default(monkeypatch) -> None:
    """Gemini 2.5 Flash reasons before answering unless told not to, and
    those tokens land entirely inside time-to-first-token -- dead air on a
    phone call. Nothing in the transcript shows it, so this is easy to
    regress by simply dropping the argument again."""
    captured, stub = _captured_llm()
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with patch.object(agent_module.google, "LLM", stub):
        agent_module._build_llm()

    assert captured["thinking_config"] == {"thinking_budget": 0}


def test_gemini_thinking_budget_is_overridable(monkeypatch) -> None:
    captured, stub = _captured_llm()
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "512")
    with patch.object(agent_module.google, "LLM", stub):
        agent_module._build_llm()

    assert captured["thinking_config"] == {"thinking_budget": 512}


def test_a_negative_thinking_budget_restores_the_model_default(monkeypatch) -> None:
    """-1 means "send no thinking_config at all", which is different from
    sending a budget of 0 -- it hands the decision back to the model."""
    captured, stub = _captured_llm()
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "-1")
    with patch.object(agent_module.google, "LLM", stub):
        agent_module._build_llm()

    assert captured["thinking_config"] is agent_module.NOT_GIVEN


def test_gemini_location_is_passed_through_on_the_vertex_path(monkeypatch) -> None:
    """The plugin's own default is us-central1, which from a worker outside
    North America adds a Pacific crossing to every streaming round trip."""
    captured, stub = _captured_llm()
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_LOCATION", "asia-south1")
    with patch.object(agent_module.google, "LLM", stub):
        agent_module._build_llm()

    assert captured["vertexai"] is True
    assert captured["location"] == "asia-south1"


def test_an_unset_gemini_location_leaves_the_plugin_default(monkeypatch) -> None:
    captured, stub = _captured_llm()
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GEMINI_LOCATION", raising=False)
    with patch.object(agent_module.google, "LLM", stub):
        agent_module._build_llm()

    assert captured["location"] is agent_module.NOT_GIVEN


# ---------------------------------------------------------------------------
# ElevenLabs STT language pinning vs. auto-detect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["en", "es", "ar", "hi"])
def test_elevenlabs_stt_pins_the_language_for_most_languages(code: str) -> None:
    captured, stub = _captured_stt("elevenlabs")
    with (
        patch.dict(
            "os.environ", {"STT_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "k"}
        ),
        patch.object(agent_module.elevenlabs, "STT", stub),
    ):
        agent_module._build_stt(LANGUAGES[code])

    assert captured["language_code"] == code


def test_bengali_lets_elevenlabs_detect_the_language() -> None:
    """Regression test for a live Bengali call that produced no transcript at
    all while English worked on the same build. The plugin only adds
    include_language_detection=true when no language_code is passed, so None
    here -- not the string "auto", and not "bn" -- is what actually reaches
    the wire. The failure this guards against is silent: an empty committed
    transcript emits no FINAL_TRANSCRIPT and raises nothing."""
    captured, stub = _captured_stt("elevenlabs")
    with (
        patch.dict(
            "os.environ", {"STT_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "k"}
        ),
        patch.object(agent_module.elevenlabs, "STT", stub),
    ):
        agent_module._build_stt(LANGUAGES["bn"])

    assert captured["language_code"] is None


def test_a_per_language_elevenlabs_code_can_pin_bengali_back() -> None:
    captured, stub = _captured_stt("elevenlabs")
    with (
        patch.dict(
            "os.environ",
            {
                "STT_PROVIDER": "elevenlabs",
                "ELEVENLABS_API_KEY": "k",
                "ELEVENLABS_STT_LANGUAGE_BN": "bn",
            },
        ),
        patch.object(agent_module.elevenlabs, "STT", stub),
    ):
        agent_module._build_stt(LANGUAGES["bn"])

    assert captured["language_code"] == "bn"


@pytest.mark.parametrize("value", ["auto", "detect", "", "  AUTO  "])
def test_auto_in_any_spelling_means_detect(value: str) -> None:
    captured, stub = _captured_stt("elevenlabs")
    with (
        patch.dict(
            "os.environ",
            {
                "STT_PROVIDER": "elevenlabs",
                "ELEVENLABS_API_KEY": "k",
                "ELEVENLABS_STT_LANGUAGE_HI": value,
            },
        ),
        patch.object(agent_module.elevenlabs, "STT", stub),
    ):
        agent_module._build_stt(LANGUAGES["hi"])

    assert captured["language_code"] is None


def test_the_unsuffixed_elevenlabs_language_applies_to_every_language() -> None:
    captured, stub = _captured_stt("elevenlabs")
    with (
        patch.dict(
            "os.environ",
            {
                "STT_PROVIDER": "elevenlabs",
                "ELEVENLABS_API_KEY": "k",
                "ELEVENLABS_STT_LANGUAGE": "auto",
            },
        ),
        patch.object(agent_module.elevenlabs, "STT", stub),
    ):
        agent_module._build_stt(LANGUAGES["en"])

    assert captured["language_code"] is None


# ---------------------------------------------------------------------------
# Transcript script detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("আজ আপনি কেমন আছেন", "non-latin"),
        ("ami bhalo achi", "latin"),
        ("আমার doctor appointment", "mixed"),
        ("مرحبا", "non-latin"),
        ("...", "none"),
        ("", "none"),
    ],
)
def test_script_of_names_the_alphabet_that_came_back(text: str, expected: str) -> None:
    """The language field on an auto-detect ElevenLabs connection is not
    trustworthy -- the plugin falls back to a hardcoded "en" when the server
    reports no language, which is indistinguishable from really hearing
    English. The script of the text is not ambiguous: on a Bengali call,
    "latin" means no Bengali was produced whatever the language field says."""
    assert agent_module._script_of(text) == expected
