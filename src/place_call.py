"""Trigger the agent to place an outbound phone call.

This dispatches the "Avery-ff5" agent into a new room with the call details
in the job metadata; the agent's entrypoint (see agent.py) reads that
metadata, dials out via LiveKit SIP, and starts the conversation.

Requires a LiveKit Cloud project (SIP is not available on a local
`livekit-server --dev` instance) and an outbound SIP trunk already created
via `lk sip outbound create` -- see the "Outbound Phone Calls (Twilio)"
section in README.md.

Usage:
    python src/place_call.py +15105550100
    python src/place_call.py +15105550100 --language es
    python src/place_call.py +15105550100 \\
        --prompt "You are Avery, calling on behalf of Jane's son to check in." \\
        --helper-prompt "Jane prefers short calls and goes by 'Janie'." \\
        --language en

With no --prompt, this uses Avery's built-in elder-companion persona (the
same default as a plain console/dev session) -- this is the quick manual
way to test that persona against a real phone call. --prompt/--helper-prompt
/--language give it the same per-call override capability as the outbound
call API (src/api.py) for testing a caller-supplied persona without running
that whole service.
"""

import argparse
import asyncio
import json

from dotenv import load_dotenv
from livekit import api
from livekit.agents.utils import shortuuid

load_dotenv(".env.local")  # local dev config, per README (git-ignored)
load_dotenv(".env")  # optional fallback for anything not in .env.local


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dispatch the Avery-ff5 agent to place an outbound call.",
    )
    parser.add_argument(
        "phone_number", help="E.164 phone number to call, e.g. +15105550100"
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Override the agent's full persona/instructions for this call. "
            "Omit to use Avery's built-in elder-companion persona."
        ),
    )
    parser.add_argument(
        "--helper-prompt",
        default=None,
        help="Optional supplementary context appended after --prompt.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="STT/TTS/LLM language for this call (default: agent.py's default, 'en').",
    )
    parser.add_argument(
        "--greeting",
        default=None,
        help=(
            "Exact opening line to speak on answer (skips the LLM round trip). "
            "Defaults to Avery's built-in greeting, or a generated one when "
            "--prompt is given without this."
        ),
    )
    return parser


async def place_call(
    phone_number: str,
    *,
    prompt: str | None = None,
    helper_prompt: str | None = None,
    language: str | None = None,
    greeting: str | None = None,
) -> None:
    room_name = f"outbound-{shortuuid()}"
    metadata: dict[str, str] = {"phone_number": phone_number}
    if prompt:
        metadata["prompt"] = prompt
    if helper_prompt:
        metadata["helper_prompt"] = helper_prompt
    if language:
        metadata["language"] = language
    if greeting:
        metadata["greeting"] = greeting

    async with api.LiveKitAPI() as lkapi:
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="Avery-ff5",
                room=room_name,
                metadata=json.dumps(metadata),
            )
        )
    print(f"dispatched outbound call to {phone_number} in room {dispatch.room}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    asyncio.run(
        place_call(
            args.phone_number,
            prompt=args.prompt,
            helper_prompt=args.helper_prompt,
            language=args.language,
            greeting=args.greeting,
        )
    )


if __name__ == "__main__":
    main()
