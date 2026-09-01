"""Trigger the agent to place an outbound phone call.

This dispatches the "Avery-ff5" agent into a new room with the target phone
number in the job metadata; the agent's entrypoint (see agent.py) reads that
metadata and dials out via LiveKit SIP before starting the conversation.

Requires a LiveKit Cloud project (SIP is not available on a local
`livekit-server --dev` instance) and an outbound SIP trunk already created
via `lk sip outbound create` — see the "Outbound Phone Calls (Twilio)"
section in README.md.

Usage:
    python src/place_call.py +15105550100
"""

import asyncio
import json
import sys

from dotenv import load_dotenv
from livekit import api
from livekit.agents.utils import shortuuid

load_dotenv(".env")


async def place_call(phone_number: str) -> None:
    room_name = f"outbound-{shortuuid()}"
    async with api.LiveKitAPI() as lkapi:
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="Avery-ff5",
                room=room_name,
                metadata=json.dumps({"phone_number": phone_number}),
            )
        )
    print(f"dispatched outbound call to {phone_number} in room {dispatch.room}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python src/place_call.py <phone-number-e164>")
    asyncio.run(place_call(sys.argv[1]))


if __name__ == "__main__":
    main()
