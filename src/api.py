"""HTTP API that places an outbound call, waits for the conversation to
finish, and returns the input echoed back with a `response_summary` field.

Trigger flow:
1. POST /calls/outbound dispatches the "Avery-ff5" agent (see agent.py) into
   a new room with the call details (prompt, helper_prompt, language, a
   callback URL) in the job metadata, then waits on an asyncio.Future keyed
   by call_id.
2. The agent dials out via LiveKit SIP and runs the conversation using the
   supplied prompt as its instructions.
3. When the session ends, the agent's on_session_end callback summarizes the
   conversation and POSTs it to POST /internal/calls/{call_id}/completed on
   this service, which resolves the waiting future.

This requires a LiveKit Cloud project (SIP is not available on a local
`livekit-server --dev` instance) and an outbound SIP trunk — see the
"Outbound Phone Calls (Twilio)" section in README.md — plus the agent
worker running separately (`uv run python src/agent.py dev` or a production
deployment).

Run with:
    uv run uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from livekit import api as lk_api
from livekit.agents.utils import shortuuid
from pydantic import BaseModel

load_dotenv(".env.local")  # local dev config, per README (git-ignored)
load_dotenv(".env")  # optional fallback for anything not in .env.local

logger = logging.getLogger("outbound-call-api")

AGENT_NAME = "Avery-ff5"

# How long to wait for a call to finish (be answered, held, hung up, and
# summarized) before giving up and returning an error. Outbound calls can
# run long, so this defaults generously.
CALL_TIMEOUT_SECONDS = float(os.environ.get("CALL_TIMEOUT_SECONDS", "900"))

# Base URL this API is reachable at from the agent worker process, used to
# build each call's callback URL. Defaults to same-host dev; in production,
# where the agent worker runs on separate (e.g. LiveKit Cloud) infrastructure,
# this must be set to this service's public/internal URL.
CALLBACK_BASE_URL = os.environ.get("CALLBACK_BASE_URL", "http://localhost:8000").rstrip(
    "/"
)


class OutboundCallRequest(BaseModel):
    call_type: str = "outbound"
    user_id: str
    number: str
    prompt: str
    helper_prompt: str | None = None
    # Field name matches the API's request/response contract as given.
    langage: str = "en"


class OutboundCallResponse(OutboundCallRequest):
    response_summary: str


class CallCompletedPayload(BaseModel):
    response_summary: str | None = None
    error: str | None = None


# call_id -> pending result future. In-memory and per-process: a restart of
# this service loses in-flight calls, and this only works with a single
# uvicorn worker process. Move to a shared store (e.g. Redis) if you need
# multiple API processes or restart resilience.
_pending_calls: dict[str, asyncio.Future[str]] = {}

app = FastAPI(title="Avery Outbound Call API")


@app.post("/calls/outbound", response_model=OutboundCallResponse)
async def create_outbound_call(body: OutboundCallRequest) -> OutboundCallResponse:
    if body.call_type != "outbound":
        raise HTTPException(400, f"unsupported call_type: {body.call_type!r}")

    call_id = uuid.uuid4().hex
    result_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    _pending_calls[call_id] = result_future

    room_name = f"outbound-{shortuuid()}"
    metadata = {
        "call_id": call_id,
        "phone_number": body.number,
        "user_id": body.user_id,
        "prompt": body.prompt,
        "helper_prompt": body.helper_prompt,
        "language": body.langage,
        "callback_url": f"{CALLBACK_BASE_URL}/internal/calls/{call_id}/completed",
    }

    try:
        async with lk_api.LiveKitAPI() as lkapi:
            await lkapi.agent_dispatch.create_dispatch(
                lk_api.CreateAgentDispatchRequest(
                    agent_name=AGENT_NAME,
                    room=room_name,
                    metadata=json.dumps(metadata),
                )
            )

        try:
            response_summary = await asyncio.wait_for(
                result_future, timeout=CALL_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                504, "call did not complete within the timeout"
            ) from None
    finally:
        _pending_calls.pop(call_id, None)

    return OutboundCallResponse(**body.model_dump(), response_summary=response_summary)


@app.post("/internal/calls/{call_id}/completed")
async def call_completed(call_id: str, payload: CallCompletedPayload) -> dict:
    future = _pending_calls.get(call_id)
    if future is None or future.done():
        # Unknown, late, or duplicate callback (for example, a retry after
        # the waiting request already timed out). Ack so the agent worker
        # doesn't keep retrying.
        logger.info("ignoring callback for unknown/finished call %s", call_id)
        return {"ok": True}

    if payload.error:
        future.set_result(f"Call did not complete: {payload.error}")
    else:
        future.set_result(payload.response_summary or "")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
    )
