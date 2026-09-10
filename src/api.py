"""HTTP API that places an outbound call and reports back once it's done.

Trigger flow:
1. POST /calls/outbound dispatches the "Avery-ff5" agent (see agent.py) into
   a new room with the call details (prompt, helper_prompt, language, a
   callback URL) in the job metadata, and returns immediately with a
   call_id and status="pending" -- it does NOT wait for the call to finish
   (see "Why this doesn't block" below).
2. The agent dials out via LiveKit SIP and runs the conversation using the
   supplied prompt as its instructions.
3. When the session ends, the agent's on_session_end callback (agent.py)
   summarizes the conversation -- with a concern_level a backend can branch
   on, see CallSummary in agent.py -- and POSTs it to
   POST /internal/calls/{call_id}/completed on this service, which updates
   the stored call record.
4. The caller polls GET /calls/{call_id} for the result (status
   transitions pending -> completed/error/timeout).

Why this doesn't block: outbound calls can run for minutes, and an earlier
version of this API held the HTTP request open for up to
CALL_TIMEOUT_SECONDS (900s default) waiting on the result. At any real
concurrency that exhausts connection/worker limits fast. Returning
immediately and letting the caller poll (or, in a future iteration, receive
a webhook of its own) scales far better than holding a connection open for
up to 15 minutes per call.

This requires a LiveKit Cloud project (SIP is not available on a local
`livekit-server --dev` instance) and an outbound SIP trunk -- see the
"Outbound Phone Calls (Twilio)" section in README.md -- plus the agent
worker running separately (`uv run python src/agent.py dev` or a production
deployment).

Run with:
    uv run uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
(or `docker build --target api .` -- see the Dockerfile's "api" stage.)
"""

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Response
from livekit import api as lk_api
from livekit.agents.utils import shortuuid
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from languages import LANGUAGES, is_supported_language, resolve_language
from logging_utils import configure_logging

# Structured (JSON) logs when LOG_FORMAT=json is set in the environment
# (e.g. this service's production hosting); plain text otherwise.
configure_logging()
logger = logging.getLogger("outbound-call-api")

load_dotenv(".env.local")  # local dev config, per README (git-ignored)
load_dotenv(".env")  # optional fallback for anything not in .env.local

AGENT_NAME = "Avery-ff5"

# Shared secret the agent worker must present on the completion callback.
# Unset by default so existing deployments keep working, but strongly
# recommended: without it, anyone who can reach this service and guess (or
# read, from a log or an API doc page) a call_id can mark that call finished
# or errored, and the real result is then discarded as a duplicate. Set the
# same value in the worker's environment.
CALLBACK_SECRET = os.environ.get("CALLBACK_SECRET") or None

# How long a call may stay "pending" before this API gives up on it and
# marks it "timeout" (the agent worker never called back -- crashed, got
# stuck, or its callback POST itself failed). Outbound calls can run long,
# so this defaults generously.
CALL_TIMEOUT_SECONDS = float(os.environ.get("CALL_TIMEOUT_SECONDS", "900"))

# Base URL this API is reachable at from the agent worker process, used to
# build each call's callback URL. Defaults to same-host dev; in production,
# where the agent worker runs on separate (e.g. LiveKit Cloud) infrastructure,
# this must be set to this service's public/internal URL.
CALLBACK_BASE_URL = os.environ.get("CALLBACK_BASE_URL", "http://localhost:8000").rstrip(
    "/"
)

if "CALLBACK_BASE_URL" not in os.environ:
    # Worth saying loudly at startup, because the failure it causes is mute
    # and slow: the worker posts each result to its *own* localhost, nothing
    # ever arrives here, and every single call looks fine for
    # CALL_TIMEOUT_SECONDS before turning into a timeout with no summary.
    logger.warning(
        "CALLBACK_BASE_URL is not set, falling back to %s. The agent worker "
        "posts every call result to that URL, so unless the worker runs on "
        "this same host it will never reach this service and all calls will "
        "time out with no summary.",
        CALLBACK_BASE_URL,
    )

# --- Metrics --------------------------------------------------------------
# Scrape GET /metrics with Prometheus (self-hosted, or a SaaS if you'd
# rather not run your own) for call-volume, latency, and error/timeout
# rates. This custom outbound-call API is otherwise a blind spot next to
# LiveKit's own Agent Observability, which only covers the voice pipeline
# itself -- see the "Observability" section in README.md.
_CALLS_CREATED = Counter(
    "avery_calls_created_total", "Outbound calls dispatched via POST /calls/outbound"
)
_CALLS_COMPLETED = Counter(
    "avery_calls_completed_total",
    "Outbound calls that reached a terminal status",
    ["status"],  # completed | error | timeout
)
_CALL_DURATION = Histogram(
    "avery_call_duration_seconds",
    "Time from dispatch to a terminal status (completed/error/timeout)",
    buckets=(5, 15, 30, 60, 120, 300, 600, 900, 1800),
)


# --- Optional distributed tracing ------------------------------------------
# Importing the SDK and creating spans is always safe/cheap; nothing is
# actually exported anywhere unless OTEL_EXPORTER_OTLP_ENDPOINT is set to a
# real OpenTelemetry collector (self-hosted -- e.g. Jaeger/Tempo/Grafana --
# or a SaaS). Unset, this is a documented no-op, not a new requirement.
def _init_tracer():
    from opentelemetry import trace

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create(
                {
                    SERVICE_NAME: os.environ.get(
                        "OTEL_SERVICE_NAME", "avery-outbound-call-api"
                    )
                }
            )
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry tracing enabled, exporting to %s", endpoint)
    return trace.get_tracer("avery.outbound_call_api")


_tracer = _init_tracer()


class OutboundCallRequest(BaseModel):
    call_type: str = "outbound"
    user_id: str
    number: str
    prompt: str
    helper_prompt: str | None = None
    # Field name matches the API's request/response contract as given.
    # Accepts a short code ("bn"), a locale ("bn-BD") or an English name
    # ("bengali"); see LANGUAGES in languages.py for what's supported.
    langage: str = "en"
    # Exact opening line to speak the moment the callee answers, sent
    # straight to TTS with no LLM round trip. Optional, but recommended
    # whenever `prompt` is a custom persona: without it the agent has to
    # generate an opening line, which is what puts a second of dead air at
    # the front of the call. See DefaultAgent.resolve_greeting in agent.py.
    greeting: str | None = None


class CallStatus(BaseModel):
    call_id: str
    status: Literal["pending", "completed", "error", "timeout"]
    request: OutboundCallRequest
    response_summary: str | None = None
    concern_level: Literal["none", "watch", "urgent"] | None = None
    flagged_topics: list[str] = []
    error: str | None = None


class CallCompletedPayload(BaseModel):
    response_summary: str | None = None
    concern_level: Literal["none", "watch", "urgent"] | None = None
    flagged_topics: list[str] = []
    error: str | None = None


# call_id -> current status. In-memory and per-process: a restart of this
# service loses in-flight calls, and this only works with a single uvicorn
# worker process. Move to a shared store (e.g. Redis) if you need multiple
# API processes or restart resilience.
_calls: dict[str, CallStatus] = {}
_call_started_at: dict[str, float] = {}

# Holds references to the background timeout-watcher tasks below so they
# aren't garbage-collected mid-flight (asyncio only keeps a weak reference
# to a bare create_task() result).
_background_tasks: set[asyncio.Task] = set()

app = FastAPI(title="Avery Outbound Call API")


def _finish_call(
    call_id: str, status: Literal["completed", "error", "timeout"]
) -> None:
    started_at = _call_started_at.pop(call_id, None)
    if started_at is not None:
        _CALL_DURATION.observe(time.monotonic() - started_at)
    _CALLS_COMPLETED.labels(status=status).inc()


async def _watch_for_timeout(call_id: str) -> None:
    """Mark a call "timeout" if it's still "pending" after
    CALL_TIMEOUT_SECONDS -- otherwise a caller polling GET /calls/{call_id}
    could see "pending" forever if the agent worker crashes before ever
    posting back."""
    await asyncio.sleep(CALL_TIMEOUT_SECONDS)
    record = _calls.get(call_id)
    if record is not None and record.status == "pending":
        record.status = "timeout"
        record.error = (
            "call did not complete within the timeout: the agent worker never "
            "posted a result back. Check the worker logs for callback failures "
            "and that CALLBACK_BASE_URL is reachable from the worker."
        )
        logger.warning(
            "call %s timed out waiting for a callback",
            call_id,
            extra={"call_id": call_id},
        )
        _finish_call(call_id, "timeout")


@app.post("/calls/outbound", response_model=CallStatus, status_code=202)
async def create_outbound_call(body: OutboundCallRequest) -> CallStatus:
    if body.call_type != "outbound":
        raise HTTPException(400, f"unsupported call_type: {body.call_type!r}")

    # Reject an unknown language here rather than letting the worker fall back
    # to English. Silently running a Bengali wellbeing call in English is worse
    # than not placing it: the callee can't answer, and the summary that comes
    # back still reads like a normal call.
    if not is_supported_language(body.langage):
        raise HTTPException(
            400,
            f"unsupported langage: {body.langage!r} "
            f"(supported: {', '.join(sorted(LANGUAGES))})",
        )

    call_id = uuid.uuid4().hex
    record = CallStatus(call_id=call_id, status="pending", request=body)
    _calls[call_id] = record
    _call_started_at[call_id] = time.monotonic()

    room_name = f"outbound-{shortuuid()}"
    metadata = {
        "call_id": call_id,
        "phone_number": body.number,
        "user_id": body.user_id,
        "prompt": body.prompt,
        "helper_prompt": body.helper_prompt,
        "language": resolve_language(body.langage).code,
        "greeting": body.greeting,
        "callback_url": f"{CALLBACK_BASE_URL}/internal/calls/{call_id}/completed",
    }

    with _tracer.start_as_current_span(
        "dispatch_outbound_call",
        attributes={"call_id": call_id, "user_id": body.user_id},
    ):
        try:
            async with lk_api.LiveKitAPI() as lkapi:
                await lkapi.agent_dispatch.create_dispatch(
                    lk_api.CreateAgentDispatchRequest(
                        agent_name=AGENT_NAME,
                        room=room_name,
                        metadata=json.dumps(metadata),
                    )
                )
        except Exception as exc:
            record.status = "error"
            record.error = f"failed to dispatch call: {exc}"
            _call_started_at.pop(call_id, None)
            _CALLS_COMPLETED.labels(status="error").inc()
            logger.exception(
                "failed to dispatch call %s", call_id, extra={"call_id": call_id}
            )
            raise HTTPException(502, "failed to dispatch call") from exc

    _CALLS_CREATED.inc()
    logger.info(
        "dispatched outbound call %s to room %s",
        call_id,
        room_name,
        extra={"call_id": call_id},
    )
    watcher = asyncio.create_task(_watch_for_timeout(call_id))
    _background_tasks.add(watcher)
    watcher.add_done_callback(_background_tasks.discard)
    return record


@app.get("/calls/{call_id}", response_model=CallStatus)
async def get_call_status(call_id: str) -> CallStatus:
    record = _calls.get(call_id)
    if record is None:
        raise HTTPException(404, f"unknown call_id: {call_id!r}")
    return record


# include_in_schema=False: this is the worker's private callback, not part of
# the public contract. Listing it in /docs invites a "Try it out" with the
# placeholder body Swagger pre-fills -- and since that body carries
# error: "string", one click marks a live call errored and makes this service
# discard the real result when it arrives.
@app.post("/internal/calls/{call_id}/completed", include_in_schema=False)
async def call_completed(
    call_id: str,
    payload: CallCompletedPayload,
    x_callback_secret: str | None = Header(default=None),
) -> dict:
    if CALLBACK_SECRET is not None and not hmac.compare_digest(
        x_callback_secret or "", CALLBACK_SECRET
    ):
        logger.warning(
            "rejected callback for call %s: bad or missing secret",
            call_id,
            extra={"call_id": call_id},
        )
        raise HTTPException(401, "invalid callback secret")

    record = _calls.get(call_id)
    if record is None or record.status != "pending":
        # Unknown, late, or duplicate callback (for example, a retry after
        # this call already timed out). Ack so the agent worker doesn't
        # keep retrying.
        logger.info(
            "ignoring callback for unknown/finished call %s",
            call_id,
            extra={"call_id": call_id},
        )
        return {"ok": True}

    if payload.error:
        record.status = "error"
        record.error = payload.error
    else:
        record.status = "completed"
        record.response_summary = payload.response_summary or ""
        record.concern_level = payload.concern_level or "none"
        record.flagged_topics = payload.flagged_topics or []

    logger.info(
        "call %s reported %s (concern_level=%s)",
        call_id,
        record.status,
        record.concern_level,
        extra={"call_id": call_id},
    )
    _finish_call(call_id, record.status)
    return {"ok": True}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
    )
