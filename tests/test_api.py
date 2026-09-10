"""Tests for the outbound call API (src/api.py).

These exercise the create -> dispatch -> callback -> poll cycle without a
real LiveKit dispatch or phone call: `LiveKitAPI.agent_dispatch.create_dispatch`
is faked to simulate what the agent worker does after a call finishes --
POSTing a structured summary to this API's own callback endpoint (the same
thing agent.py's on_session_end does in production; see CallSummary there).

POST /calls/outbound now returns immediately with status="pending" instead
of blocking until the call finishes (see the module docstring in api.py for
why), so these tests drive the callback and polling endpoints explicitly
instead of racing a background task against the initial request.
"""

import asyncio
from unittest.mock import patch

import httpx
import pytest

import api


class _FakeAgentDispatch:
    def __init__(self, on_dispatch):
        self._on_dispatch = on_dispatch

    async def create_dispatch(self, request):
        import json

        await self._on_dispatch(json.loads(request.metadata))
        return object()


class _FakeLiveKitAPI:
    def __init__(self, on_dispatch):
        self.agent_dispatch = _FakeAgentDispatch(on_dispatch)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FailingAgentDispatch:
    async def create_dispatch(self, request):
        raise RuntimeError("livekit is unreachable")


class _FailingLiveKitAPI:
    def __init__(self):
        self.agent_dispatch = _FailingAgentDispatch()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    )


REQUEST_BODY = {
    "call_type": "outbound",
    "user_id": "user-1",
    "number": "+15105550100",
    "prompt": "full prompt is here",
    "helper_prompt": "extra info",
    "langage": "en",
}


@pytest.mark.asyncio
async def test_outbound_call_returns_immediately_with_pending_status() -> None:
    captured_metadata: dict = {}

    async def on_dispatch(metadata: dict) -> None:
        captured_metadata.update(metadata)

    with patch.object(api.lk_api, "LiveKitAPI", lambda: _FakeLiveKitAPI(on_dispatch)):
        async with _client() as client:
            response = await client.post("/calls/outbound", json=REQUEST_BODY)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["call_id"]
    assert body["request"]["number"] == REQUEST_BODY["number"]

    # The dispatch itself happens synchronously before the response returns.
    assert captured_metadata["phone_number"] == REQUEST_BODY["number"]
    assert captured_metadata["user_id"] == REQUEST_BODY["user_id"]
    assert captured_metadata["prompt"] == REQUEST_BODY["prompt"]
    assert captured_metadata["helper_prompt"] == REQUEST_BODY["helper_prompt"]
    assert captured_metadata["language"] == REQUEST_BODY["langage"]
    assert captured_metadata["call_id"] == body["call_id"]
    assert captured_metadata["callback_url"].endswith(
        f"/internal/calls/{body['call_id']}/completed"
    )


@pytest.mark.asyncio
async def test_call_completes_via_callback_with_structured_summary() -> None:
    async def on_dispatch(metadata: dict) -> None:
        return None

    with patch.object(api.lk_api, "LiveKitAPI", lambda: _FakeLiveKitAPI(on_dispatch)):
        async with _client() as client:
            create_response = await client.post("/calls/outbound", json=REQUEST_BODY)
            call_id = create_response.json()["call_id"]

            callback_response = await client.post(
                f"/internal/calls/{call_id}/completed",
                json={
                    "response_summary": "The caller said they're doing well.",
                    "concern_level": "watch",
                    "flagged_topics": ["slept poorly"],
                },
            )
            assert callback_response.status_code == 200
            assert callback_response.json() == {"ok": True}

            status_response = await client.get(f"/calls/{call_id}")

    body = status_response.json()
    assert body["status"] == "completed"
    assert body["response_summary"] == "The caller said they're doing well."
    assert body["concern_level"] == "watch"
    assert body["flagged_topics"] == ["slept poorly"]


@pytest.mark.asyncio
async def test_sip_failure_is_reported_as_error_status() -> None:
    async def on_dispatch(metadata: dict) -> None:
        return None

    with patch.object(api.lk_api, "LiveKitAPI", lambda: _FakeLiveKitAPI(on_dispatch)):
        async with _client() as client:
            create_response = await client.post("/calls/outbound", json=REQUEST_BODY)
            call_id = create_response.json()["call_id"]

            await client.post(
                f"/internal/calls/{call_id}/completed",
                json={"error": "sip call failed: 486 Busy Here"},
            )
            status_response = await client.get(f"/calls/{call_id}")

    body = status_response.json()
    assert body["status"] == "error"
    assert "486 Busy Here" in body["error"]


@pytest.mark.asyncio
async def test_dispatch_failure_returns_502() -> None:
    with patch.object(api.lk_api, "LiveKitAPI", _FailingLiveKitAPI):
        async with _client() as client:
            response = await client.post("/calls/outbound", json=REQUEST_BODY)

    assert response.status_code == 502


@pytest.mark.asyncio
async def test_call_times_out_if_no_callback_arrives() -> None:
    async def on_dispatch(metadata: dict) -> None:
        return None

    with (
        patch.object(api.lk_api, "LiveKitAPI", lambda: _FakeLiveKitAPI(on_dispatch)),
        patch.object(api, "CALL_TIMEOUT_SECONDS", 0.05),
    ):
        async with _client() as client:
            create_response = await client.post("/calls/outbound", json=REQUEST_BODY)
            call_id = create_response.json()["call_id"]

            await asyncio.sleep(0.2)  # let the background timeout watcher fire

            status_response = await client.get(f"/calls/{call_id}")

    body = status_response.json()
    assert body["status"] == "timeout"


@pytest.mark.asyncio
async def test_late_callback_for_unknown_call_is_acknowledged_not_errored() -> None:
    async with _client() as client:
        response = await client.post(
            "/internal/calls/nonexistent-call-id/completed",
            json={"response_summary": "too late"},
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.asyncio
async def test_get_call_status_for_unknown_call_returns_404() -> None:
    async with _client() as client:
        response = await client.get("/calls/nonexistent-call-id")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_prometheus_text() -> None:
    async with _client() as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert "avery_calls_created_total" in response.text
    assert "avery_calls_completed_total" in response.text


@pytest.mark.asyncio
async def test_supported_language_is_dispatched_with_a_normalized_code() -> None:
    captured: dict = {}

    async def on_dispatch(metadata: dict) -> None:
        captured.update(metadata)

    with patch.object(api.lk_api, "LiveKitAPI", lambda: _FakeLiveKitAPI(on_dispatch)):
        async with _client() as client:
            response = await client.post(
                "/calls/outbound", json={**REQUEST_BODY, "langage": "bn-BD"}
            )

    assert response.status_code == 202
    # The worker gets the canonical code, not whatever spelling the caller used.
    assert captured["language"] == "bn"


@pytest.mark.asyncio
async def test_unsupported_language_is_rejected_rather_than_run_in_english() -> None:
    # Silently downgrading a call to English is worse than refusing it: the
    # callee can't answer, and the summary still comes back looking normal.
    async def on_dispatch(metadata: dict) -> None:
        raise AssertionError("should not dispatch an unsupported language")

    with patch.object(api.lk_api, "LiveKitAPI", lambda: _FakeLiveKitAPI(on_dispatch)):
        async with _client() as client:
            response = await client.post(
                "/calls/outbound", json={**REQUEST_BODY, "langage": "klingon"}
            )

    assert response.status_code == 400
    assert "klingon" in response.json()["detail"]


@pytest.mark.asyncio
async def test_internal_callback_is_not_advertised_in_the_public_schema() -> None:
    # It used to be. Swagger pre-fills a str field with "string", so one
    # "Try it out" on this endpoint marked a live call errored and made the
    # agent's real result get dropped as a duplicate.
    async with _client() as client:
        schema = (await client.get("/openapi.json")).json()

    assert "/calls/outbound" in schema["paths"]
    assert not [p for p in schema["paths"] if p.startswith("/internal/")]


@pytest.mark.asyncio
async def test_callback_without_the_secret_is_rejected_when_one_is_set() -> None:
    async def on_dispatch(metadata: dict) -> None:
        return None

    with (
        patch.object(api.lk_api, "LiveKitAPI", lambda: _FakeLiveKitAPI(on_dispatch)),
        patch.object(api, "CALLBACK_SECRET", "s3cret"),
    ):
        async with _client() as client:
            call_id = (await client.post("/calls/outbound", json=REQUEST_BODY)).json()[
                "call_id"
            ]

            rejected = await client.post(
                f"/internal/calls/{call_id}/completed", json={"error": "string"}
            )
            assert rejected.status_code == 401

            # The call is untouched, so the worker's real result still lands.
            assert (await client.get(f"/calls/{call_id}")).json()["status"] == "pending"

            accepted = await client.post(
                f"/internal/calls/{call_id}/completed",
                json={"response_summary": "all well", "concern_level": "none"},
                headers={"X-Callback-Secret": "s3cret"},
            )
            assert accepted.status_code == 200

            body = (await client.get(f"/calls/{call_id}")).json()

    assert body["status"] == "completed"
    assert body["response_summary"] == "all well"
