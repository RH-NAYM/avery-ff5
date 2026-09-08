"""Tests for the outbound call API (src/api.py).

These exercise the full request/callback/response cycle without a real
LiveKit dispatch or phone call: `LiveKitAPI.agent_dispatch.create_dispatch`
is faked to simulate what the agent worker does after a call finishes --
POSTing the summary to this API's own callback endpoint (the same thing
agent.py's on_session_end does in production).
"""

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest

import api


class _FakeAgentDispatch:
    def __init__(self, on_dispatch):
        self._on_dispatch = on_dispatch

    async def create_dispatch(self, request):
        await self._on_dispatch(json.loads(request.metadata))
        return object()


class _FakeLiveKitAPI:
    def __init__(self, on_dispatch):
        self.agent_dispatch = _FakeAgentDispatch(on_dispatch)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    )


# Held so the fire-and-forget callback tasks below aren't garbage-collected
# mid-flight (asyncio only holds a weak reference to a bare create_task()).
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


REQUEST_BODY = {
    "call_type": "outbound",
    "user_id": "user-1",
    "number": "+15105550100",
    "prompt": "full prompt is here",
    "helper_prompt": "extra info",
    "langage": "en",
}


@pytest.mark.asyncio
async def test_outbound_call_returns_summary_once_agent_calls_back() -> None:
    async def on_dispatch(metadata: dict) -> None:
        assert metadata["phone_number"] == REQUEST_BODY["number"]
        assert metadata["user_id"] == REQUEST_BODY["user_id"]
        assert metadata["prompt"] == REQUEST_BODY["prompt"]
        assert metadata["helper_prompt"] == REQUEST_BODY["helper_prompt"]
        assert metadata["language"] == REQUEST_BODY["langage"]
        assert metadata["call_id"]
        assert metadata["callback_url"].endswith(
            f"/internal/calls/{metadata['call_id']}/completed"
        )

        async def send_callback() -> None:
            async with _client() as client:
                response = await client.post(
                    metadata["callback_url"],
                    json={"response_summary": "The caller said they're doing well."},
                )
                assert response.status_code == 200

        _spawn(send_callback())

    with patch.object(api.lk_api, "LiveKitAPI", lambda: _FakeLiveKitAPI(on_dispatch)):
        async with _client() as client:
            response = await client.post("/calls/outbound", json=REQUEST_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body == {
        **REQUEST_BODY,
        "response_summary": "The caller said they're doing well.",
    }
    # The future is cleaned up once resolved.
    assert not api._pending_calls


@pytest.mark.asyncio
async def test_outbound_call_reports_sip_failure_as_summary_text() -> None:
    async def on_dispatch(metadata: dict) -> None:
        async def send_callback() -> None:
            async with _client() as client:
                await client.post(
                    metadata["callback_url"],
                    json={"error": "sip call failed: 486 Busy Here"},
                )

        _spawn(send_callback())

    with patch.object(api.lk_api, "LiveKitAPI", lambda: _FakeLiveKitAPI(on_dispatch)):
        async with _client() as client:
            response = await client.post("/calls/outbound", json=REQUEST_BODY)

    assert response.status_code == 200
    assert "486 Busy Here" in response.json()["response_summary"]


@pytest.mark.asyncio
async def test_outbound_call_times_out_if_no_callback_arrives() -> None:
    async def on_dispatch(metadata: dict) -> None:
        pass  # simulate the agent worker never calling back

    with (
        patch.object(api.lk_api, "LiveKitAPI", lambda: _FakeLiveKitAPI(on_dispatch)),
        patch.object(api, "CALL_TIMEOUT_SECONDS", 0.2),
    ):
        async with _client() as client:
            response = await client.post("/calls/outbound", json=REQUEST_BODY)

    assert response.status_code == 504
    assert not api._pending_calls


@pytest.mark.asyncio
async def test_late_callback_for_unknown_call_is_acknowledged_not_errored() -> None:
    async with _client() as client:
        response = await client.post(
            "/internal/calls/nonexistent-call-id/completed",
            json={"response_summary": "too late"},
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
