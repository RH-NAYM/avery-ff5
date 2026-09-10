"""Tests for the STT stall-recovery layer (src/stt_recovery.py).

These drive the wrapper against a fake recognizer that reproduces, exactly,
the two shapes of the failure that has cost this project live calls: a
recognizer that commits an empty transcript, and one that goes quiet
mid-utterance and never comes back. Both are silent -- neither raises, and
neither is visible to stt.FallbackAdapter -- so a test that only checked "no
exception" would pass against the bug itself.
"""

from __future__ import annotations

import asyncio

import pytest
from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, stt
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType, STTCapabilities

from stt_recovery import StallRecoveringSTT

SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = SAMPLE_RATE // 100  # 10ms


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * SAMPLES_PER_FRAME,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=SAMPLES_PER_FRAME,
    )


class _ScriptedStream(stt.RecognizeStream):
    """Emits a fixed list of events, then does whatever `then` says."""

    def __init__(self, *, stt_, conn_options, script, then):
        super().__init__(stt=stt_, conn_options=conn_options)
        self._script = script
        self._then = then

    async def _run(self) -> None:
        # Wait until audio is actually flowing before saying anything, so the
        # wrapper has something buffered to recover with.
        drained = 0
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                continue
            drained += 1
            if drained >= 20:  # 200ms of audio
                break

        for ev in self._script:
            self._event_ch.send_nowait(ev)

        if self._then == "hang":
            # The live failure: the socket stays open, the caller keeps
            # talking, and nothing else is ever emitted.
            await asyncio.sleep(3600)
        # "end" falls through, closing the stream normally.


class _ScriptedSTT(stt.STT):
    def __init__(self, *, script, then="end"):
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=True)
        )
        self._script = script
        self._then = then
        self.streams_opened = 0

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
        raise NotImplementedError

    def stream(self, *, language=None, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        self.streams_opened += 1
        return _ScriptedStream(
            stt_=self,
            conn_options=conn_options,
            script=self._script,
            then=self._then,
        )


class _BatchSTT(stt.STT):
    """Stands in for Google Cloud STT's batch recognize()."""

    def __init__(self, text: str = "আমি ভালো আছি"):
        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=False)
        )
        self._text = text
        self.calls = 0
        self.seconds_received = 0.0

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
        self.calls += 1
        frames = buffer if isinstance(buffer, list) else [buffer]
        self.seconds_received = sum(f.duration for f in frames)
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(language="bn", text=self._text)],
        )


async def _drive(wrapper: StallRecoveringSTT, *, seconds: float, collect_for: float):
    """Push `seconds` of audio, then collect events for `collect_for`."""
    stream = wrapper.stream()
    events: list[SpeechEvent] = []

    async def collect():
        async for ev in stream:
            events.append(ev)

    collector = asyncio.create_task(collect())

    async def push():
        for _ in range(int(seconds * 100)):
            stream.push_frame(_frame())
            await asyncio.sleep(0.001)

    pusher = asyncio.create_task(push())
    await asyncio.sleep(collect_for)
    pusher.cancel()
    collector.cancel()
    await asyncio.gather(pusher, collector, return_exceptions=True)
    await stream.aclose()
    return events


def _finals(events):
    return [
        ev.alternatives[0].text
        for ev in events
        if ev.type is SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives
    ]


@pytest.mark.asyncio
async def test_an_empty_commit_is_recovered_by_the_fallback_recognizer():
    """The ElevenLabs shape: START_OF_SPEECH, a partial, then an END_OF_SPEECH
    with no transcript attached. The plugin emits FINAL_TRANSCRIPT only `if
    text:`, so an empty commit reaches the session as an end-of-speech alone
    and the framework never commits a turn."""
    primary = _ScriptedSTT(
        script=[
            SpeechEvent(type=SpeechEventType.START_OF_SPEECH),
            SpeechEvent(
                type=SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[SpeechData(language="bn", text="আমি")],
            ),
            SpeechEvent(type=SpeechEventType.END_OF_SPEECH),
        ],
        then="hang",
    )
    recovery = _BatchSTT()
    wrapper = StallRecoveringSTT(primary=primary, recovery=recovery, stall_timeout=30.0)

    events = await _drive(wrapper, seconds=1.0, collect_for=1.5)

    assert recovery.calls == 1, "the fallback recognizer was never asked"
    assert _finals(events) == ["আমি ভালো আছি"]
    # Order matters: the framework's _run_eou_detection returns early while
    # there is no transcript, so the recovered final has to land *before* the
    # end-of-speech that triggers the turn commit.
    types = [ev.type for ev in events]
    assert types.index(SpeechEventType.FINAL_TRANSCRIPT) < types.index(
        SpeechEventType.END_OF_SPEECH
    )
    # An empty commit means the connection is healthy; recycling it would
    # throw away a working socket for nothing.
    assert primary.streams_opened == 1


@pytest.mark.asyncio
async def test_a_silent_stall_is_recovered_and_the_connection_recycled():
    """The measured shape: a 7-character partial at 17:30:24 and then nothing
    at all. No error, no end-of-speech, no commit -- the socket is wedged."""
    primary = _ScriptedSTT(
        script=[
            SpeechEvent(type=SpeechEventType.START_OF_SPEECH),
            SpeechEvent(
                type=SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[SpeechData(language="bn", text="আমি")],
            ),
        ],
        then="hang",
    )
    recovery = _BatchSTT()
    wrapper = StallRecoveringSTT(primary=primary, recovery=recovery, stall_timeout=0.5)

    events = await _drive(wrapper, seconds=2.0, collect_for=2.0)

    assert recovery.calls == 1
    assert _finals(events) == ["আমি ভালো আছি"]
    # A wedged socket stays wedged: it has to be torn down and reopened, or
    # every later turn in the call is lost the same way.
    assert primary.streams_opened >= 2
    # ...and the recovered turn must be closed out, since no END_OF_SPEECH is
    # ever coming from the connection that stalled.
    assert SpeechEventType.END_OF_SPEECH in [ev.type for ev in events]


@pytest.mark.asyncio
async def test_a_healthy_final_transcript_is_passed_through_untouched():
    primary = _ScriptedSTT(
        script=[
            SpeechEvent(type=SpeechEventType.START_OF_SPEECH),
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(language="bn", text="হ্যাঁ শুনছি")],
            ),
            SpeechEvent(type=SpeechEventType.END_OF_SPEECH),
        ],
        then="hang",
    )
    recovery = _BatchSTT()
    wrapper = StallRecoveringSTT(primary=primary, recovery=recovery, stall_timeout=0.5)

    events = await _drive(wrapper, seconds=1.5, collect_for=1.5)

    assert _finals(events) == ["হ্যাঁ শুনছি"]
    # The whole point: recovery costs a second recognition request, so it must
    # not fire on a turn the primary handled.
    assert recovery.calls == 0
    assert primary.streams_opened == 1


@pytest.mark.asyncio
async def test_silence_is_not_invented_into_a_turn():
    """Both recognizers hearing nothing in the same audio is much more likely
    to be real silence than two independent failures -- and the agent staying
    quiet is then the correct behaviour, not the bug."""
    primary = _ScriptedSTT(
        script=[
            SpeechEvent(type=SpeechEventType.START_OF_SPEECH),
            SpeechEvent(
                type=SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[SpeechData(language="bn", text="…")],
            ),
            SpeechEvent(type=SpeechEventType.END_OF_SPEECH),
        ],
        then="hang",
    )
    recovery = _BatchSTT(text="   ")
    wrapper = StallRecoveringSTT(primary=primary, recovery=recovery, stall_timeout=30.0)

    events = await _drive(wrapper, seconds=1.0, collect_for=1.5)

    assert recovery.calls == 1
    assert _finals(events) == []


@pytest.mark.asyncio
async def test_a_failing_fallback_never_takes_the_call_down_with_it():
    class _BrokenSTT(_BatchSTT):
        async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
            self.calls += 1
            raise RuntimeError("fallback provider is having a bad day")

    primary = _ScriptedSTT(
        script=[
            SpeechEvent(type=SpeechEventType.START_OF_SPEECH),
            SpeechEvent(
                type=SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[SpeechData(language="bn", text="আমি")],
            ),
            SpeechEvent(type=SpeechEventType.END_OF_SPEECH),
        ],
        then="hang",
    )
    recovery = _BrokenSTT()
    wrapper = StallRecoveringSTT(primary=primary, recovery=recovery, stall_timeout=30.0)

    events = await _drive(wrapper, seconds=1.0, collect_for=1.5)

    assert recovery.calls >= 1
    assert _finals(events) == []
    # The turn is lost, which is exactly where we started -- but the call
    # itself keeps running, and the stream is still usable.
    assert SpeechEventType.END_OF_SPEECH in [ev.type for ev in events]


@pytest.mark.asyncio
async def test_only_the_current_utterance_is_sent_to_the_fallback():
    """Audio is retained from the moment speech is detected, not for the whole
    call: a rolling buffer of everything would ship tens of seconds of silence
    and agent echo to the fallback on every recovery."""
    primary = _ScriptedSTT(
        script=[
            SpeechEvent(type=SpeechEventType.START_OF_SPEECH),
            SpeechEvent(type=SpeechEventType.END_OF_SPEECH),
        ],
        then="hang",
    )
    recovery = _BatchSTT()
    wrapper = StallRecoveringSTT(
        primary=primary,
        recovery=recovery,
        stall_timeout=30.0,
        preroll_seconds=0.2,
    )

    await _drive(wrapper, seconds=2.0, collect_for=1.5)

    assert recovery.calls == 1
    # The scripted stream waits for 200ms of audio before emitting anything,
    # so everything before that is pre-roll and is capped at 0.2s.
    assert recovery.seconds_received <= 1.0


def test_a_non_streaming_primary_is_rejected():
    """A StreamAdapter-wrapped recognizer is cut into utterances by the
    session's VAD and cannot stall this way, so wrapping one would add cost
    and a second failure mode for no benefit."""
    with pytest.raises(ValueError, match="streaming"):
        StallRecoveringSTT(primary=_BatchSTT(), recovery=_BatchSTT())
