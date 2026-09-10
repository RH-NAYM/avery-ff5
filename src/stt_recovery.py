"""Recover a user turn when the recognizer silently produces nothing.

This module exists because of a failure that has now cost this project live
calls across three STT providers, and looks identical from the outside every
time: the caller talks, the agent says nothing, and the log is clean.

It is silent by construction, not by accident:

1. A streaming recognizer that finalizes an utterance as *empty* reaches the
   session as an end-of-speech with no transcript attached. The ElevenLabs
   plugin emits a FINAL_TRANSCRIPT only `if text:` -- an empty commit falls
   through to a bare END_OF_SPEECH.
2. `AudioRecognition._run_eou_detection` returns immediately while
   `self._stt and not self._audio_transcript`. No transcript means no turn
   commit, ever -- VAD hearing end-of-speech does nothing on its own.
3. Nothing in that chain raises, so no retry, no fallback and no error log
   is triggered anywhere in the framework or the plugin.

`stt.FallbackAdapter` does not help, because it fails over on *errors* and
this failure produces none. `AgentSession(transcription_timeout=...)` does not
help either: it emits a `user_transcription_timeout` event carrying only a
duration, with no access to the audio, so it can report the dropped turn but
cannot recover it.

Recovery has to live at the recognizer layer, which is the only place that
still has the audio. `StallRecoveringSTT` wraps a streaming STT, keeps the
current utterance's audio in memory, and watches for the two shapes this
failure takes:

* **Empty commit** -- an END_OF_SPEECH arrives while an utterance is still
  waiting for its transcript. The connection is healthy; the recognizer just
  decided the audio was nothing.
* **Silent stall** -- no event of any kind arrives for `stall_timeout` while
  the caller is mid-utterance. Measured on a live Bengali call: a 7-character
  partial, then nothing for ~16 seconds. The socket is wedged, so it is torn
  down and reopened as well.

In both cases the buffered audio is re-recognized by a second recognizer
(`recovery=`, a batch/offline STT -- a different code path and, in this
project's wiring, a different vendor) and the resulting transcript is emitted
into the stream as a normal FINAL_TRANSCRIPT. The framework cannot tell it
apart from one the primary produced, so the turn commits and the agent
replies, roughly a second late instead of never.

The wrapper is deliberately provider-agnostic: it uses only the public
`stt.STT` / `stt.RecognizeStream` surface, so it works over a single plugin or
over a `stt.FallbackAdapter` composed of several.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEvent, SpeechEventType, STTCapabilities

logger = logging.getLogger("agent-Avery-ff5")


# How long the primary can go without producing *any* event, while the caller
# is known to be mid-utterance, before the connection is treated as wedged.
# The floor under this is the primary's own commit latency: ElevenLabs'
# server VAD finalizes at vad_silence_threshold_secs + min_silence_duration_ms
# (0.7s + 400ms with this project's settings), so 3s is roughly three times
# the longest a healthy connection should ever stay quiet mid-turn.
DEFAULT_STALL_TIMEOUT = 3.0

# Cap on the audio kept for a single utterance. Elderly callers ramble, and
# the point of the cap is bounding memory and the recovery request's cost, not
# cutting anyone off -- 30s is far past any turn this agent expects.
DEFAULT_MAX_UTTERANCE_SECONDS = 30.0

# Audio kept from just *before* speech was detected, so a recovered transcript
# still contains the first word's onset. Mirrors the session VAD's own
# prefix_padding_duration.
DEFAULT_PREROLL_SECONDS = 1.0

# A recovery that takes longer than this is worse than useless: the caller has
# already waited out the stall, and a transcript arriving 15s late lands in the
# wrong part of the conversation.
DEFAULT_RECOVERY_TIMEOUT = 8.0


def _text_of(ev: SpeechEvent) -> str:
    if not ev.alternatives:
        return ""
    return (ev.alternatives[0].text or "").strip()


class _RollingAudio:
    """A duration-capped FIFO of audio frames."""

    def __init__(self, max_seconds: float) -> None:
        self._max_seconds = max_seconds
        self._frames: deque[rtc.AudioFrame] = deque()
        self._seconds = 0.0

    @property
    def seconds(self) -> float:
        return self._seconds

    def __len__(self) -> int:
        return len(self._frames)

    def append(self, frame: rtc.AudioFrame) -> None:
        self._frames.append(frame)
        self._seconds += frame.duration
        while len(self._frames) > 1 and self._seconds > self._max_seconds:
            self._seconds -= self._frames.popleft().duration

    def extend(self, other: _RollingAudio) -> None:
        for frame in other._frames:
            self.append(frame)

    def frames(self) -> list[rtc.AudioFrame]:
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()
        self._seconds = 0.0


class StallRecoveringSTT(stt.STT):
    """Wrap a streaming STT so a silently dropped utterance is re-recognized
    by a second recognizer instead of being lost.

    Args:
        primary: the streaming recognizer that actually serves the call. May
            itself be a `stt.FallbackAdapter`, which handles the *error* case;
            this wrapper handles the *silent* case, which the adapter cannot
            see.
        recovery: the recognizer used to re-transcribe a dropped utterance.
            Only its batch `recognize()` path is used, so a non-streaming STT
            is fine here -- and preferable, since the whole point is to not
            share a code path (or ideally a vendor) with the thing that just
            stalled.
        stall_timeout: seconds of total silence from `primary`, while an
            utterance is pending, before the connection is recycled and the
            audio re-recognized.
        on_recovery: optional callback, `(reason, text) -> None`, invoked after
            a successful recovery. Used by agent.py to log with the call id.
    """

    def __init__(
        self,
        *,
        primary: stt.STT,
        recovery: stt.STT,
        stall_timeout: float = DEFAULT_STALL_TIMEOUT,
        max_utterance_seconds: float = DEFAULT_MAX_UTTERANCE_SECONDS,
        preroll_seconds: float = DEFAULT_PREROLL_SECONDS,
        recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT,
        on_recovery: Callable[[str, str], None] | None = None,
    ) -> None:
        if not primary.capabilities.streaming:
            raise ValueError(
                "StallRecoveringSTT wraps a streaming recognizer; a non-streaming "
                "one is already segmented by stt.StreamAdapter and cannot stall "
                "this way. Wrap it in StreamAdapter (or a FallbackAdapter with a "
                "vad=) before passing it here."
            )
        if not recovery.capabilities.offline_recognize:
            raise ValueError(
                "the recovery STT must support batch recognize(); recovery works "
                "by re-recognizing buffered audio, not by opening a second stream."
            )

        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=primary.capabilities.interim_results,
                diarization=primary.capabilities.diarization,
                aligned_transcript=primary.capabilities.aligned_transcript,
                keyterms=primary.capabilities.keyterms,
                chat_context=primary.capabilities.chat_context,
            )
        )
        self._primary = primary
        self._recovery = recovery
        self._stall_timeout = stall_timeout
        self._max_utterance_seconds = max_utterance_seconds
        self._preroll_seconds = preroll_seconds
        self._recovery_timeout = recovery_timeout
        self._on_recovery = on_recovery

    @property
    def model(self) -> str:
        return self._primary.model

    @property
    def provider(self) -> str:
        return self._primary.provider

    @property
    def primary(self) -> stt.STT:
        return self._primary

    @property
    def recovery(self) -> stt.STT:
        return self._recovery

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._primary._update_session_keyterms(keyterms)

    def _push_conversation_item(self, ev) -> None:
        self._primary._push_conversation_item(ev)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        # Batch recognition has no stall to recover from: it either returns a
        # result or raises, and both are already handled upstream.
        return await self._primary.recognize(
            buffer, language=language, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _StallRecoveringStream:
        return _StallRecoveringStream(
            stt=self, language=language, conn_options=conn_options
        )

    def prewarm(self) -> None:
        self._primary.prewarm()

    async def aclose(self) -> None:
        await self._primary.aclose()
        await self._recovery.aclose()


class _StallRecoveringStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: StallRecoveringSTT,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._parent = stt
        self._language = language

        # Audio is only retained while an utterance is actually in flight;
        # a rolling pre-roll covers the moment before the first partial.
        self._preroll = _RollingAudio(stt._preroll_seconds)
        self._utterance = _RollingAudio(stt._max_utterance_seconds)

        self._pending = False
        """The caller is mid-utterance and no transcript has arrived for it."""
        self._speaking = False
        """A START_OF_SPEECH has been forwarded and not yet closed."""

        self._last_event_at = time.monotonic()
        self._recycle = asyncio.Event()
        self._recycle_reason = ""

    # -- audio bookkeeping ---------------------------------------------------

    def _remember(self, frame: rtc.AudioFrame) -> None:
        if self._pending:
            self._utterance.append(frame)
        else:
            self._preroll.append(frame)

    def _begin_utterance(self) -> None:
        if self._pending:
            return
        self._pending = True
        self._utterance.clear()
        self._utterance.extend(self._preroll)
        self._preroll.clear()

    def _end_utterance(self) -> None:
        self._pending = False
        self._utterance.clear()

    # -- tasks ---------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            inner = self._parent._primary.stream(
                language=self._language, conn_options=self._conn_options
            )
            self._recycle.clear()
            self._last_event_at = time.monotonic()

            forward = asyncio.create_task(self._forward_task(inner))
            receive = asyncio.create_task(self._receive_task(inner))
            watchdog = asyncio.create_task(self._watchdog_task())
            group = asyncio.gather(forward, receive, watchdog)
            recycle_wait = asyncio.create_task(self._recycle.wait())

            try:
                done, _ = await asyncio.wait(
                    (group, recycle_wait), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if task is not recycle_wait:
                        task.result()  # re-raise whatever the inner stream hit
                if recycle_wait not in done:
                    return  # input ended and the inner stream drained cleanly
            finally:
                await utils.aio.cancel_and_wait(
                    forward, receive, watchdog, recycle_wait
                )
                group.cancel()
                with contextlib.suppress(BaseException):
                    group.exception()
                await inner.aclose()

            # Only reached when the watchdog asked for a recycle. The socket is
            # gone now, which also means any transcript it was sitting on is
            # gone -- so there is no risk of the recovered turn being duplicated
            # by a late commit from the connection that stalled.
            await self._recover(self._recycle_reason, emit_end_of_speech=True)

    async def _forward_task(self, inner: stt.RecognizeStream) -> None:
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                inner.flush()
                continue
            self._remember(data)
            inner.push_frame(data)
        inner.end_input()

    async def _receive_task(self, inner: stt.RecognizeStream) -> None:
        async for ev in inner:
            self._last_event_at = time.monotonic()

            if ev.type is SpeechEventType.START_OF_SPEECH:
                self._begin_utterance()
                self._speaking = True
            elif ev.type in (
                SpeechEventType.INTERIM_TRANSCRIPT,
                SpeechEventType.PREFLIGHT_TRANSCRIPT,
            ):
                self._begin_utterance()
            elif ev.type is SpeechEventType.FINAL_TRANSCRIPT:
                if _text_of(ev):
                    self._end_utterance()
            elif ev.type is SpeechEventType.END_OF_SPEECH:
                if self._pending:
                    # The recognizer finalized this utterance as empty. The
                    # connection is fine, so re-recognize without recycling it
                    # -- and do it *before* forwarding the end-of-speech, so
                    # the framework sees the transcript first and can commit
                    # the turn on it.
                    await self._recover(
                        "the recognizer committed an empty transcript",
                        emit_end_of_speech=False,
                    )
                self._speaking = False

            self._event_ch.send_nowait(ev)

    async def _watchdog_task(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            if not self._pending or len(self._utterance) == 0:
                continue
            idle = time.monotonic() - self._last_event_at
            if idle < self._parent._stall_timeout:
                continue
            self._recycle_reason = (
                f"the recognizer returned nothing for {idle:.1f}s while the "
                f"caller was mid-sentence"
            )
            self._recycle.set()
            return

    # -- recovery ------------------------------------------------------------

    async def _recover(self, reason: str, *, emit_end_of_speech: bool) -> bool:
        frames = self._utterance.frames()
        seconds = self._utterance.seconds
        self._end_utterance()
        if not frames:
            return False

        parent = self._parent
        logger.warning(
            "STT stall: %s. Re-recognizing %.1fs of buffered audio with %s.",
            reason,
            seconds,
            parent._recovery.label,
        )
        started = time.monotonic()
        try:
            ev = await asyncio.wait_for(
                parent._recovery.recognize(frames),
                timeout=parent._recovery_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                "STT stall recovery timed out after %.1fs; this turn is lost.",
                parent._recovery_timeout,
            )
            return False
        except Exception:
            logger.exception("STT stall recovery failed; this turn is lost.")
            return False

        text = _text_of(ev)
        if not text:
            # Both recognizers heard nothing in the same audio, which is much
            # more likely to be true silence (or line noise) than two
            # independent failures. Nothing to recover, and saying so matters:
            # it is the one case where the agent staying quiet is correct.
            logger.info(
                "STT stall recovery found no speech in %.1fs of audio; "
                "treating the turn as silence.",
                seconds,
            )
            return False

        if not self._speaking:
            self._event_ch.send_nowait(
                SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
            )
            self._speaking = True
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=ev.alternatives
            )
        )
        if emit_end_of_speech:
            self._event_ch.send_nowait(SpeechEvent(type=SpeechEventType.END_OF_SPEECH))
            self._speaking = False

        logger.warning(
            "STT stall recovered in %.2fs: %d characters from %s. The turn was "
            "saved, but the primary recognizer dropped it -- if this repeats "
            "every few turns, the primary is the problem, not the recovery.",
            time.monotonic() - started,
            len(text),
            parent._recovery.label,
        )
        if parent._on_recovery is not None:
            with contextlib.suppress(Exception):
                parent._on_recovery(reason, text)
        return True
