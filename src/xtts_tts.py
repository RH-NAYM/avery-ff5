"""Local text-to-speech using Coqui XTTS v2.

Unlike the other TTS providers in this project (ElevenLabs, Gemini, Cartesia),
XTTS v2 runs entirely on the local machine (CPU or GPU) via the `coqui-tts`
package -- no API key or network call required at synthesis time. The model
is heavy (~1.8 GB) and is downloaded once to `~/.local/share/tts` on first
use.

See the "Coqui XTTS v2 (local)" section in README.md for installation and
setup instructions (conda environment, TOS acceptance, etc).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass

import numpy as np
from livekit.agents import APIConnectOptions, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid

logger = logging.getLogger("agent-Avery-ff5.xtts")

# Numba (used internally by librosa, a coqui-tts dependency) JIT-compiles many
# small functions on first use and, at DEBUG level, logs the full bytecode/IR
# for each one. That's irrelevant noise for this project, so it's silenced
# regardless of the root log level (LiveKit's `console` mode defaults to DEBUG).
logging.getLogger("numba").setLevel(logging.WARNING)

DEFAULT_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
# One of the built-in XTTS v2 speaker voices, used when no reference
# `speaker_wav` is supplied for voice cloning.
DEFAULT_SPEAKER = "Claribel Dervla"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# The underlying model is not safe for concurrent inference calls; this lock
# serializes synthesis across all XTTS instances/sessions in this process.
_inference_lock = threading.Lock()
_model_lock = threading.Lock()
_model = None
_model_key: tuple[str, str] | None = None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(model_name: str, device: str):
    """Lazily load (and cache) the XTTS model for this worker process."""
    global _model, _model_key

    key = (model_name, device)
    if _model is not None and _model_key == key:
        return _model

    with _model_lock:
        if _model is not None and _model_key == key:
            return _model

        # XTTS v2 is distributed under the Coqui Public Model License, which
        # requires accepting its terms before the model is downloaded/loaded.
        # See https://coqui.ai/cpml
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        import TTS.api as coqui_api

        logger.info(
            "loading XTTS model %s on %s (first run downloads ~1.8GB)",
            model_name,
            device,
        )
        _model = coqui_api.TTS(model_name).to(device)
        _model_key = key
        logger.info("XTTS model loaded")

    return _model


@dataclass
class _TTSOptions:
    model_name: str
    language: str
    speaker: str | None
    speaker_wav: str | None
    device: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        language: str = "en",
        speaker: str | None = None,
        speaker_wav: str | None = None,
        device: str = "auto",
    ) -> None:
        """Create a new Coqui XTTS v2 TTS instance.

        Exactly one voice source should be provided:
        - `speaker`: name of one of XTTS v2's built-in voices (defaults to
          "Claribel Dervla" if neither `speaker` nor `speaker_wav` is set).
        - `speaker_wav`: path to a short (~6-30s) reference clip to clone a
          voice instead of using a built-in one.

        `device` is "auto" (use CUDA if available, else CPU), "cuda", or
        "cpu". CPU inference is significantly slower than realtime.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        if speaker_wav and not os.path.isfile(speaker_wav):
            raise ValueError(f"speaker_wav file not found: {speaker_wav}")

        self._opts = _TTSOptions(
            model_name=model_name,
            language=language,
            speaker=speaker if not speaker_wav else None,
            speaker_wav=speaker_wav,
            device=_resolve_device(device),
        )
        if not self._opts.speaker and not self._opts.speaker_wav:
            self._opts.speaker = DEFAULT_SPEAKER

    @property
    def model(self) -> str:
        return self._opts.model_name

    @property
    def provider(self) -> str:
        return "coqui"

    def prewarm(self) -> None:
        """Load the model in a background thread so the first synthesis is fast."""

        def _prewarm() -> None:
            try:
                _get_model(self._opts.model_name, self._opts.device)
            except Exception:
                logger.exception("failed to prewarm XTTS model")

        threading.Thread(target=_prewarm, daemon=True).start()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    def _synthesize_sync(self) -> bytes:
        opts = self._tts._opts
        model = _get_model(opts.model_name, opts.device)

        with _inference_lock:
            samples = model.tts(
                text=self.input_text,
                language=opts.language,
                speaker=opts.speaker,
                speaker_wav=opts.speaker_wav,
            )

        pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        return (pcm * 32767.0).astype(np.int16).tobytes()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        pcm_bytes = await asyncio.to_thread(self._synthesize_sync)
        output_emitter.push(pcm_bytes)
