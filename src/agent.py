import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Literal

import httpx
from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    ChatContext,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import (
    ai_coustics,
    cartesia,
    elevenlabs,
    google,
    openai,
    silero,
)
from livekit.plugins.google.beta import GeminiSTT, GeminiTTS

from languages import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    TURN_DETECTOR_LANGUAGES,
    LanguageProfile,
    resolve_language,
)
from logging_utils import configure_logging

# Structured (JSON) logs when LOG_FORMAT=json is set in the environment
# (e.g. a production deployment); plain text otherwise, including LiveKit's
# own colored `console`/`dev` CLI output, which this leaves untouched.
configure_logging()
logger = logging.getLogger("agent-Avery-ff5")

load_dotenv(".env.local")  # local dev config, per README (git-ignored)
load_dotenv(".env")  # optional fallback for anything not in .env.local


def _materialize_google_credentials() -> None:
    """Allow the Google service-account key to be supplied as inline JSON.

    Locally, GOOGLE_APPLICATION_CREDENTIALS points at a real key file on
    disk (see .env.local). LiveKit Cloud's agent secrets, however, are only
    ever plain strings mounted as environment variables -- there's no way
    to upload the key file itself, and the file is git-ignored so it never
    reaches the deployed container's build context.

    So in production, set a GOOGLE_CREDENTIALS_JSON secret containing the
    *full contents* of the service-account JSON key file instead. If it's
    present and no real file already exists at GOOGLE_APPLICATION_CREDENTIALS
    (i.e. we're not in local dev), this writes it out to a temp file and
    repoints GOOGLE_APPLICATION_CREDENTIALS at that file. Every google.STT /
    google.LLM(vertexai=True) / GeminiTTS(vertexai=True) client resolves
    credentials via that same env var (directly, or through Google's
    Application Default Credentials chain), so this is a one-time fixup
    that keeps all three working unmodified.
    """
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not creds_json:
        return

    existing_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if existing_path and os.path.isfile(existing_path):
        return  # a real credentials file already exists (e.g. local dev)

    fd, tmp_path = tempfile.mkstemp(prefix="gcp-credentials-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write(creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_path
    logger.info(
        "Materialized Google credentials from GOOGLE_CREDENTIALS_JSON to %s", tmp_path
    )


_materialize_google_credentials()

# ai-coustics noise cancellation is billed either through a LiveKit Cloud
# project or directly through your own ai-coustics license. For fully
# self-hosted/local development we skip it unless a license key is supplied,
# instead of silently depending on LiveKit Cloud.
_AI_COUSTICS_LICENSE_KEY = os.environ.get("AI_COUSTICS_LICENSE_KEY")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (check .env.local)")
    return value


def _env_float(name: str, default: float) -> float:
    """Read a float tuning knob from the environment, falling back to the
    default (with a warning) rather than crashing the worker on a typo."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring invalid %s=%r, using %s", name, raw, default)
        return default


def _env_int(name: str, default: int, *, allowed: tuple[int, ...] | None = None) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ignoring invalid %s=%r, using %s", name, raw, default)
        return default
    if allowed is not None and value not in allowed:
        logger.warning(
            "ignoring out-of-range %s=%r (allowed: %s), using %s",
            name,
            raw,
            allowed,
            default,
        )
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Fixed opening line for the built-in elder-companion persona, spoken
# straight to TTS on answer (see DefaultAgent.on_enter). Sending a known
# string to TTS instead of asking the LLM to compose a greeting removes an
# entire model round trip from the moment the callee picks up -- the
# difference between a voice starting almost immediately and one to two
# seconds of dead air while someone is holding a phone to their ear
# wondering if anyone is there.
#
# Override per call with `greeting` in the job metadata, or globally with
# the AGENT_GREETING environment variable. The per-language defaults live in
# LANGUAGES below; AGENT_GREETING, if set, wins for every language, so only
# set it in a single-language deployment.
AGENT_GREETING_OVERRIDE = os.environ.get("AGENT_GREETING") or None

# Kept as a module-level name for the English default specifically; every
# other language reads its own greeting off its profile.
DEFAULT_GREETING = AGENT_GREETING_OVERRIDE or LANGUAGES["en"].greeting


def resolve_turn_detection(profile: LanguageProfile):
    """The turn detector for this language, or None to fall back to VAD.

    The audio turn detector covers 14 languages; of the ones this product
    supports, only Bengali is outside that set (English, Spanish, Arabic and
    Hindi are all covered). Handing it an unsupported language isn't an error
    -- it just quietly stops contributing, and turns commit on the
    endpointing delay alone. That is a real difference in how the call feels,
    so it gets said out loud once per call rather than being discovered later
    from a transcript.
    """
    if profile.code not in TURN_DETECTOR_LANGUAGES:
        logger.warning(
            "no turn-detector support for %s (%s); falling back to VAD-only "
            "endpointing, which makes turn-taking less responsive on this call",
            profile.name,
            profile.code,
        )
        return None
    return inference.TurnDetector()


def _google_stt_model(profile: LanguageProfile) -> str:
    """Per-language model, overridable globally (GOOGLE_STT_MODEL) or per
    language (GOOGLE_STT_MODEL_BN, ...). The per-language form is what you
    want once one language is moved onto chirp_2 and the others aren't."""
    per_language = os.environ.get(f"GOOGLE_STT_MODEL_{profile.code.upper()}")
    if per_language:
        return per_language
    return os.environ.get("GOOGLE_STT_MODEL") or profile.google_stt_model


# Seconds to wait after the callee answers before speaking. A carrier can
# complete SIP signalling (which is what wait_until_answered=True waits for)
# slightly before the RTP media path is actually carrying audio, and anything
# spoken in that window gets clipped -- classically the first syllable of the
# greeting. A short settle delay costs far less than a caller hearing
# "...there, it's Avery". Only applied to phone calls; console/web sessions
# use 0.
SIP_GREETING_DELAY = _env_float("SIP_GREETING_DELAY_SECONDS", 0.25)

# --- End-of-call reporting budget ------------------------------------------
# on_session_end() runs inside the worker's own session_end_timeout (300s by
# default in livekit-agents 1.7). Overrun it and the SDK abandons the callback
# entirely -- nothing is posted back, so the outbound call API sits "pending"
# until CALL_TIMEOUT_SECONDS and then reports a timeout with no summary. That
# is the single worst outcome here, so everything on that path is bounded to
# finish well inside it: one summary attempt (45s) plus every callback retry
# (4 x 10s + 7s of backoff) is ~92s, leaving a wide margin.
SUMMARY_TIMEOUT_SECONDS = _env_float("SUMMARY_TIMEOUT_SECONDS", 45.0)
CALLBACK_TIMEOUT_SECONDS = _env_float("CALLBACK_TIMEOUT_SECONDS", 10.0)
CALLBACK_MAX_ATTEMPTS = _env_int("CALLBACK_MAX_ATTEMPTS", 4)
CALLBACK_BACKOFF_SECONDS = _env_float("CALLBACK_BACKOFF_SECONDS", 1.0)

# Must match CALLBACK_SECRET on the outbound call API when that side sets one.
# Unset on both sides means the callback is unauthenticated, which is the old
# behaviour.
CALLBACK_SECRET = os.environ.get("CALLBACK_SECRET") or None


def _resolve_stt_provider(profile: LanguageProfile) -> str:
    """Which STT provider serves this language.

    Precedence: an explicit per-language override (STT_PROVIDER_<CODE>), then
    the language's own pin, then the global STT_PROVIDER, then ElevenLabs.

    ElevenLabs (`scribe_v2_realtime`) became the default on 2026-09-09, now
    that ELEVENLABS_API_KEY is set. It covers all five supported languages in
    one streaming model, rating "Good" (Arabic) to "Excellent" (English,
    Spanish) on ElevenLabs' own WER accuracy tiers, and it sidesteps a Google
    v1 streaming failure that was never root-caused: on a live Bengali call
    with `google_stt_model="latest_long"`, the recognizer worked for one
    exchange and then produced no interim or final results for the rest of
    the call (no error, no reconnect, nothing in the logs) until the caller
    hung up. Google Cloud Speech-to-Text remains fully wired (STT_PROVIDER=
    google) -- it authenticates with the same service account as the LLM/TTS
    and is the fallback if ElevenLabs has its own problems on a live call.
    Gemini live transcription (`gemini-3.5-transcribe-live`) would cover all
    five languages too and reuse that same service account, but on two
    separate live calls Vertex AI returned "Publisher model ... was not
    found" for it, once with `location="us-central1"` and once with
    `location="global"` -- same error both times means the model genuinely
    isn't deployed as a Vertex publisher model on this project right now, not
    a location mistake. Set GEMINI_API_KEY (from Google AI Studio, not a
    service account) to switch STT_PROVIDER=gemini onto that non-Vertex path
    and try it again; until then it will fail every call.
    """
    override = os.environ.get(f"STT_PROVIDER_{profile.code.upper()}")
    if override:
        return override.lower()
    if profile.stt_provider:
        return profile.stt_provider.lower()

    default = os.environ.get("STT_PROVIDER", "elevenlabs").lower()
    # Being forced onto a Google path that can't stream this language is the
    # worst option on the menu, so prefer ElevenLabs over it if a key exists.
    if (
        default == "google"
        and not profile.google_stt_streaming
        and os.environ.get("ELEVENLABS_API_KEY")
    ):
        return "elevenlabs"
    return default


def _build_stt(profile: LanguageProfile):
    provider = _resolve_stt_provider(profile)
    degraded = provider == "google" and not profile.google_stt_streaming
    if degraded:
        # Say this loudly and every call. It is the difference between "the
        # prompt needs work" and "the recognizer is the wrong one", and from
        # the transcript alone those look identical.
        logger.warning(
            "STT for %s is on a degraded path: Google's v1 %r model, recognized "
            "in VAD-cut segments because streaming returns nothing for this "
            "language. Expect poor accuracy and no transcript until the caller "
            "stops speaking. Fix by setting ELEVENLABS_API_KEY (this language "
            "then switches to scribe_v2_realtime automatically), or by enabling "
            "Google chirp_2 -- see 'Language support' in README.md.",
            profile.name,
            _google_stt_model(profile),
        )
    else:
        logger.info(
            "using %s STT for %s, expecting %s",
            provider,
            profile.name,
            "/".join(profile.stt_language_codes),
        )
    if provider == "gemini":
        # gemini-3.5-transcribe-live, over the Live API. Streams with interim
        # results and supports bn-BD / hi-IN / ar-EG / es-US / en-US directly,
        # so it needs neither the v2 Chirp IAM grant nor an ElevenLabs key --
        # it reuses the Vertex service account the LLM and TTS already use.
        gemini_stt_model = os.environ.get(
            "GEMINI_STT_MODEL", "gemini-3.5-transcribe-live"
        )
        gemini_stt_key = os.environ.get("GEMINI_API_KEY")
        if gemini_stt_key:
            return GeminiSTT(
                model=gemini_stt_model,
                language=profile.bcp47,
                language_codes=list(profile.stt_language_codes),
                api_key=gemini_stt_key,
            )
        return GeminiSTT(
            model=gemini_stt_model,
            language=profile.bcp47,
            language_codes=list(profile.stt_language_codes),
            vertexai=True,
            # Reads project_id straight out of the key file, same as the
            # Vertex fallbacks in _build_llm / _build_tts.
            credentials_path=_require_env("GOOGLE_APPLICATION_CREDENTIALS"),
            # The plugin itself defaults to "global" when no location is
            # passed (see gemini_stt.py: `location = self._opts.location or
            # "global"`) -- that is also what the Vertex LLM/TTS fallbacks
            # above use successfully. "us-central1" was tried first and
            # produced "Publisher model ... was not found" on this project,
            # so match what already works instead of pinning a region.
            location=os.environ.get("GEMINI_STT_LOCATION", "global"),
        )

    if provider == "elevenlabs":
        # scribe_v2_realtime is the streaming model; the older scribe_v1 is
        # batch-only, which on a live call means waiting for an utterance to
        # end before any text exists. Scribe also covers all five supported
        # languages, which is why it's the fallback when Google's model for a
        # language isn't available -- see README's "Language support".
        #
        # server_vad is NOT optional. Without it the plugin connects with
        # commit_strategy=manual, which tells ElevenLabs' server to never
        # finalize a transcript on its own -- it only commits when the client
        # sends an explicit "commit": true message, which the installed
        # plugin (livekit-plugins-elevenlabs, checked in
        # .venv/.../elevenlabs/stt.py) only ever sends when its stream's
        # .flush() is called. Nothing in livekit-agents' AudioRecognition
        # calls .flush() on a streaming STT mid-call (that's a StreamAdapter-
        # only concept, for non-streaming plugins) -- it only pushes extra
        # silence *audio* through _commit_user_turn(), which manual-strategy
        # ElevenLabs has no way to interpret as "finalize now". Net effect,
        # confirmed on a live Bengali call: partial_transcript kept arriving
        # and drifting/hallucinating for 40+ seconds after the caller spoke
        # once, committed_transcript never arrived, and the agent never
        # produced a reply. Passing server_vad switches the connection to
        # commit_strategy=vad, which makes ElevenLabs' own server finalize on
        # silence the way every other streaming STT here already does.
        vad_silence_threshold_secs = _env_float("ELEVENLABS_STT_VAD_SILENCE_SECS", 1.5)
        vad_min_silence_duration_ms = _env_int("ELEVENLABS_STT_VAD_MIN_SILENCE_MS", 800)
        return elevenlabs.STT(
            api_key=_require_env("ELEVENLABS_API_KEY"),
            model=os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v2_realtime"),
            # ISO-639 here, not the BCP-47 locale: ElevenLabs takes a bare
            # language code and would reject "bn-BD". Google is the opposite --
            # region genuinely changes the recognition target there.
            language_code=profile.code,
            server_vad={
                "vad_silence_threshold_secs": vad_silence_threshold_secs,
                # ElevenLabs' own default (2500ms) is sluggish for a phone
                # call -- 800ms roughly matches VAD_MIN_SILENCE_DURATION used
                # for the session's own turn-taking VAD below.
                "min_silence_duration_ms": vad_min_silence_duration_ms,
            },
        )
    if provider == "google":
        # Google Cloud Speech-to-Text, authenticated with a service account
        # key file (see GOOGLE_APPLICATION_CREDENTIALS in .env.local). Same
        # service account used by the Vertex AI fallback for
        # LLM_PROVIDER=gemini / TTS_PROVIDER=gemini below.
        #
        # The model comes from the language profile, because it decides both
        # the quality and (via the plugin) the API version: v2 for
        # telephony/chirp_2/chirp_3, v1 for everything else. chirp_2 is the
        # right answer for the non-English languages but needs the v2 API,
        # roles/speech.client on the service account, and a non-global
        # GOOGLE_STT_LOCATION -- see "Language support" in README.md.
        return google.STT(
            credentials_file=_require_env("GOOGLE_APPLICATION_CREDENTIALS"),
            model=_google_stt_model(profile),
            # Without this the plugin uses its own "en-US" default, which is
            # what made every call transcribe as English regardless of the
            # language requested.
            languages=list(profile.stt_language_codes),
            # "Detection" only in the sense of choosing among the languages
            # listed above. The call's language is fixed, but a caller dropping
            # English words into it is normal speech, not an error to correct.
            detect_language=len(profile.stt_language_codes) > 1,
            location=os.environ.get("GOOGLE_STT_LOCATION", "global"),
            # False here doesn't disable transcription -- it makes the
            # framework segment audio with VAD and recognize each utterance as
            # it ends, instead of trusting a streaming recognizer that (for
            # some languages) returns nothing until the call is over.
            use_streaming=profile.google_stt_streaming,
        )
    raise ValueError(
        f"Unknown STT_PROVIDER: {provider!r} "
        "(expected 'gemini', 'google', or 'elevenlabs')"
    )


def _build_llm():
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            return google.LLM(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                api_key=api_key,
            )
        # No API key: fall back to Vertex AI using a Google Cloud service
        # account (GOOGLE_APPLICATION_CREDENTIALS / GOOGLE_CLOUD_PROJECT /
        # GOOGLE_CLOUD_LOCATION), same credentials as STT_PROVIDER=google.
        return google.LLM(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            vertexai=True,
        )
    if provider == "openai":
        return openai.LLM(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=_require_env("OPENAI_API_KEY"),
        )
    if provider == "ollama":
        return openai.LLM.with_ollama(
            model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r} (expected 'gemini', 'openai', or 'ollama')"
    )


def _build_tts(profile: LanguageProfile):
    provider = os.environ.get("TTS_PROVIDER", "elevenlabs").lower()
    if provider == "elevenlabs":
        # A voice is not language-neutral: the default voice is an English
        # one, and it carries an English accent into every other language.
        # ELEVENLABS_VOICE_ID_<CODE> picks a native voice per language.
        voice_id = os.environ.get(
            f"ELEVENLABS_VOICE_ID_{profile.code.upper()}"
        ) or os.environ.get("ELEVENLABS_VOICE_ID", "hpp4J3VqNfWAUOO0d1Us")
        return elevenlabs.TTS(
            voice_id=voice_id,
            model=os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5"),
            api_key=_require_env("ELEVENLABS_API_KEY"),
            language=profile.code,
            # Lets the <break time="..."/> tags in DefaultAgent's instructions
            # (see _supports_ssml_breaks) render as actual pauses instead of
            # being read aloud.
            enable_ssml_parsing=True,
        )
    if provider == "gemini":
        # Gemini TTS has no language parameter -- it infers the language from
        # the text it is handed, and covers all five supported languages that
        # way. That makes the greeting load-bearing: hand it an English
        # opening line on a Bengali call and the first thing the callee hears
        # is an English voice. See LANGUAGES above.
        gemini_tts_api_key = os.environ.get("GEMINI_API_KEY")
        gemini_voice = os.environ.get(
            f"GEMINI_TTS_VOICE_{profile.code.upper()}"
        ) or os.environ.get("GEMINI_TTS_VOICE", "Kore")
        if gemini_tts_api_key:
            return GeminiTTS(
                voice_name=gemini_voice,
                api_key=gemini_tts_api_key,
            )
        # No API key: fall back to Vertex AI using a Google Cloud service
        # account (GOOGLE_APPLICATION_CREDENTIALS / GOOGLE_CLOUD_PROJECT /
        # GOOGLE_CLOUD_LOCATION), same as LLM_PROVIDER=gemini above. The
        # project is inferred from the service account key file and the
        # location defaults to "us-central1" if not set.
        return GeminiTTS(
            voice_name=gemini_voice,
            vertexai=True,
        )
    if provider == "cartesia":
        return cartesia.TTS(
            model="sonic-3",
            voice="a167e0f3-df7e-4d52-a9c3-f949145efdab",
            language=profile.code,
            api_key=_require_env("CARTESIA_API_KEY"),
        )
    if provider == "google":
        # Google Cloud Text-to-Speech, authenticated with a service account
        # key file (see GOOGLE_APPLICATION_CREDENTIALS in .env.local). This
        # is distinct from TTS_PROVIDER=gemini above, which uses Gemini's
        # own TTS model instead of the Cloud Text-to-Speech API.
        return google.TTS(
            credentials_file=_require_env("GOOGLE_APPLICATION_CREDENTIALS"),
            # Chirp3-HD voices only cover a subset of locales -- bn-IN and
            # ar-XA, not bn-BD/ar-EG -- so this uses google_tts_bcp47 when the
            # profile has one instead of bcp47 directly. See that field's
            # docstring in languages.py.
            language=profile.google_tts_bcp47 or profile.bcp47,
            voice_name=os.environ.get(f"GOOGLE_TTS_VOICE_{profile.code.upper()}")
            or os.environ.get("GOOGLE_TTS_VOICE")
            or NOT_GIVEN,
        )
    if provider == "coqui":
        # Local import: torch/coqui-tts are heavy, GPU-oriented dependencies
        # that are only needed when this provider is actually selected. See
        # the "Coqui XTTS v2 (local)" section in README.md for setup.
        import xtts_tts

        return xtts_tts.TTS(
            model_name=os.environ.get("COQUI_XTTS_MODEL", xtts_tts.DEFAULT_MODEL_NAME),
            language=os.environ.get("COQUI_XTTS_LANGUAGE", "en"),
            speaker=os.environ.get("COQUI_XTTS_SPEAKER") or None,
            speaker_wav=os.environ.get("COQUI_XTTS_SPEAKER_WAV") or None,
            device=os.environ.get("COQUI_XTTS_DEVICE", "auto"),
        )
    raise ValueError(
        f"Unknown TTS_PROVIDER: {provider!r} "
        "(expected 'elevenlabs', 'gemini', 'cartesia', 'google', or 'coqui')"
    )


def _supports_ssml_breaks() -> bool:
    # <break time="..."/> only becomes an actual pause instead of literal
    # spoken text on TTS providers that parse SSML. ElevenLabs needs
    # enable_ssml_parsing=True (set in _build_tts); Cartesia's sonic models
    # parse SSML natively. Other providers would just read the tag aloud.
    return os.environ.get("TTS_PROVIDER", "elevenlabs").lower() in (
        "elevenlabs",
        "cartesia",
    )


def _voice_realism_instructions(profile: LanguageProfile) -> str:
    """Output-formatting and speech-pacing rules shared by every persona
    (the hardcoded default and any dynamically supplied `prompt`), so a
    caller-supplied prompt still gets short, natural, phone-call-paced
    replies instead of written-style text."""
    pause_examples = (
        """
            # Pauses and filler words

            Real speech has small pauses and filler words like "um" and "so" — written text doesn't. After a standalone "um" or "hmm", insert <break time="300ms"/> before continuing.

            Examples:
            * Bad: "I can definitely check on that for you."
            * Good: "Yeah, um <break time="300ms"/> so, let me check on that."
            * Bad: "That sounds like it was a good afternoon."
            * Good: "Hmm <break time="400ms"/> that sounds like it was a good afternoon."
            """
        if _supports_ssml_breaks()
        else """
            # Pauses and filler words

            Real speech has small pauses and filler words like "um" and "so" — written text doesn't. Sprinkle these in naturally, the way a person thinking out loud would.

            Examples:
            * Bad: "I can definitely check on that for you."
            * Good: "Yeah, um, so, let me check on that."
            * Bad: "That sounds like it was a good afternoon."
            * Good: "Hmm, that sounds like it was a good afternoon."
            """
    )
    # Named, not coded: "Speak only in Bengali (Bangla)" steers a model far
    # better than "Speak only in bn". The instructions themselves stay in
    # English on purpose -- the model follows them fine and it keeps one
    # reviewable copy of the persona rather than five translations that drift.
    language_line = (
        f"* Speak only in {profile.name}, regardless of what language this prompt is written in. "
        f"This applies to every single turn, including the first one and any numbers, dates or names you say.\n"
        if profile.code != DEFAULT_LANGUAGE
        else ""
    )
    return f"""
        # Output rules

        You are talking on a phone call, not writing a message. Apply these rules so you sound natural and don't drag the call out:

        * Keep replies short: one to two sentences per turn, like a real phone conversation. Never deliver a monologue.
        * Ask exactly one question at a time, then stop talking and let them answer.
        * Never use markdown, lists, emojis, or other written-text formatting — everything you say is spoken aloud.
        * Don't open consecutive turns with the same acknowledgment (for example, "Oh, that's nice" twice in a row). Rotate through different natural openers instead.
        {language_line}
        {pause_examples}

        # Self-corrections

        Occasionally, drop a phrase mid-sentence and pick a different one, the way people naturally reconsider what they're about to say. Don't apologize for it, just continue.

        Examples:
        * Bad: "That sounds like a nice walk."
        * Good: "That sounds like a nice — well, a really peaceful walk."
        """


class DefaultAgent(Agent):
    def __init__(
        self,
        *,
        prompt: str | None = None,
        helper_prompt: str | None = None,
        language: str = "en",
        greeting: str | None = None,
        greeting_delay: float = 0.0,
        greeting_interruptible: bool = True,
    ) -> None:
        profile = resolve_language(language)
        if prompt:
            # Caller-supplied persona (see the outbound call API in api.py):
            # `prompt` is the full identity/goal instructions, `helper_prompt`
            # is optional supplementary context. Voice pacing/formatting
            # rules still apply on top so it sounds like a phone call.
            sections = [prompt.strip()]
            if helper_prompt and helper_prompt.strip():
                sections.append(f"# Additional context\n\n{helper_prompt.strip()}")
            sections.append(_voice_realism_instructions(profile))
            instructions = "\n\n".join(sections)
        else:
            instructions = f"""You are a warm, caring family companion speaking with an elderly parent on behalf of their son or daughter.

                Your personality:

                * Speak naturally, warmly, and respectfully.
                * Sound like a caring family member checking in, not a survey agent or customer service representative.
                * Show genuine interest in the person's day and well-being.
                * Keep the conversation relaxed and friendly.
                * Feel free to start sentences with "And", "But", or "So", the way people do when talking, not writing.
                * Reference something they said earlier loosely ("about what you mentioned a minute ago") rather than quoting it back verbatim.

                {_voice_realism_instructions(profile)}

                Conversation goals:

                * Understand how the person is feeling physically and emotionally.
                * Learn about their daily activities, meals, sleep, exercise, medication, and overall well-being through natural conversation.
                * Provide companionship and make them feel heard.
                * Encourage them to share details about their day.

                Important rules:

                * Never ask questions like a form or questionnaire.
                * Never rapidly fire multiple questions.
                * Ask one natural question at a time.
                * Follow up on what they say.
                * If they mention a symptom, discomfort, poor sleep, appetite changes, loneliness, stress, or medication issues, gently explore further.
                * If they seem happy or energetic, continue the conversation naturally.
                * Feel free to talk about family, hobbies, weather, memories, food, walking, gardening, television, friends, or daily routines when appropriate.
                * Do not make the conversation feel like data collection.

                Examples of natural questions:

                * How has your day been so far?
                * Did you sleep well last night?
                * What did you have for breakfast today?
                * Have you been able to get outside a little today?
                * Have you spoken with any friends or family recently?
                * Is there anything bothering you today?
                * What have you been doing to keep yourself busy?

                When health concerns appear:

                * Ask gentle follow-up questions.
                * Do not diagnose medical conditions.
                * Encourage professional medical care if symptoms sound serious.

                At the end of the conversation:

                * Briefly summarize how the person is doing.
                * Mention any notable health concerns, mood updates, eating habits, sleep issues, or positive activities they shared.
                * End with warmth and encouragement.
                """
        super().__init__(instructions=instructions)
        self._profile = profile
        self._greeting = greeting
        self._greeting_delay = greeting_delay
        self._greeting_interruptible = greeting_interruptible
        self._custom_persona = bool(prompt)

    def resolve_greeting(self) -> str | None:
        """The exact line to speak on answer, or None to have the LLM compose
        one instead.

        A caller-supplied persona (the outbound call API's `prompt`) gets no
        fixed default: "it's Avery calling to see how you're doing" would be
        wrong coming from an arbitrary persona, so unless that caller also
        supplies its own `greeting`, the opening line is generated. The
        built-in elder-companion persona always has a fixed line available,
        which is the case that benefits most from skipping the LLM.
        """
        if self._greeting:
            return self._greeting
        if self._custom_persona:
            return None
        # Per-language, so a Bengali call opens in Bengali. AGENT_GREETING,
        # if set, overrides every language at once.
        return AGENT_GREETING_OVERRIDE or self._profile.greeting

    async def on_enter(self) -> None:
        if self._greeting_delay > 0:
            await asyncio.sleep(self._greeting_delay)

        greeting = self.resolve_greeting()
        if greeting:
            # Straight to TTS -- no LLM round trip, so audio starts as soon
            # as the first synthesis chunk lands rather than after a full
            # generate/synthesize cycle.
            await self.session.say(
                greeting, allow_interruptions=self._greeting_interruptible
            )
            return

        await self.session.generate_reply(
            instructions=(
                "Greet the person warmly and briefly say why you're calling."
            ),
            allow_interruptions=True,
        )

    @function_tool
    async def begin_wrap_up(self) -> Agent:
        """Call this once, and only once, when the conversation has run its
        natural course and it's time to say goodbye -- for example once the
        person has shared how they're doing and there's no new ground left
        to cover, or if they signal they want to end the call. Do not call
        this early or in the middle of the conversation.
        """
        return ClosingAgent(language=self._profile.code)


class ClosingAgent(Agent):
    """Dedicated closing phase, entered via DefaultAgent.begin_wrap_up().

    Kept as its own Agent/task rather than folded into DefaultAgent's single
    instructions block -- see AGENTS.md's guidance to use handoffs/tasks for
    distinct conversation phases instead of one long prompt covering all of
    them. This is the phase boundary most worth a hard structural trigger:
    whether a call is *ending* shouldn't depend on the model inferring it
    from prose alone, and this applies to both the default persona and any
    caller-supplied `prompt` (see api.py), since begin_wrap_up() is defined
    on DefaultAgent itself rather than tucked inside persona-specific text.
    """

    def __init__(self, *, language: str = "en") -> None:
        profile = resolve_language(language)
        instructions = f"""The conversation is ending now -- you're wrapping up the call.

            * In one or two sentences, warmly note how the person seems to be doing today.
            * If something notable came up (mood, health, an activity they mentioned), you can mention it briefly -- only if it's relevant, don't force it.
            * Thank them and say goodbye the way a caring family member would, not a script. Vary your wording -- don't reuse the same sign-off every call.
            * Do not ask any new questions. The conversation is over; this is the last thing you say.

            {_voice_realism_instructions(profile)}
            """
        super().__init__(instructions=instructions)

    async def on_enter(self) -> None:
        await self.session.generate_reply(allow_interruptions=True)


def _parse_dial_info(ctx: JobContext) -> dict:
    # Outbound phone calls are triggered by dispatching this agent with JSON
    # metadata. `phone_number` alone is set by src/place_call.py for manual
    # testing; the full set (call_id, prompt, helper_prompt, language,
    # callback_url) is set by the outbound call API in api.py. Regular
    # sessions (console, web/mobile frontends) have no metadata at all.
    if not ctx.job.metadata:
        return {}
    try:
        metadata = json.loads(ctx.job.metadata)
    except json.JSONDecodeError:
        logger.warning("ignoring non-JSON job metadata: %r", ctx.job.metadata)
        return {}
    if not isinstance(metadata, dict):
        # Valid JSON of the wrong shape (a bare string, a list). Everything
        # downstream calls .get() on this, so reject it at the trust boundary
        # rather than dying with an AttributeError halfway through a call.
        logger.warning("ignoring non-object job metadata: %r", ctx.job.metadata)
        return {}
    return metadata


async def _post_callback(callback_url: str, payload: dict) -> bool:
    """POST a call's result back to the outbound call API (see api.py).

    This is the only moment that result exists: it was just built from an
    in-memory transcript, in a worker process that is already shutting down,
    and nothing re-derives it afterwards. So a dropped request doesn't delay
    the result, it destroys it -- the API stays "pending" until
    CALL_TIMEOUT_SECONDS and then reports a timeout with no summary, which
    looks identical to a call that never happened. CALLBACK_BASE_URL commonly
    points at a tunnel or a single-process service that can blip for a second,
    so transient failures are retried with exponential backoff instead of
    being logged and forgotten.

    Never raises: a failed callback should not crash the job. Returns whether
    the API accepted the result.
    """
    call_id = payload.get("call_id")
    backoff = CALLBACK_BACKOFF_SECONDS
    reason = "unknown"

    for attempt in range(1, CALLBACK_MAX_ATTEMPTS + 1):
        try:
            headers = (
                {"X-Callback-Secret": CALLBACK_SECRET} if CALLBACK_SECRET else None
            )
            async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    callback_url, json=payload, headers=headers
                )
                response.raise_for_status()
            return True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                # The API understood the request and refused it -- an unknown
                # call_id, or a body it can't parse. Retrying cannot change
                # the answer, and this job is holding up its own shutdown.
                logger.error(
                    "call-completed callback to %s rejected with HTTP %s; not retrying",
                    callback_url,
                    exc.response.status_code,
                    extra={"call_id": call_id},
                )
                return False
            reason = f"HTTP {exc.response.status_code}"
        except httpx.HTTPError as exc:
            reason = f"{type(exc).__name__}: {exc}"

        if attempt < CALLBACK_MAX_ATTEMPTS:
            logger.warning(
                "call-completed callback to %s failed (%s), attempt %d/%d, retrying in %.1fs",
                callback_url,
                reason,
                attempt,
                CALLBACK_MAX_ATTEMPTS,
                backoff,
                extra={"call_id": call_id},
            )
            await asyncio.sleep(backoff)
            backoff *= 2

    logger.error(
        "giving up on the call-completed callback to %s after %d attempts (%s); "
        "this call's result is lost and the API will report it as a timeout",
        callback_url,
        CALLBACK_MAX_ATTEMPTS,
        reason,
        extra={"call_id": call_id},
    )
    return False


@dataclass
class CallSummary:
    """Structured result of summarize_session(), so a backend can branch on
    `concern_level` programmatically instead of a human re-reading prose to
    notice something urgent (see api.py's /calls/outbound response and the
    /internal/calls/{call_id}/completed callback payload)."""

    text: str
    concern_level: Literal["none", "watch", "urgent"] = "none"
    flagged_topics: list[str] = field(default_factory=list)


@function_tool(name="record_call_summary")
async def _record_call_summary(
    summary: str,
    concern_level: Literal["none", "watch", "urgent"],
    flagged_topics: list[str],
) -> None:
    """Record a structured summary of the phone conversation for a business record.

    Args:
        summary: 2-4 factual, concise sentences summarizing what was discussed.
        concern_level: "urgent" if the caller disclosed something needing
            prompt human attention (an injury, a medical emergency, a safety
            concern, or severe distress). "watch" for a minor or ambiguous
            concern worth a caregiver's attention but not urgent (a mild
            symptom, low mood, a medication question). "none" for a routine,
            unremarkable call.
        flagged_topics: short topic tags for anything concerning that was
            raised (for example "fall", "chest pain", "loneliness"). Leave
            empty if concern_level is "none".
    """
    # This function is never actually executed -- summarize_session() below
    # reads the tool call's arguments directly instead of running this body.
    # It exists purely to give the LLM a schema to call into, forcing a
    # structured response instead of free text.
    return None


async def summarize_session(summarizer, chat_ctx: ChatContext) -> CallSummary | None:
    """Generate a structured summary of the user/assistant turns using a
    separate, non-conversational LLM call that's forced to report through
    the `record_call_summary` tool instead of free text. Based on the
    "Summarizing context" pattern in the LiveKit Agents docs
    (agents/logic/agents-handoffs), extended with tool-calling so the result
    is machine-actionable, not just human-readable."""
    summary_ctx = ChatContext()
    summary_ctx.add_message(
        role="system",
        content=(
            "Summarize the following phone conversation for a business "
            "record by calling record_call_summary exactly once."
        ),
    )

    n_summarized = 0
    for item in chat_ctx.items:
        if item.type != "message" or item.role not in ("user", "assistant"):
            continue
        item_text = (item.text_content or "").strip()
        if item_text:
            summary_ctx.add_message(role="user", content=f"{item.role}: {item_text}")
            n_summarized += 1

    if n_summarized == 0:
        return None

    response = await summarizer.chat(
        chat_ctx=summary_ctx,
        tools=[_record_call_summary],
        tool_choice="required",
    ).collect()

    for call in response.tool_calls:
        if call.name != "record_call_summary":
            continue
        try:
            args = json.loads(call.arguments)
            return CallSummary(
                text=str(args["summary"]).strip(),
                concern_level=args.get("concern_level", "none"),
                flagged_topics=list(args.get("flagged_topics") or []),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.exception(
                "failed to parse record_call_summary arguments: %r", call.arguments
            )
            break

    # The model didn't call the tool as instructed, or returned malformed
    # arguments. Fall back to whatever free text came back rather than
    # losing the summary entirely.
    fallback_text = response.text.strip() if response.text else None
    return CallSummary(text=fallback_text) if fallback_text else None


async def on_session_end(ctx: JobContext) -> None:
    dial_info = _parse_dial_info(ctx)
    call_id = dial_info.get("call_id")
    callback_url = dial_info.get("callback_url")
    if not call_id or not callback_url:
        # No outbound call API is waiting on this job (console/web session,
        # or a call placed via place_call.py) -- nothing to report back.
        return

    # JobContext.primary_session *raises* when no AgentSession was ever
    # started -- it does not return None (livekit-agents 1.7, job.py). An
    # unhandled raise here is silent: the worker catches it, logs it and moves
    # on, so nothing is posted back at all and the API sits "pending" until
    # CALL_TIMEOUT_SECONDS before reporting a timeout with no summary. Report
    # the failure instead, immediately.
    try:
        session = ctx.primary_session
    except RuntimeError:
        logger.warning(
            "no agent session was started for call %s",
            call_id,
            extra={"call_id": call_id},
        )
        await _post_callback(
            callback_url, {"call_id": call_id, "error": "session never started"}
        )
        return

    # The worker calls aclose() on the session *before* invoking this, which
    # is fine to summarize from: session.history and session.llm are plain
    # attributes that outlive the close, and only the live audio activity is
    # torn down. It does mean nothing can rescue this if the summarizer
    # stalls, though -- and a stall that outlasts the worker's
    # session_end_timeout kills the callback with it -- hence the hard bound.
    summary: CallSummary | None = None
    summary_error: str | None = None
    try:
        summary = await asyncio.wait_for(
            summarize_session(session.llm, session.history),
            timeout=SUMMARY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        summary_error = (
            f"summary generation timed out after {SUMMARY_TIMEOUT_SECONDS:g}s"
        )
        logger.error(
            "summary generation timed out for call %s after %.1fs",
            call_id,
            SUMMARY_TIMEOUT_SECONDS,
            extra={"call_id": call_id},
        )
    except Exception as exc:
        summary_error = f"summary generation failed: {exc}"
        logger.exception(
            "failed to summarize session for call %s",
            call_id,
            extra={"call_id": call_id},
        )

    payload: dict = {"call_id": call_id}
    if summary_error is not None:
        # Deliberately not reported as a completed call carrying an empty
        # summary and concern_level "none". On a wellbeing check-in those mean
        # "nothing came up"; a failed summary means "we don't know what came
        # up". Collapsing the second into the first is how an urgent call ends
        # up indistinguishable from a chat about the garden.
        payload["error"] = summary_error
    else:
        payload["response_summary"] = summary.text if summary else ""
        payload["concern_level"] = summary.concern_level if summary else "none"
        payload["flagged_topics"] = summary.flagged_topics if summary else []

    logger.info(
        "call %s ended, concern_level=%s",
        call_id,
        payload.get("concern_level", "unknown"),
        extra={"call_id": call_id},
    )
    await _post_callback(callback_url, payload)


def prewarm(proc: JobProcess) -> None:
    """Load the VAD model once per worker process instead of once per call.
    Without this, entrypoint() would call silero.VAD.load() fresh on every
    single job, paying ONNX model init time before the agent can start
    listening -- on the critical path of every call's setup latency.

    The defaults below are tuned for telephony rather than for Silero's
    stock assumption of a clean, wideband desktop microphone. A phone line
    carries constant low-level hiss, comfort noise and codec artifacts, and
    the stock settings treat a lot of that as speech -- which is what makes
    an agent talk over people, cut itself off mid-sentence, or take turns
    nobody asked for. Every value is overridable by environment variable so
    a deployment can retune against real call recordings without a code
    change.
    """
    proc.userdata["vad"] = silero.VAD.load(
        # 50ms (the stock value) is short enough that a line pop, a door, or
        # a single clacked key registers as the start of a turn. 150ms still
        # catches a real "yes" but ignores the blips.
        min_speech_duration=_env_float("VAD_MIN_SPEECH_DURATION", 0.15),
        # How long a pause has to run before the turn is considered over.
        # Nudged up from 0.55 because this agent's callers are elderly and
        # often pause mid-thought; being cut off mid-sentence is worse here
        # than a slightly later reply.
        min_silence_duration=_env_float("VAD_MIN_SILENCE_DURATION", 0.65),
        # Audio kept from just before speech was detected, so the STT still
        # hears the word's onset. Left at the stock value.
        prefix_padding_duration=_env_float("VAD_PREFIX_PADDING_DURATION", 0.5),
        # Raised from 0.5: on a noisy line, 0.5 fires on the noise floor.
        activation_threshold=_env_float("VAD_ACTIVATION_THRESHOLD", 0.6),
        # Deliberately far below the activation threshold. This hysteresis
        # band is what keeps a turn alive through the quiet dips inside
        # normal speech; with the two thresholds close together, the VAD
        # flickers between speech and silence mid-sentence.
        deactivation_threshold=_env_float("VAD_DEACTIVATION_THRESHOLD", 0.35),
        # Silero ships an 8kHz variant matching telephony's native rate.
        # Left at 16000 because this same worker also serves console/web
        # sessions with wideband audio; a phone-only deployment can try
        # VAD_SAMPLE_RATE=8000.
        sample_rate=_env_int("VAD_SAMPLE_RATE", 16000, allowed=(8000, 16000)),
    )


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session(agent_name="Avery-ff5", on_session_end=on_session_end)
async def entrypoint(ctx: JobContext):
    dial_info = _parse_dial_info(ctx)

    phone_number = dial_info.get("phone_number")
    call_id = dial_info.get("call_id")
    callback_url = dial_info.get("callback_url")
    profile = resolve_language(dial_info.get("language"))
    logger.info(
        "call %s running in %s (%s)",
        call_id,
        profile.name,
        profile.bcp47,
        extra={"call_id": call_id},
    )

    llm = _build_llm()
    # Fire-and-forget: prewarm() schedules a background task and returns
    # immediately, so this overlaps with the SIP dial-out below instead of
    # adding connection-setup time to the caller's first turn.
    llm.prewarm()

    session = AgentSession(
        stt=_build_stt(profile),
        llm=llm,
        tts=_build_tts(profile),
        turn_handling=TurnHandlingOptions(
            turn_detection=resolve_turn_detection(profile),
            # "dynamic" adapts the end-of-turn wait to each caller's actual
            # pause patterns instead of always waiting the full min_delay,
            # so replies come faster for callers who don't pause much.
            endpointing={"mode": "dynamic", "min_delay": 0.5, "max_delay": 3.0},
            preemptive_generation={
                "enabled": True,
                # Start TTS before the turn is fully confirmed, not just the
                # LLM. Cuts more latency at the cost of occasionally wasted
                # synthesis when a preemptive guess gets discarded. On a
                # noisy line that waste goes up along with the false starts,
                # so PREEMPTIVE_TTS=0 is the first thing to try if calls
                # sound unstable after the VAD retune.
                "preemptive_tts": _env_bool("PREEMPTIVE_TTS", True),
            },
            # Adaptive interruption uses an audio model to tell real
            # barge-ins apart from backchannel acknowledgments ("mm-hmm",
            # "yeah"), so the agent doesn't stop mid-sentence for those. This
            # calls LiveKit Cloud's inference gateway, which phone calls
            # already require (see SIP_OUTBOUND_TRUNK_ID / README) — it only
            # adds a cloud dependency for fully self-hosted local testing.
            interruption={
                "mode": "adaptive",
                "min_duration": _env_float("INTERRUPTION_MIN_DURATION", 0.6),
                # The single biggest source of an agent being cut off by
                # nothing: with min_words at 0, any burst of audio the VAD
                # accepts stops the agent mid-sentence -- a television in
                # the background, a cough, a passing siren. Requiring at
                # least one actually-transcribed word means a real utterance
                # interrupts and ambient noise doesn't.
                "min_words": _env_int("INTERRUPTION_MIN_WORDS", 1),
                # The greeting is uninterruptible on phone calls (see
                # DefaultAgent.on_enter), and the stock behaviour is to
                # throw away audio captured while the agent can't be
                # interrupted. People answer the phone with "Hello?" -- that
                # belongs in the transcript, not in the bin, so keep it and
                # let it be handled as soon as the greeting finishes.
                "discard_audio_if_uninterruptible": False,
            },
        ),
        vad=ctx.proc.userdata["vad"],
    )

    room_options_kwargs = {}
    if _AI_COUSTICS_LICENSE_KEY:
        room_options_kwargs["audio_input"] = room_io.AudioInputOptions(
            noise_cancellation=ai_coustics.audio_enhancement(
                model=ai_coustics.EnhancerModel.QUAIL_VF_L,
                auth=ai_coustics.Auth.ai_coustics_api(
                    license_key=_AI_COUSTICS_LICENSE_KEY
                ),
            ),
        )
    else:
        logger.warning(
            "AI_COUSTICS_LICENSE_KEY not set; skipping ai-coustics noise "
            "cancellation (it otherwise requires a LiveKit Cloud project)"
        )

    if phone_number:
        sip_trunk_id = _require_env("SIP_OUTBOUND_TRUNK_ID")
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=sip_trunk_id,
                    sip_call_to=phone_number,
                    participant_identity=phone_number,
                    wait_until_answered=True,
                )
            )
        except api.ServerError as e:
            # api.SipCallError (raised when the far end actively rejects the
            # call -- busy, declined, invalid number) carries a SIP status
            # code/reason. A ring that just times out with nobody answering
            # -- or a lower-level SIP/trunk failure -- surfaces as a plain
            # api.ServerError instead, with no SIP status attached. Catch the
            # broader type so neither case crashes the job with an unhandled
            # exception; report whichever detail is available.
            if isinstance(e, api.SipCallError):
                detail = f"{e.sip_status_code} {e.sip_status}"
            else:
                detail = f"{e.code} {e.message}"
            logger.warning(
                "outbound call to %s failed: %s",
                phone_number,
                detail,
                extra={"call_id": call_id},
            )
            if call_id and callback_url:
                # on_session_end won't fire since no session ever started;
                # report the failure directly so the outbound call API
                # doesn't just block until its timeout.
                await _post_callback(
                    callback_url,
                    {
                        "call_id": call_id,
                        "error": f"sip call failed: {detail}",
                    },
                )
            ctx.shutdown(reason="sip call failed")
            return

    try:
        await session.start(
            agent=DefaultAgent(
                prompt=dial_info.get("prompt"),
                helper_prompt=dial_info.get("helper_prompt"),
                language=profile.code,
                greeting=dial_info.get("greeting"),
                # Only phone calls need the media-path settle delay, and only
                # phone calls need an uninterruptible greeting: on a console or
                # web session there's no carrier in the middle and no line noise
                # to protect the opening line from.
                greeting_delay=SIP_GREETING_DELAY if phone_number else 0.0,
                greeting_interruptible=not phone_number,
            ),
            room=ctx.room,
            room_options=room_io.RoomOptions(**room_options_kwargs),
        )
    except Exception as exc:
        # Anything that stops the session from starting -- a bad
        # service-account key, a provider outage, the inference gateway
        # refusing -- would otherwise end this job with nobody told. There is
        # no session for on_session_end to summarize, so without this the
        # caller waits out CALL_TIMEOUT_SECONDS for a timeout that says
        # nothing about what actually broke. Report it while we still know.
        logger.exception(
            "failed to start session for call %s",
            call_id,
            extra={"call_id": call_id},
        )
        if call_id and callback_url:
            await _post_callback(
                callback_url,
                {"call_id": call_id, "error": f"failed to start session: {exc}"},
            )
        raise

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=1.0)
    )
    await background_audio.start(room=ctx.room, agent_session=session)


if __name__ == "__main__":
    cli.run_app(server)
