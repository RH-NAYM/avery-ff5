"""The languages this product supports, and how to recognize each one.

Kept separate from agent.py on purpose: api.py validates the incoming
`langage` field against this list, and importing agent.py to do that would
drag in every STT/TTS plugin and run the Google credential materialization
in a process that has no use for either.

Adding a language means adding a row to LANGUAGES *and* checking three things
that do not come for free:

1. STT has a model that covers it (see LanguageProfile.google_stt_model, and
   the "Language support" section in README.md).
2. TTS can speak it in a voice that sounds native, not an English voice
   reading foreign words.
3. Whether the turn detector supports it (TURN_DETECTOR_LANGUAGES). If not,
   the call still works, but turn-taking falls back to VAD alone.
"""

import logging
from dataclasses import dataclass

from livekit.agents import LanguageCode

logger = logging.getLogger("agent-Avery-ff5")


# --- Supported languages ----------------------------------------------------
# One row per language the product supports, because "multilingual" is not a
# single switch: each language needs its own STT locale, its own recognition
# model (not every Google model covers every language), its own spoken opening
# line, and its own name to steer the LLM with. Previously `language` was
# threaded through the whole pipeline but never actually reached STT, so every
# call was transcribed as en-US no matter what the caller asked for.


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    """Canonical short code, as accepted in the API's `langage` field."""

    name: str
    """English name of the language, used to steer the LLM. The raw code is
    useless here -- "Speak only in bn" is a much weaker instruction than
    "Speak only in Bengali (Bangla)"."""

    bcp47: str
    """Locale handed to STT. Region matters: bn-BD and bn-IN are different
    recognition targets, and picking the wrong one costs real accuracy."""

    greeting: str
    """Spoken the instant the callee answers, straight to TTS. This has to be
    in the call's language or the whole call opens wrong -- and with Gemini
    TTS, which infers language from the text it is given, an English greeting
    also makes the first audio come out in an English voice."""

    stt_provider: str | None = None
    """Pin this language to a specific STT provider, overriding STT_PROVIDER.
    Left unset for every language today; the per-language escape hatch is the
    STT_PROVIDER_<CODE> environment variable. Set this only for a language the
    default provider genuinely cannot serve."""

    google_stt_streaming: bool = True
    """Whether Google's *streaming* recognizer works for this language.

    Confirmed on a live Bengali call: Google v1 StreamingRecognize returned no
    interim and no final result for the entire call, then delivered the whole
    minute of speech as a single transcript the moment the caller hung up. The
    agent had nothing to answer for the whole call, which reads to the caller
    as "the agent is broken".

    Setting this False makes the plugin advertise itself as non-streaming, and
    the framework then wraps it in stt.StreamAdapter, which uses the session's
    VAD to cut the audio into utterances and recognizes each one as it ends.
    Transcription starts after the person stops talking rather than while they
    speak -- slower, but it produces a turn, which streaming here did not.
    """

    google_stt_model: str = "default"
    """Google STT model for this language. `latest_long` is a v1 model and
    does not cover every language; the plugin picks the API version from this
    name (v2 for telephony/chirp_2/chirp_3, v1 otherwise), so this field
    decides both. Override per language with GOOGLE_STT_MODEL_<CODE>."""


# NOTE: the greetings below were written to match the English one's tone (warm,
# first-name, "calling to see how you are"). Have a native speaker review them
# before real calls -- this is the first thing an elderly person hears, and a
# stiff or oddly formal opening is exactly the wrong first impression. Override
# any of them per call with the API's `greeting` field.
LANGUAGES: dict[str, LanguageProfile] = {
    "en": LanguageProfile(
        code="en",
        name="English",
        bcp47="en-US",
        greeting="Hi there, it's Avery calling to see how you're doing today.",
        google_stt_model="latest_long",
    ),
    "bn": LanguageProfile(
        code="bn",
        name="Bengali (Bangla)",
        bcp47="bn-BD",
        greeting="হ্যালো, আমি অ্যাভেরি বলছি। আজ আপনি কেমন আছেন, তাই জানতে ফোন করলাম।",
        google_stt_model="default",
        google_stt_streaming=False,
    ),
    "es": LanguageProfile(
        code="es",
        name="Spanish",
        bcp47="es-US",
        greeting="Hola, soy Avery. Te llamo para ver cómo estás hoy.",
        google_stt_model="latest_long",
    ),
    "ar": LanguageProfile(
        code="ar",
        name="Arabic",
        bcp47="ar-EG",
        greeting="مرحباً، معك أيفري. اتصلت لأطمئن عليك اليوم.",
        # Not separately re-tested on a live call, but it shares Bengali's
        # situation exactly: a non-latest_long v1 model, and latest_long is the
        # only one observed to stream properly here. Flip to True and retest if
        # Arabic turns out to stream fine.
        google_stt_model="default",
        google_stt_streaming=False,
    ),
    "ms": LanguageProfile(
        code="ms",
        name="Malay",
        bcp47="ms-MY",
        greeting="Hai, ini Avery. Saya menelefon untuk bertanya khabar hari ini.",
        # Same reasoning as Bengali.
        google_stt_model="default",
        google_stt_streaming=False,
    ),
}

DEFAULT_LANGUAGE = "en"


# Languages the turn detector can actually score. Everything else falls back to
# VAD-only endpointing, which is a real quality difference, not a silent no-op
# -- see resolve_turn_detection() below.
TURN_DETECTOR_LANGUAGES = frozenset(
    {"ar", "de", "en", "es", "fr", "hi", "id", "it", "ja", "ko", "nl", "pt", "tr", "zh"}
)


def normalize_language(value: str | None) -> str:
    """Base language code for whatever the caller wrote, without deciding
    whether it's supported. "bn-BD", "bengali" and "bn" all give "bn"."""
    if not value:
        return DEFAULT_LANGUAGE
    try:
        return LanguageCode(value).language
    except Exception:
        return value.strip().lower().split("-")[0]


def is_supported_language(value: str | None) -> bool:
    """Whether this is a language we actually support, as opposed to one
    resolve_language() would quietly default to English."""
    return normalize_language(value) in LANGUAGES


def resolve_language(value: str | None) -> LanguageProfile:
    """Map whatever the caller sent in `langage` onto a supported profile.

    Accepts short codes ("bn"), locales ("bn-BD"), and English names
    ("bengali") -- the SDK's LanguageCode normalizes all three -- so a caller
    doesn't have to guess the exact spelling. Unknown languages fall back to
    the default rather than raising, because by the time this runs the phone
    is already ringing; api.py rejects them up front instead.
    """
    profile = LANGUAGES.get(normalize_language(value))
    if profile is None:
        logger.warning(
            "unsupported language %r, falling back to %s",
            value,
            DEFAULT_LANGUAGE,
        )
        return LANGUAGES[DEFAULT_LANGUAGE]
    return profile
