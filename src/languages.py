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

    stt_language_codes: tuple[str, ...] = ()
    """Languages the recognizer should expect on this call, most likely first.

    Every non-English profile lists English second on purpose. Bangla, Hindi
    and Arabic speakers routinely drop English words mid-sentence -- numbers,
    names, medicine names, "doctor", "appointment" -- and a recognizer locked
    to one language turns those into the nearest native-sounding nonsense,
    which is then what the LLM has to reason about. Listing both means the
    model expects the mix instead of fighting it."""

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

    elevenlabs_language_code: str | None = None
    """Language code handed to ElevenLabs STT, when it should differ from
    `code`, or the literal "auto" to send none at all.

    ElevenLabs takes a bare ISO-639 code, not a locale, so `code` is the right
    value for most languages. "auto" is different in kind: the plugin only
    adds `include_language_detection=true` to the websocket URL when no
    `language_code` is passed (see the elevenlabs plugin's `_connect_ws`), so
    this is the only way to let the model decide.

    That matters for Bengali specifically. Pinning `language_code=bn` produced
    no transcript at all on a live call while English -- same pipeline, same
    server-VAD settings, same build -- worked, and the failure is silent by
    construction: the plugin emits a FINAL_TRANSCRIPT only when the committed
    text is non-empty, so an empty commit reaches the session as an
    end-of-speech with nothing attached and no error anywhere. Auto-detect
    also matches what `stt_language_codes` has always described for this
    profile, which the ElevenLabs branch otherwise ignores entirely --
    Bengali speakers drop English words mid-sentence, and a connection pinned
    to one code cannot represent that.
    """

    google_tts_bcp47: str | None = None
    """Locale to hand Google Cloud TTS (TTS_PROVIDER=google) instead of
    `bcp47`, when they differ. Google's Chirp3-HD voices -- the named,
    same-character-across-languages voices (Leda, Kore, Aoede, ...; see
    GOOGLE_TTS_VOICE_<CODE> in .env.local) -- aren't available for every
    STT-side locale. bn-BD and ar-EG are not Chirp3-HD locales; bn-IN and
    ar-XA are. Sending bcp47 straight through for those two would pair a
    Chirp3-HD voice_name with a language_code it doesn't match, which the
    API rejects outright. None means "same as bcp47" (true for en/es/hi)."""


# NOTE: the greetings below were written to match the English one's tone (warm,
# first-name, "calling to see how you are"). Have a native speaker review them
# before real calls -- this is the first thing an elderly person hears, and a
# stiff or oddly formal opening is exactly the wrong first impression. Override
# any of them per call with the API's `greeting` field.
LANGUAGES: dict[str, LanguageProfile] = {
    "en": LanguageProfile(
        code="en",
        stt_language_codes=("en-US",),
        name="English",
        bcp47="en-US",
        greeting="Hi there, it's Avery calling to see how you're doing today.",
        google_stt_model="latest_long",
    ),
    "bn": LanguageProfile(
        code="bn",
        stt_language_codes=("bn-BD", "en-US"),
        name="Bengali (Bangla)",
        bcp47="bn-BD",
        greeting="হ্যালো, আমি অ্যাভেরি বলছি। আজ আপনি কেমন আছেন, তাই জানতে ফোন করলাম।",
        # Chirp3-HD (TTS_PROVIDER=google) doesn't have a bn-BD voice; bn-IN is
        # the closest Chirp3-HD-covered Bengali locale. See google_tts_bcp47's
        # docstring above.
        google_tts_bcp47="bn-IN",
        # See elevenlabs_language_code's docstring: pinning "bn" produced no
        # transcript on a live call while English worked. Override with
        # ELEVENLABS_STT_LANGUAGE_BN=bn to go back to pinning it.
        elevenlabs_language_code="auto",
        # Google's official v1 language table lists bn-BD under latest_long,
        # not just default/command_and_search -- confirmed 2026-09-09 by two
        # separate reads of the table (the earlier "latest_long doesn't cover
        # Bengali" note was never actually confirmed by a live API error, only
        # inferred from a less careful doc read). latest_long is the same
        # model en/es already stream successfully on this deployment, so this
        # is the one real chance to fix both the accuracy and the "waits
        # until you stop talking" latency in one change. Needs a live call to
        # confirm; if it's wrong, Google's API should reject it cleanly and
        # the previous default/False pairing is the known-safe fallback.
        google_stt_model="latest_long",
        google_stt_streaming=True,
    ),
    "es": LanguageProfile(
        code="es",
        stt_language_codes=("es-US", "en-US"),
        name="Spanish",
        bcp47="es-US",
        greeting="Hola, soy Avery. Te llamo para ver cómo estás hoy.",
        google_stt_model="latest_long",
    ),
    "ar": LanguageProfile(
        code="ar",
        stt_language_codes=("ar-EG", "en-US"),
        name="Arabic",
        bcp47="ar-EG",
        greeting="مرحباً، معك أيفري. اتصلت لأطمئن عليك اليوم.",
        # Chirp3-HD (TTS_PROVIDER=google) doesn't have an ar-EG voice; ar-XA
        # ("Generic Arabic") is the Chirp3-HD-covered locale. See
        # google_tts_bcp47's docstring above.
        google_tts_bcp47="ar-XA",
        # Same evidence and reasoning as Bengali above: Google's official v1
        # table lists ar-EG under latest_long. Not yet live-tested for Arabic
        # specifically.
        google_stt_model="latest_long",
        google_stt_streaming=True,
    ),
    "hi": LanguageProfile(
        code="hi",
        stt_language_codes=("hi-IN", "en-US"),
        name="Hindi",
        bcp47="hi-IN",
        greeting="नमस्ते, मैं एवरी बोल रही हूँ। आज आप कैसे हैं, यह जानने के लिए फोन किया है।",
        # bn-BD and ar-EG are both confirmed on Google's official v1 table
        # under latest_long (see above); hi-IN's row wasn't reachable in the
        # same doc fetch (page truncated before the H's), but Hindi is one of
        # Google's most broadly supported languages generally, so this is a
        # reasonable bet rather than a confirmed fact -- lower confidence than
        # bn/ar above. Revert to model="default", streaming=False (the
        # previous, known-safe pairing) if a live call shows it's wrong.
        # Hindi *is* in TURN_DETECTOR_LANGUAGES, so it gets real turn
        # detection regardless of which STT model is used.
        google_stt_model="latest_long",
        google_stt_streaming=True,
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
