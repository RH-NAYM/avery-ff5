import json
import logging
import os

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
    TurnHandlingOptions,
    cli,
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
from livekit.plugins.google.beta import GeminiTTS
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent-Avery-ff5")

load_dotenv(".env.local")  # local dev config, per README (git-ignored)
load_dotenv(".env")  # optional fallback for anything not in .env.local

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


def _build_stt(language: str = "en"):
    provider = os.environ.get("STT_PROVIDER", "google").lower()
    if provider == "elevenlabs":
        return elevenlabs.STT(
            api_key=_require_env("ELEVENLABS_API_KEY"),
            model=os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1"),
        )
    if provider == "google":
        # Google Cloud Speech-to-Text, authenticated with a service account
        # key file (see GOOGLE_APPLICATION_CREDENTIALS in .env.local). Same
        # service account used by the Vertex AI fallback for
        # LLM_PROVIDER=gemini / TTS_PROVIDER=gemini below.
        return google.STT(
            credentials_file=_require_env("GOOGLE_APPLICATION_CREDENTIALS")
        )
    raise ValueError(
        f"Unknown STT_PROVIDER: {provider!r} (expected 'elevenlabs' or 'google')"
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


def _build_tts(language: str = "en"):
    provider = os.environ.get("TTS_PROVIDER", "elevenlabs").lower()
    if provider == "elevenlabs":
        return elevenlabs.TTS(
            voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "hpp4J3VqNfWAUOO0d1Us"),
            model=os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5"),
            api_key=_require_env("ELEVENLABS_API_KEY"),
            language=language,
            # Lets the <break time="..."/> tags in DefaultAgent's instructions
            # (see _supports_ssml_breaks) render as actual pauses instead of
            # being read aloud.
            enable_ssml_parsing=True,
        )
    if provider == "gemini":
        gemini_tts_api_key = os.environ.get("GEMINI_API_KEY")
        if gemini_tts_api_key:
            return GeminiTTS(
                voice_name=os.environ.get("GEMINI_TTS_VOICE", "Kore"),
                api_key=gemini_tts_api_key,
            )
        # No API key: fall back to Vertex AI using a Google Cloud service
        # account (GOOGLE_APPLICATION_CREDENTIALS / GOOGLE_CLOUD_PROJECT /
        # GOOGLE_CLOUD_LOCATION), same as LLM_PROVIDER=gemini above. The
        # project is inferred from the service account key file and the
        # location defaults to "us-central1" if not set.
        return GeminiTTS(
            voice_name=os.environ.get("GEMINI_TTS_VOICE", "Kore"),
            vertexai=True,
        )
    if provider == "cartesia":
        return cartesia.TTS(
            model="sonic-3",
            voice="a167e0f3-df7e-4d52-a9c3-f949145efdab",
            language=language,
            api_key=_require_env("CARTESIA_API_KEY"),
        )
    if provider == "google":
        # Google Cloud Text-to-Speech, authenticated with a service account
        # key file (see GOOGLE_APPLICATION_CREDENTIALS in .env.local). This
        # is distinct from TTS_PROVIDER=gemini above, which uses Gemini's
        # own TTS model instead of the Cloud Text-to-Speech API.
        return google.TTS(
            credentials_file=_require_env("GOOGLE_APPLICATION_CREDENTIALS"),
            voice_name=os.environ.get("GOOGLE_TTS_VOICE") or NOT_GIVEN,
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


def _voice_realism_instructions(language: str = "en") -> str:
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
    language_line = (
        f"* Speak only in {language}, regardless of what language this prompt is written in.\n"
        if language.lower() not in ("en", "en-us", "english")
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
    ) -> None:
        if prompt:
            # Caller-supplied persona (see the outbound call API in api.py):
            # `prompt` is the full identity/goal instructions, `helper_prompt`
            # is optional supplementary context. Voice pacing/formatting
            # rules still apply on top so it sounds like a phone call.
            sections = [prompt.strip()]
            if helper_prompt and helper_prompt.strip():
                sections.append(f"# Additional context\n\n{helper_prompt.strip()}")
            sections.append(_voice_realism_instructions(language))
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

                {_voice_realism_instructions(language)}

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

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="""Greet the user and offer your assistance.""",
            allow_interruptions=True,
        )


def _parse_dial_info(ctx: JobContext) -> dict:
    # Outbound phone calls are triggered by dispatching this agent with JSON
    # metadata. `phone_number` alone is set by src/place_call.py for manual
    # testing; the full set (call_id, prompt, helper_prompt, language,
    # callback_url) is set by the outbound call API in api.py. Regular
    # sessions (console, web/mobile frontends) have no metadata at all.
    if not ctx.job.metadata:
        return {}
    try:
        return json.loads(ctx.job.metadata)
    except json.JSONDecodeError:
        logger.warning("ignoring non-JSON job metadata: %r", ctx.job.metadata)
        return {}


async def _post_callback(callback_url: str, payload: dict) -> None:
    """Best-effort POST back to the outbound call API (see api.py) so it can
    return a response to its caller instead of blocking until its timeout.
    Never raises: a failed callback should not crash the job."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()
    except httpx.HTTPError:
        logger.exception("failed to post call-completed callback to %s", callback_url)


async def summarize_session(summarizer, chat_ctx: ChatContext) -> str | None:
    """Generate a brief summary of the user/assistant turns using a separate,
    non-conversational LLM call. Based on the "Summarizing context" pattern
    in the LiveKit Agents docs (agents/logic/agents-handoffs)."""
    summary_ctx = ChatContext()
    summary_ctx.add_message(
        role="system",
        content=(
            "Summarize the following phone conversation for a business "
            "record. Be factual and concise, 2-4 sentences."
        ),
    )

    n_summarized = 0
    for item in chat_ctx.items:
        if item.type != "message" or item.role not in ("user", "assistant"):
            continue
        text = (item.text_content or "").strip()
        if text:
            summary_ctx.add_message(role="user", content=f"{item.role}: {text}")
            n_summarized += 1

    if n_summarized == 0:
        return None

    response = await summarizer.chat(chat_ctx=summary_ctx).collect()
    return response.text.strip() if response.text else None


async def on_session_end(ctx: JobContext) -> None:
    dial_info = _parse_dial_info(ctx)
    call_id = dial_info.get("call_id")
    callback_url = dial_info.get("callback_url")
    if not call_id or not callback_url:
        # No outbound call API is waiting on this job (console/web session,
        # or a call placed via place_call.py) -- nothing to report back.
        return

    session = ctx.primary_session
    if session is None:
        await _post_callback(
            callback_url, {"call_id": call_id, "error": "session never started"}
        )
        return

    try:
        summary = await summarize_session(session.llm, session.history)
    except Exception:
        logger.exception("failed to summarize session for call %s", call_id)
        summary = None

    await _post_callback(
        callback_url, {"call_id": call_id, "response_summary": summary or ""}
    )


server = AgentServer()


@server.rtc_session(agent_name="Avery-ff5", on_session_end=on_session_end)
async def entrypoint(ctx: JobContext):
    dial_info = _parse_dial_info(ctx)

    phone_number = dial_info.get("phone_number")
    call_id = dial_info.get("call_id")
    callback_url = dial_info.get("callback_url")
    language = dial_info.get("language") or "en"

    session = AgentSession(
        stt=_build_stt(language),
        llm=_build_llm(),
        tts=_build_tts(language),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            # "dynamic" adapts the end-of-turn wait to each caller's actual
            # pause patterns instead of always waiting the full min_delay,
            # so replies come faster for callers who don't pause much.
            endpointing={"mode": "dynamic", "min_delay": 0.5, "max_delay": 3.0},
            preemptive_generation={
                "enabled": True,
                # Start TTS before the turn is fully confirmed, not just the
                # LLM. Cuts more latency at the cost of occasionally wasted
                # synthesis when a preemptive guess gets discarded.
                "preemptive_tts": True,
            },
            # Adaptive interruption uses an audio model to tell real
            # barge-ins apart from backchannel acknowledgments ("mm-hmm",
            # "yeah"), so the agent doesn't stop mid-sentence for those. This
            # calls LiveKit Cloud's inference gateway, which phone calls
            # already require (see SIP_OUTBOUND_TRUNK_ID / README) — it only
            # adds a cloud dependency for fully self-hosted local testing.
            interruption={"mode": "adaptive", "min_duration": 0.5, "min_words": 0},
        ),
        vad=silero.VAD.load(),
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
        except api.SipCallError as e:
            logger.warning(
                "outbound call to %s failed: %s %s",
                phone_number,
                e.sip_status_code,
                e.sip_status,
            )
            if call_id and callback_url:
                # on_session_end won't fire since no session ever started;
                # report the failure directly so the outbound call API
                # doesn't just block until its timeout.
                await _post_callback(
                    callback_url,
                    {
                        "call_id": call_id,
                        "error": f"sip call failed: {e.sip_status_code} {e.sip_status}",
                    },
                )
            ctx.shutdown(reason="sip call failed")
            return

    await session.start(
        agent=DefaultAgent(
            prompt=dial_info.get("prompt"),
            helper_prompt=dial_info.get("helper_prompt"),
            language=language,
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(**room_options_kwargs),
    )

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=1.0)
    )
    await background_audio.start(room=ctx.room, agent_session=session)


if __name__ == "__main__":
    cli.run_app(server)
