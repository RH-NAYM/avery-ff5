import json
import logging
import os

from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    TurnHandlingOptions,
    cli,
    room_io,
)
from livekit.plugins import (
    ai_coustics,
    cartesia,
    deepgram,
    elevenlabs,
    google,
    openai,
    silero,
)
from livekit.plugins.google.beta import GeminiTTS
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent-Avery-ff5")

load_dotenv(".env")

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


def _build_stt():
    provider = os.environ.get("STT_PROVIDER", "deepgram").lower()
    if provider == "deepgram":
        return deepgram.STT(model="nova-3", language="en", api_key=_require_env("DEEPGRAM_API_KEY"))
    if provider == "elevenlabs":
        return elevenlabs.STT(
            api_key=_require_env("ELEVENLABS_API_KEY"),
            model=os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1"),
        )
    raise ValueError(f"Unknown STT_PROVIDER: {provider!r} (expected 'deepgram' or 'elevenlabs')")


def _build_llm():
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        return google.LLM(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            api_key=_require_env("GEMINI_API_KEY"),
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


def _build_tts():
    provider = os.environ.get("TTS_PROVIDER", "elevenlabs").lower()
    if provider == "elevenlabs":
        return elevenlabs.TTS(
            voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "hpp4J3VqNfWAUOO0d1Us"),
            model=os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5"),
            api_key=_require_env("ELEVENLABS_API_KEY"),
        )
    if provider == "gemini":
        return GeminiTTS(
            voice_name=os.environ.get("GEMINI_TTS_VOICE", "Kore"),
            api_key=_require_env("GEMINI_API_KEY"),
        )
    if provider == "cartesia":
        return cartesia.TTS(
            model="sonic-3",
            voice="a167e0f3-df7e-4d52-a9c3-f949145efdab",
            language="en",
            api_key=_require_env("CARTESIA_API_KEY"),
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
        "(expected 'elevenlabs', 'gemini', 'cartesia', or 'coqui')"
    )


class DefaultAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a warm, caring family companion speaking with an elderly parent on behalf of their son or daughter.

                Your personality:

                * Speak naturally, warmly, and respectfully.
                * Sound like a caring family member checking in, not a survey agent or customer service representative.
                * Show genuine interest in the person's day and well-being.
                * Keep the conversation relaxed and friendly.

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
                """,
                    )
    async def on_enter(self):
        await self.session.generate_reply(
            instructions="""Greet the user and offer your assistance.""",
            allow_interruptions=True,
        )


server = AgentServer()
@server.rtc_session(agent_name="Avery-ff5")
async def entrypoint(ctx: JobContext):
    # Outbound phone calls are triggered by dispatching this agent with
    # metadata like `{"phone_number": "+15105550100"}` (see
    # src/place_call.py). Regular sessions (console, web/mobile frontends)
    # have no metadata and proceed unchanged.
    dial_info: dict = {}
    if ctx.job.metadata:
        try:
            dial_info = json.loads(ctx.job.metadata)
        except json.JSONDecodeError:
            logger.warning("ignoring non-JSON job metadata: %r", ctx.job.metadata)

    phone_number = dial_info.get("phone_number")

    session = AgentSession(
        stt=_build_stt(),
        llm=_build_llm(),
        tts=_build_tts(),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            preemptive_generation={"enabled": True},
            # Force local VAD-based interruption instead of the
            # AdaptiveInterruptionDetector, which calls out to LiveKit
            # Cloud's inference gateway even outside of console mode.
            interruption={"mode": "vad"},
        ),
        vad=silero.VAD.load(),
    )

    room_options_kwargs = {}
    if _AI_COUSTICS_LICENSE_KEY:
        room_options_kwargs["audio_input"] = room_io.AudioInputOptions(
            noise_cancellation=ai_coustics.audio_enhancement(
                model=ai_coustics.EnhancerModel.QUAIL_VF_L,
                auth=ai_coustics.Auth.ai_coustics_api(license_key=_AI_COUSTICS_LICENSE_KEY),
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
            ctx.shutdown(reason="sip call failed")
            return

    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(**room_options_kwargs),
    )

    background_audio = BackgroundAudioPlayer(ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=1.0))
    await background_audio.start(room=ctx.room, agent_session=session)


if __name__ == "__main__":
    cli.run_app(server)
