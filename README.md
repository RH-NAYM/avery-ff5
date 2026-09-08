# LiveKit Agent

A voice AI project built with [LiveKit Agents for Python](https://github.com/livekit/agents) and [LiveKit Cloud](https://cloud.livekit.io/). This project is designed to work with coding agents like [Claude Code](https://claude.com/product/claude-code), [Cursor](https://www.cursor.com/), and [Codex](https://openai.com/codex/) — see [Coding agent support](https://docs.livekit.io/intro/coding-agents/) for setup tips.

> [!IMPORTANT]
> This project was converted to code from the LiveKit Agent Builder. The code is identical to production deployments from the builder. Follow the steps below to make it your own and deploy it to LiveKit Cloud. once you do so, you can delete the version in the builder.

## Next steps

### Run and deploy your agent

**Get your agent running locally and in production:**

1. **Run locally**: Follow the [Quickstart](#quickstart) section below to set up your environment and test the agent
2. **Deploy to production**: See the [Deploy to production](#deploy-to-production) section for deployment options and best practices

### Quickstart

**Get up and running** so you can start customizing:

1. **Install dependencies:**
   ```console
   uv sync
   ```

2. **Set up your LiveKit credentials:**

   Sign up for [LiveKit Cloud](https://cloud.livekit.io/), then configure your environment. You can either:

   - **Manual setup**: Copy `.env.example` to `.env.local` and fill in:
     - `LIVEKIT_URL`
     - `LIVEKIT_API_KEY`
     - `LIVEKIT_API_SECRET`
     - any provider API key listed in `.env.example` (realtime models are bring-your-own-key)

   - **Automatic setup** (recommended): Use the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/):
     ```bash
     lk cloud auth
     lk app env -w -d .env.local
     ```

3. **Download required models:**
   ```console
   uv run python src/agent.py download-files
   ```
   This downloads the model files used by the audio enhancement and noise cancellation plugins. [VAD](https://docs.livekit.io/agents/logic/turns/vad/) and [turn detection](https://docs.livekit.io/agents/logic/turns/turn-detector/) run in LiveKit Inference, so they need no local models.

4. **Test your agent:**
   ```console
   uv run python src/agent.py console
   ```
   This lets you speak to your agent directly in your terminal.

5. **Run for development:**
   ```console
   uv run python src/agent.py dev
   ```
   Use this when connecting to a frontend or telephony. This puts your agent into your LiveKit Cloud project, so use a different project if you don't want to affect production traffic.


## Local Self-Hosted Development

This project can run entirely on your own machine, with a self-hosted LiveKit Server, and **without a LiveKit Cloud account**. The agent still talks to external AI providers (Deepgram, Google Gemini, Cartesia) directly, using your own API keys, instead of routing through [LiveKit Inference](https://docs.livekit.io/agents/models/inference).

```text
Local LiveKit Server (livekit-server --dev)
        │  ws://localhost:7880
        ▼
LiveKit Agent Worker (src/agent.py)
        │
        ├── STT  → Deepgram API      (DEEPGRAM_API_KEY)
        ├── LLM  → Google Gemini API (GOOGLE_API_KEY)
        └── TTS  → Cartesia API      (CARTESIA_API_KEY)
```

### 1. Install LiveKit Server

**macOS:** `brew update && brew install livekit`
**Linux:** `curl -sSL https://get.livekit.io | bash`
**Windows:** download from the [latest release page](https://github.com/livekit/livekit/releases/latest)

Alternatively, if you'd rather not install anything system-wide, run it via Docker instead (no `docker compose` needed):
```console
docker run --rm -d --name livekit-dev \
  -p 7880:7880 -p 7881:7881 \
  -p 50000-50100:50000-50100/udp \
  livekit/livekit-server --dev --bind 0.0.0.0
```
- The UDP range is narrowed to `50000-50100` (instead of the full `50000-60000`) to avoid clashing with other apps on your machine (VPNs, remote-desktop tools, etc.) that may already be using a port somewhere in that range.
- `--bind 0.0.0.0` is required in Docker: `--dev` alone binds the server to the container's loopback address only, which Docker's port-forwarding can't reach, causing connection resets from the host.

### 2. Start LiveKit Server

```console
livekit-server --dev
```

This starts a signaling server at `ws://localhost:7880` with fixed dev credentials (`devkey` / `secret`) — no cloud account involved.

### 3. Configure `.env.local`

Copy `.env.example` to `.env.local` and fill in your own Deepgram, Google, and Cartesia API keys. The LiveKit values already default to the local dev server:
```console
cp .env.example .env.local
```

### 4. Install Python dependencies

```console
uv sync
```

### 5. Download local model files

VAD and turn detection run locally (not via LiveKit Inference), so their model files need to be downloaded once:
```console
uv run python src/agent.py download-files
```

### 6. Start the agent

```console
uv run python src/agent.py dev
```

With `LIVEKIT_URL=ws://localhost:7880` in `.env.local`, the worker registers with your local server instead of LiveKit Cloud.

### 7. Connect a client

The agent worker only joins rooms — something still needs to create a room and connect a participant to it. Use `uv run python src/agent.py console` for a quick terminal-based mic/speaker test without any server or frontend, or point one of the [frontend starter templates](#frontend-development) at `ws://localhost:7880` with the local `devkey`/`secret` to test a full room-based session.

### Troubleshooting

- **`docker: ... ports are not available` / `bind: address already in use`**: another app on your machine already holds a port inside the UDP range you asked Docker to publish. Narrow the range (e.g. `50000-50100`) instead of publishing the full `50000-60000`.
- **Console mode connects but the server resets the connection**: if running LiveKit Server in Docker, make sure `--bind 0.0.0.0` is passed — otherwise the server only listens on the container's loopback address, which Docker's port-forwarding can't reach.
- **`console` mode fails with `PortAudioError: Invalid sample rate`**: your OS's default audio input device is a raw ALSA hardware device that only supports its native sample rate (no resampling), while LiveKit needs a different rate. List devices and pick your system's audio server device (e.g. `pipewire` or `pulse`) instead, which resamples automatically:
  ```console
  uv run python src/agent.py console --list-devices
  uv run python src/agent.py console --input-device <id> --output-device <id>
  ```
  `--input-device`/`--output-device` are optional — omit them to use the system default, or pass device IDs from `--list-devices` if the default fails (e.g. `uv run python src/agent.py console --input-device 7 --output-device 7` for a `pipewire` device).

### What's not covered here

- **ai-coustics noise cancellation** is billed either through a LiveKit Cloud project or your own ai-coustics license. Locally, it's skipped unless you set `AI_COUSTICS_LICENSE_KEY` in `.env.local`.
- This setup is for local development only — no TLS, TURN, load balancing, or production hardening. See [Deploy to production](#deploy-to-production) for that.

## Coqui XTTS v2 (local TTS)

This project also supports [Coqui XTTS v2](https://huggingface.co/coqui/XTTS-v2) as a `TTS_PROVIDER`: a local, open-weight text-to-speech model that runs on your own CPU/GPU instead of calling a cloud API (no API key needed). It supports voice cloning from a short reference clip. Because it depends on `torch`, a large hardware-specific package, it's kept out of the project's default `uv`-managed dependencies and installed separately — typically into its own conda environment.

### 1. Create/use a conda environment

```console
conda create -n avery-ff5 python=3.11
conda activate avery-ff5
```

### 2. Install PyTorch

Install the build that matches your hardware using the [official instructions](https://pytorch.org/get-started/locally/). For an NVIDIA GPU:

```console
pip install torch torchaudio
```

`pip install torch` on Linux installs a CUDA-enabled build by default. For CPU-only, use the CPU index instead: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`. CPU inference works but is significantly slower than realtime, which matters for a voice agent.

### 3. Install this project + Coqui XTTS v2

```console
cd "Avery-ff5"
pip install -e ".[coqui]"
```

### 4. Configure `.env.local`

```console
TTS_PROVIDER=coqui
# Optional overrides (defaults shown):
# COQUI_XTTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
# COQUI_XTTS_LANGUAGE=en
# COQUI_XTTS_SPEAKER=Claribel Dervla   # one of XTTS v2's built-in voices
# COQUI_XTTS_SPEAKER_WAV=/path/to/reference.wav  # clone a voice instead
# COQUI_XTTS_DEVICE=auto               # auto | cuda | cpu
```

Set either `COQUI_XTTS_SPEAKER` (a built-in voice name) or `COQUI_XTTS_SPEAKER_WAV` (a ~6-30s reference clip to clone), not both.

`pip install -e ".[coqui]"` pins `transformers<5` and adds `torchcodec` alongside `coqui-tts`, since coqui-tts 0.27.x's XTTS code isn't compatible with the `transformers` 5.x API (`ImportError: cannot import name 'isin_mps_friendly'`) and `torch>=2.9` requires `torchcodec` for audio I/O. If you installed `coqui-tts` some other way and hit either error, install those two constraints manually.

### 5. Run the agent from that environment

> [!IMPORTANT]
> Do **not** prefix this with `uv run`. `uv run` always creates/uses its own separate `.venv` for this project (even with the conda env active) and will not see `torch`/`coqui-tts`, which only exist inside the `avery-ff5` conda env — you'll hit `ModuleNotFoundError: No module named 'torch'`. Call `python` directly instead.

```console
conda activate avery-ff5
python src/agent.py console
```

The first synthesis triggers a one-time ~1.8 GB model download to `~/.local/share/tts`, and requires accepting the [Coqui Public Model License](https://coqui.ai/cpml) (auto-accepted non-interactively by this integration — read the license before using it in production).

## Outbound Phone Calls (Twilio)

This project supports placing outbound phone calls through [LiveKit SIP](https://docs.livekit.io/sip/) backed by a Twilio Elastic SIP Trunk. The agent doesn't call Twilio's API directly — Twilio hands audio to LiveKit's SIP service over SIP/RTP, which creates a room and puts the callee in it as a regular participant, indistinguishable from a web/console participant to `entrypoint()` in `agent.py`.

> [!IMPORTANT]
> This requires a **LiveKit Cloud** project — SIP is pre-configured there. A local `livekit-server --dev` instance doesn't include it; self-hosting SIP yourself means running a separate `livekit-sip` server plus Redis, with SIP (port 5060) and RTP (10000-20000) reachable from the public internet, which a home network typically can't do. If you're currently using [local self-hosted development](#local-self-hosted-development), switch `LIVEKIT_URL`/`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET` in `.env.local` to your LiveKit Cloud project's values for outbound calling (see [Automatic setup](#quickstart) with `lk cloud auth`).

### 1. Twilio: create an Elastic SIP Trunk for Termination (outbound)

1. In the Twilio Console, go to **Elastic SIP Trunking** → **Trunks** → create a new trunk.
2. Open the **Termination** tab and set a unique Termination SIP URI domain, e.g. `my-trunk.pstn.twilio.com`.
3. Under **Authentication**, create (or select) a **Credential List** with a username/password — LiveKit will authenticate to Twilio with these.
4. Make sure the Twilio phone number you're calling *from* is associated with this trunk.

### 2. LiveKit Cloud: create an outbound SIP trunk

Install/update the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/) (`lk --version` >= 2.15.0), then:

```console
lk cloud auth
```

Create `outbound-trunk.json`:

```json
{
  "trunk": {
    "name": "Twilio outbound",
    "address": "my-trunk.pstn.twilio.com",
    "numbers": ["+15105550100"]
  }
}
```

```console
lk sip outbound create outbound-trunk.json \
  --auth-user "<twilio credential list username>" \
  --auth-pass "<twilio credential list password>"
```

> [!NOTE]
> Some destinations (e.g. Bangladesh) require Twilio "Secure Trunking": TLS-encrypted signaling **and** SRTP media, or calls fail with `488 ... TLS transport is required to place a secure call` and then, once TLS is added, `488 ... SIP trunk or domain is required to use secure media (SRTP)`. Fix both at once with `--transport tls --media-enc require` on the `create` command above (or `"transport": "SIP_TRANSPORT_TLS", "mediaEncryption": "SIP_MEDIA_ENCRYPT_REQUIRE"` in the JSON) and recreate the trunk.

Copy the returned `SIPTrunkID` (looks like `ST_xxxxxxxx`) into `.env.local`:

```console
SIP_OUTBOUND_TRUNK_ID=ST_xxxxxxxx
```

### 3. Place a call

```console
conda activate avery-ff5   # or your uv-managed env, if not using Coqui XTTS v2
python src/place_call.py +12135550100
```

This dispatches the agent (`src/place_call.py`) into a new room with the phone number in the job's metadata. `entrypoint()` in `agent.py` reads `phone_number` from that metadata, dials out via `ctx.api.sip.create_sip_participant(...)` with `wait_until_answered=True`, and only starts the conversation once the call is answered. A busy signal, decline, or no-answer raises `api.SipCallError` and the job shuts down without starting a session — check the agent's logs for the SIP status code/reason.

### 4. Or trigger calls from a backend via the outbound call API

`src/api.py` is a small FastAPI service for triggering a call from a real backend (a webhook, cron job, CRM integration) and getting a summary back once the conversation is over, instead of using the `place_call.py` CLI. Run it alongside the agent worker:

```console
# terminal 1: the agent worker (as in step 3 above, but via `dev` so it keeps running)
python src/agent.py dev

# terminal 2: the API
uv run python src/api.py
```

`API_HOST` / `API_PORT` (defaults `0.0.0.0` / `8000`) override the bind address if needed.

Then:

```console
curl -X POST http://localhost:8000/calls/outbound \
  -H "Content-Type: application/json" \
  -d '{
    "call_type": "outbound",
    "user_id": "user-123",
    "number": "+12135550100",
    "prompt": "You are Avery, calling to check in on ...",
    "helper_prompt": "The user prefers short calls.",
    "langage": "en"
  }'
```

The request blocks until the call finishes (or `CALL_TIMEOUT_SECONDS`, default 900, elapses), then returns the same body with a `response_summary` field added. Unlike `place_call.py`'s fixed elder-companion persona, `prompt` here becomes the agent's full instructions for that call — `helper_prompt` is appended as supplementary context, and `langage` selects the STT/TTS language.

Under the hood, `create_outbound_call` dispatches the agent with a `callback_url` in the job metadata pointing back at this API. When the conversation ends, the agent's `on_session_end` callback (in `agent.py`) summarizes `session.history` with a separate LLM call and POSTs it to that URL, which resolves the waiting request. Because that callback needs to reach this API's process, set `CALLBACK_BASE_URL` in `.env` (or the agent worker's environment) to wherever this API is actually reachable — the default `http://localhost:8000` only works when both processes share a host, which won't be true once the agent worker is deployed separately (for example, via `lk agent create`).

### What's not covered here

- **Inbound calls** (someone calls your Twilio number and reaches the agent) need an inbound SIP trunk and a [dispatch rule](https://docs.livekit.io/sip/dispatch-rule/) instead of explicit dispatch — see [Accepting inbound calls](https://docs.livekit.io/sip/accepting-calls/).
- **Restart resilience / multiple API processes** — `api.py` tracks in-flight calls in an in-memory dict, so a restart loses them and it only works behind a single process. Move `_pending_calls` to a shared store (Redis, a database) if you need either.

## Customize your agent

Once your agent is running, enhance it for your use case:

- **Customize AI models**: Your agent uses a voice AI pipeline built on [LiveKit Inference](https://docs.livekit.io/agents/models/inference). More than 50 model providers are supported, including [Realtime models](https://docs.livekit.io/agents/models/realtime).

- **Add tests**: You can add a full test suite to your agent. See the [testing documentation](https://docs.livekit.io/agents/start/testing/) for more information.

- **Build reliable workflows**: For complex agents, use [tasks and handoffs](https://docs.livekit.io/agents/build/workflows/) instead of long instruction prompts. This minimizes latency and improves reliability by structuring your agent into focused, reusable components.

### Get help from AI coding assistants

**Supercharge your development** with AI coding assistants that understand LiveKit. This project works seamlessly with [Claude Code](https://claude.com/product/claude-code), [Cursor](https://www.cursor.com/), [Codex](https://openai.com/codex/), and other AI coding tools.

For your convenience, LiveKit offers both a CLI and an [MCP server](https://docs.livekit.io/reference/developer-tools/docs-mcp/) that can be used to browse and search its documentation. The [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/) (`lk docs`) works with any coding agent that can run shell commands. Install it for your platform:

**macOS:**

```console
brew install livekit-cli
```

**Linux:**

```console
curl -sSL https://get.livekit.io/cli | bash
```

**Windows:**

```console
winget install LiveKit.LiveKitCLI
```

The `lk docs` subcommand requires version 2.15.0 or higher. Check your version with `lk --version` and update if needed. Once installed, your coding agent can search and browse LiveKit documentation directly from the terminal:

```console
lk docs search "voice agents"
lk docs get-page /agents/start/voice-ai-quickstart
```

See the [Using coding agents](https://docs.livekit.io/intro/coding-agents/) guide for more details, including MCP server setup.

**Customize the AI assistant context**: The project includes an [AGENTS.md](AGENTS.md) file that guides AI assistants on how to work with this codebase. **Edit this file** to add your own project-specific context, patterns, and preferences. Learn more at [https://agents.md](https://agents.md).

## Frontend development

If you don't alread have a frontend, use the following templates and guides to get started on one:

| Platform | Starter Template | What to customize |
|----------|----------|-------------|
| **Web** | [`livekit-examples/agent-starter-react`](https://github.com/livekit-examples/agent-starter-react) | React & Next.js—customize UI, add features, integrate with your backend |
| **iOS/macOS** | [`livekit-examples/agent-starter-swift`](https://github.com/livekit-examples/agent-starter-swift) | Native apps for iOS, macOS, visionOS—add platform-specific features |
| **Flutter** | [`livekit-examples/agent-starter-flutter`](https://github.com/livekit-examples/agent-starter-flutter) | Cross-platform—customize for Android, iOS, web, desktop |
| **React Native** | [`livekit-examples/voice-assistant-react-native`](https://github.com/livekit-examples/voice-assistant-react-native) | Mobile with Expo—add native modules, customize navigation |
| **Android** | [`livekit-examples/agent-starter-android`](https://github.com/livekit-examples/agent-starter-android) | Kotlin & Jetpack Compose—build Material Design UI |
| **Web Embed** | [`livekit-examples/agent-starter-embed`](https://github.com/livekit-examples/agent-starter-embed) | Widget for any website—customize styling, add to your site |
| **Telephony** | [Documentation](https://docs.livekit.io/telephony/) | Add phone calling—configure SIP, add call routing, customize prompts |

## Observability

LiveKit provides deep session insights for your agents through [Agent Observability](https://docs.livekit.io/deploy/observability/). Monitor conversation quality, track latency metrics, and debug agent behavior in production.

## Deploy to production

To deploy your agent to production, you can use the LiveKit CLI:

```console
lk agent create
```

See the [deploying to production](https://docs.livekit.io/deploy/agents/) guide for detailed instructions and optimization tips.

## Join the LiveKit community

Join the [LiveKit Slack Community](https://livekit.io/join-slack) to get help from the LiveKit team and other developers.
