# avery-ff5 Production Readiness Audit

**Repo:** `avery-ff5` (LiveKit Cloud voice agent, "Avery" — an elder-companion check-in caller)
**Reviewed:** `src/agent.py`, `src/api.py`, `src/place_call.py`, `src/xtts_tts.py`, `tests/`, `Dockerfile`, `pyproject.toml`, `README.md`, `AGENTS.md`, `.gitignore`, `livekit.toml`
**Date:** 2026-09-08 (re-verified against the live repo later the same day — see verification notes)

## Summary

This is a well-structured, thoughtfully commented starter (generated from LiveKit's Agent Builder) that's had real engineering put into it — provider abstraction, non-root Docker build, a working test suite, careful `.gitignore` hygiene. The voice-agent worker half (`agent.py`, deployed to LiveKit Cloud) is close to production-ready as far as its own scope goes.

The gap is the other half: `src/api.py`, the outbound-call trigger service that a real backend (CRM, cron job, webhook) is meant to call to make Avery dial someone. It has **no authentication, no rate limiting, no input validation beyond types, in-memory-only state, and no deployment story** — yet its entire job is to spend money and place real phone calls to real (often elderly, vulnerable) people on arbitrary prompts. That combination is the main blocker to calling this "production grade."

Below: what's already solid, then gaps ordered by how much damage they can do, then a concrete roadmap.

## What's already solid

- **Docker build**: multi-stage, non-root `appuser`, dependency-layer caching, and a genuinely subtle bug (HF model cache ending up owned by `root` and invisible to the runtime user) already fixed and explained in a comment. This is better than most starter Dockerfiles.
- **Provider abstraction**: STT/LLM/TTS are each swappable via env vars (Google, ElevenLabs, Cartesia, OpenAI, Ollama, local Coqui XTTS) behind small factory functions (`_build_stt`/`_build_llm`/`_build_tts`) with sane fallbacks (e.g. Gemini API key → Vertex AI service account).
- **Lint/format**: `ruff check src tests` passes clean with a reasonably strict rule set (`E,F,W,I,N,B,A,C4,UP,SIM,RUF`).
- **Tests run and pass**: `tests/test_api.py`'s 4 tests pass end-to-end against a real `FastAPI` app via `httpx.ASGITransport` (no live LiveKit dispatch needed). `tests/test_agent.py` uses LLM-judge evals for conversational quality — a genuinely good pattern for voice-agent behavior, though it needs a live LiveKit API key to run (documented).
- **Secret hygiene in the repo itself**: `.env.local`, the service-account JSON, and `outbound-trunk.json`/`outbound-trunk copy.json` are now present in the local working tree (the developer has since set up real local-dev credentials) — but every one of them is confirmed correctly matched by a `.gitignore` rule (`git check-ignore -v` traced each to its specific line), so none of them are at risk of being committed. Good real-world validation that the `.gitignore` hardening actually works, not just that it looks right on paper.
- **SIP failure handling**: `entrypoint()` in `agent.py` distinguishes a rejected/busy call (`SipCallError`, has a SIP status code) from a lower-level trunk failure, logs appropriately, and reports either back through the callback instead of crashing the job.
- **Google credentials handling**: supports both a local key file and a `GOOGLE_CREDENTIALS_JSON` env var materialized to a temp file for cloud deploys where only string secrets are available — a real deployment problem solved cleanly.

## Critical gaps (P0 — block production)

**1. `POST /calls/outbound` has zero authentication.** Anyone who can reach this endpoint can make Avery call any phone number with any prompt, for free (to them) and at your cost. Given the target audience (elderly parents), this is also a harassment/abuse vector, not just a billing one. There is no API key, bearer token, mTLS, or network-boundary requirement anywhere in `api.py`.

**2. `POST /internal/calls/{call_id}/completed` is also unauthenticated.** It's meant to be called only by the agent worker, but nothing enforces that — the `call_id` is a UUID4 (unguessable), which helps, but this endpoint should still require a shared secret or HMAC signature, and ideally shouldn't be reachable from the public internet at all.

**3. No rate limiting or per-caller quota.** Nothing stops one client (malicious or buggy) from firing hundreds of outbound calls per second. Combined with #1, this is the single largest cost/abuse exposure in the project.

**4. Input validation stops at Pydantic's type checking.** `number` isn't validated as E.164, `prompt`/`helper_prompt` have no length cap, `user_id` is free text. A malformed number reaches LiveKit's SIP layer and fails there instead of being rejected at the edge with a clear 400.

**5. `_pending_calls` is an in-memory `dict` in a single process.** The README already flags this itself ("a restart of this service loses in-flight calls, and this only works with a single uvicorn worker process"), which is good self-awareness, but it's still true today. This blocks horizontal scaling, rolling deploys, and zero-downtime restarts — any of which will silently drop in-flight calls and leave the caller hanging until `CALL_TIMEOUT_SECONDS` (900s default).

**6. `api.py` has no deployment story at all.** The `Dockerfile`'s only `CMD` runs the agent worker (`uv run src/agent.py start`); nothing builds, containerizes, or documents how the outbound-call API itself gets deployed, scaled, or restarted. The README shows it being run with `uv run python src/api.py` in a second terminal — that's a dev instruction, not a production one.

**7. `.env.example` and `outbound-trunk.example.json` don't exist in the repo**, despite being referenced repeatedly by `README.md` and explicitly allow-listed in `.gitignore` (`!.env.example`, `!outbound-trunk.example.json`). Today, the only way to know every configuration knob (`STT_PROVIDER`, `LLM_PROVIDER`, `TTS_PROVIDER`, `SIP_OUTBOUND_TRUNK_ID`, `CALLBACK_BASE_URL`, `AI_COUSTICS_LICENSE_KEY`, `GOOGLE_CREDENTIALS_JSON`, …) is to read every `os.environ.get(...)` call across `agent.py`/`api.py`/`xtts_tts.py` by hand. This is a real operability risk during setup or incident response, not just a documentation nicety.

## High priority (P1 — needed before general availability)

- **No CI/CD.** There's no `.github/workflows` or equivalent — `ruff check` and `pytest` are documented as manual commands only, so nothing stops a broken or unlinted commit from reaching `main`/deploy.
- **No dependency vulnerability scanning.** `requirements.txt` pins ~400 transitive packages with no `pip-audit`/`safety`/Dependabot/Renovate wired up to catch known CVEs over time.
- **No health/readiness endpoint on `api.py`.** Nothing for a load balancer or orchestrator to probe before routing traffic or restarting the container.
- **No metrics or tracing on `api.py`**, despite `opentelemetry-*` and `prometheus-client` already sitting in `requirements.txt` as transitive dependencies (pulled in by `livekit-agents`) — they're present but not configured or exported anywhere. LiveKit's own Agent Observability (mentioned in the README) covers the voice pipeline itself, but the custom outbound-call service is a blind spot: no call-volume, latency, error-rate, or timeout metrics.
- **Logging is unstructured plain-text**, with no consistent correlation ID threading `call_id` through every log line across both `api.py` and `agent.py`. Fine for local dev, not for debugging a production incident across two processes.
- **Long-held synchronous HTTP requests.** `POST /calls/outbound` blocks the HTTP connection for up to `CALL_TIMEOUT_SECONDS` (900s default) waiting on a call to finish. At any real concurrency this exhausts worker/connection limits fast; a webhook-callback or polling pattern (return a call ID immediately, let the caller poll or receive a webhook) scales far better than holding connections open for 15 minutes.
- **Thin test coverage on the more complex file.** `agent.py` (576 lines — SIP dial-out, credential materialization, provider selection, session lifecycle) has no direct unit tests; only two end-to-end "does the reply sound natural" LLM-judge tests exist in `test_agent.py`, and those require a live LiveKit Cloud API key to run at all. `_build_stt`/`_build_llm`/`_build_tts` provider branching, `_materialize_google_credentials`, `_parse_dial_info`'s malformed-JSON path, and the SIP-failure callback path are all untested.
- **Python version drift.** `pyproject.toml` requires `>=3.10`, the checked-in `.venv` is 3.10.12, but the `Dockerfile` builds on 3.13. Nothing pins one version across dev and prod, which is exactly the kind of gap that produces a "works locally, breaks in prod" surprise.
- **`xtts_tts.py`'s local Coqui TTS provider fully serializes synthesis** behind one process-wide lock — correctly scoped to local/dev use in the README, but nothing technically prevents `TTS_PROVIDER=coqui` from being set in a production deployment and silently bottlenecking every concurrent call through one lock.

## Medium priority (P2 — hardening and polish)

- The `langage` field name (not `language`) in `OutboundCallRequest` is called out in code as intentionally matching "the request/response contract as given" — worth confirming with whoever owns that contract before more clients build against it, since it'll be an awkward thing to fix later.
- No audit trail of who triggered which call — `user_id` is free text, unauthenticated, and there's no persisted call log (this matters more once #1 is fixed, so you can tell who's calling on whose behalf).
- No graceful-shutdown handling for `api.py`'s in-flight `_pending_calls` — a `SIGTERM` mid-deploy can drop pending futures with no response ever sent to the caller.
- No load/capacity testing story for `api.py` itself — LiveKit Cloud scales the agent workers, but the custom trigger service is a single `uvicorn` process holding long-lived futures, which is the more likely bottleneck under real traffic.
- **Product/safety consideration, not just engineering**: this agent's whole purpose is a wellbeing check-in with elderly parents, and the prompt already tells it to "encourage professional medical care if symptoms sound serious" — but there's no automated flag or alert path if a call surfaces something urgent (a fall, chest pain, distress). Today that information only shows up buried in a `response_summary` string returned to whatever backend called the API. Worth deciding, as a product question, whether certain transcript content should trigger an active alert to a family member/caregiver rather than a passive summary.
- No `LICENSE` file — likely fine for an internal project, just flagging.

## Suggested roadmap

**Before any real (non-test) phone numbers are dialed through `api.py`:**
1. Add authentication to `POST /calls/outbound` (API key or bearer token at minimum; mTLS or a private network boundary if this sits behind an internal backend only).
2. Add HMAC/shared-secret verification to the `/internal/calls/{call_id}/completed` callback, and don't expose it publicly if avoidable.
3. Add per-key rate limiting / quota.
4. Validate `number` as E.164 and cap `prompt`/`helper_prompt` length at the API boundary.
5. Commit real `.env.example` and `outbound-trunk.example.json` templates (both are already allow-listed in `.gitignore` — they're just missing).

**Before calling this generally available:**
6. Move `_pending_calls` to Redis (or similar) so the API can run multiple replicas and survive restarts without dropping in-flight calls.
7. Write a `Dockerfile`/deployment target for `api.py` (or a second stage in the existing one) and document how it's actually deployed and scaled.
8. Add CI: run `ruff check` and `pytest` on every push/PR at minimum; add `pip-audit` or Dependabot for dependency CVEs.
9. Add a `/health` endpoint; wire up the already-installed OpenTelemetry/Prometheus stack for `api.py`'s own metrics (call volume, error rate, callback latency, timeout rate).
10. Pin one Python version across `pyproject.toml`, local dev, and the Dockerfile.
11. Switch `/calls/outbound` from "hold the connection open for up to 15 minutes" to an immediate-response-plus-webhook (or polling) pattern.

**Hardening / ongoing:**
12. Unit-test `agent.py`'s provider-selection branches and credential materialization directly (not just through the slower LLM-judge tests).
13. Add structured (JSON) logging with `call_id` threaded through every log line in both processes.
14. Decide on and implement an escalation path for concerning call content, given the target population.
15. Add a basic incident runbook (what to check first when calls are failing, timing out, or costing more than expected).

## Notes on verification

- **Re-verified end to end** against the live repo: `git diff HEAD --stat` is empty and `HEAD` is the identical commit (`917b667...`) reviewed originally — every tracked file is byte-for-byte what this audit describes. Nothing below is stale.
- Re-ran `ruff check src tests`: still passes with zero findings.
- Re-ran `tests/test_api.py` (`PYTHONPATH=src pytest tests/test_api.py`): still **4/4 pass.**
- Confirmed via `git check-ignore -v` that all locally-present secrets (`.env.local`, service-account JSON, both `outbound-trunk*.json` files) are correctly excluded from version control.
- Noted a new untracked `Claude outputs/` folder in the repo root — this is the desktop app's own sync of files delivered in this chat (this audit and its companions), not a code change. Gitignore it (or relocate it) if you don't want AI-session deliverables sitting alongside application code.
- Did not run `tests/test_agent.py`'s LLM-judge tests (require a live LiveKit Cloud API key per the project's own docs) or attempt a real Docker build/deploy — those remain reasonable next steps if you want this audit independently re-verified end-to-end.
