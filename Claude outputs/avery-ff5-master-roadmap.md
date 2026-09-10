# avery-ff5 — Master Findings Index (What's Now → What to Improve)

Consolidates the [production/infra audit](avery-ff5-production-audit.md) and the [agent design audit](avery-ff5-agent-design-audit.md) into one prioritized index. 34 findings, ordered by severity.

**Re-verified 2026-09-08** directly against the live repo: same git commit (`917b667...`), `git diff HEAD --stat` empty, `ruff check` and `tests/test_api.py` both still pass. Nothing below is stale — the codebase has not changed since the underlying audits were written.

**Applied 2026-09-08 (uncommitted):** #6, #11, #12, #13, #17, #19, #28, #33 — see ✅ notes below. Changes are in the working tree only, not yet committed; `ruff check`/`ruff format --check` and the full offline test suite (15/15) pass. #17/#33 landed as a scoped closing-phase handoff (a real `begin_wrap_up` tool → `ClosingAgent`), not a full 4-phase decomposition — see the note on row 17.

## The one to read first

| Area | What's true now | What to do |
|---|---|---|
| **Safety guardrails vs. real traffic** | `DefaultAgent`'s safety rules ("don't diagnose," "handle health disclosures carefully," graceful closing) only exist in the no-`prompt` branch — the one nothing in production dispatches. `api.py`, the actual outbound-call integration, always supplies a `prompt`, which drops every one of those rules and keeps only pacing/formatting. Verified against the framework source: nothing underneath adds them back. | Extract the safety/conduct rules into a shared constant, always prepended to *every* persona — default and caller-supplied alike. Highest-leverage single fix in either audit. |

## P0 — Blocks production

| # | Area | What's true now | What to do |
|---|---|---|---|
| 1 | Outbound-call auth | `POST /calls/outbound` has no API key, bearer token, or network boundary — anyone who can reach it can dial anyone, on your bill. | Add API-key/bearer auth at minimum; mTLS or private networking if it's backend-to-backend only. |
| 2 | Callback auth | `POST /internal/calls/{call_id}/completed` is also unauthenticated; only the unguessable UUID protects it. | Add HMAC/shared-secret verification; keep it off the public internet if possible. |
| 3 | Rate limiting | No rate limit or per-caller quota anywhere. | Add per-key rate limiting/quota. |
| 4 | Input validation | Pydantic checks types only — no E.164 check on `number`, no length cap on `prompt`/`helper_prompt`. | Validate phone format and cap field lengths at the API boundary. |
| 5 | Call-state durability | `_pending_calls` is an in-memory `dict` in one process — a restart drops every in-flight call (README already admits this). | Move to Redis or similar shared store. |
| 6 | Deployment story | ✅ Applied — `Dockerfile` now has a named `api` stage (`docker build --target api`) sharing the `agent` stage's base/deps; `agent` stays the default target so a plain `docker build .` is unchanged. Not build-verified (Docker isn't installed on the dev machine used) — verify with a real build before relying on it. | — |
| 7 | Missing config templates | `.env.example` and `outbound-trunk.example.json` are referenced throughout `README.md` and allow-listed in `.gitignore`, but don't exist in the repo — every env var has to be reverse-engineered from source. | Commit real template files for both. |

## P1 — Needed before general availability

| # | Area | What's true now | What to do |
|---|---|---|---|
| 8 | CI/CD | No `.github/workflows` or equivalent; `ruff`/`pytest` are manual-only commands. | Add CI running lint + tests (and ideally `pip-audit`) on every push/PR. |
| 9 | Dependency scanning | ~400 pinned transitive packages, no CVE monitoring. | Wire up `pip-audit`, Dependabot, or Renovate. |
| 10 | Health checks | No `/health`/`/ready` endpoint on `api.py`. | Add one for load balancer/orchestrator probing. |
| 11 | Metrics & tracing | ✅ Applied — `GET /metrics` on `api.py` exposes Prometheus counters/histogram (call volume, status, duration); OpenTelemetry tracing is wired but opt-in/no-op until `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Both libs added as explicit `pyproject.toml` deps (were transitive-only). | — |
| 12 | Logging | ✅ Applied — new `src/logging_utils.py`, opt-in via `LOG_FORMAT=json` (defaults to plain text so local dev / LiveKit's own CLI output is unchanged). `call_id` threaded via `extra=` through the relevant log lines in both `agent.py` and `api.py`. | — |
| 13 | Long-held requests | ✅ Applied — `POST /calls/outbound` now returns `202` immediately with `status: "pending"`; caller polls `GET /calls/{call_id}`. **This is a breaking API contract change** for any existing caller of the old blocking response — see README's updated "Outbound Phone Calls" section. A background watcher marks a call `"timeout"` if it's still pending after `CALL_TIMEOUT_SECONDS`. | — |
| 14 | Test coverage | `agent.py` (576 lines — SIP handling, credential materialization, provider selection) has no direct unit tests; only 2 e2e tests exist and need a live API key to even run. | Unit-test provider branching, credential materialization, dial-info parsing, and the SIP-failure path directly. |
| 15 | Environment parity | `pyproject.toml` requires `>=3.10`, local `.venv` is 3.10.12, `Dockerfile` builds on 3.13. | Pin one Python version across dev and prod. |
| 16 | Local TTS in prod | Coqui XTTS serializes all synthesis behind one process-wide lock; nothing stops `TTS_PROVIDER=coqui` from being set in a real deployment. | Guard against/document that this provider is dev-only. |
| 17 | Monolithic prompt | ⚠️ Partially applied — the *closing* phase is now a real handoff: `DefaultAgent.begin_wrap_up()` (a `@function_tool`, registered on both the default and caller-supplied personas) hands off to a dedicated `ClosingAgent`, so ending the call no longer depends on the model inferring it from prose alone. Greeting/check-in/concern-handling are still one prompt — a full 4-phase decomposition was judged higher-risk to land without live model testing (no LIVEKIT_API_KEY available to run the LLM-judge tests against the change). | Decompose the remaining phases the same way if warranted. |
| 18 | No tools/actions | `Agent.__init__` supports `tools=`; `DefaultAgent` registers none. The agent can't log a symptom, alert anyone, or look anything up mid-call — only a passive after-the-fact summary. | Add tools for real-time concern flagging / grounding in caller-specific data. |
| 19 | Unstructured call summary | ✅ Applied — `summarize_session()` now forces a `record_call_summary` tool call (`tool_choice="required"`) instead of free text, returning a `CallSummary{text, concern_level, flagged_topics}` with a graceful free-text fallback if the model doesn't call it. Threaded through `on_session_end`'s callback payload and `api.py`'s `GET /calls/{call_id}` response. | — |
| 20 | No call-length guard | No idle timeout or max-duration anywhere; the only `ctx.shutdown()` call is on SIP failure. A call runs until someone hangs up. | Add an idle-silence cutoff and/or hard max duration. |

## P2 — Hardening and polish

| # | Area | What's true now | What to do |
|---|---|---|---|
| 21 | API contract typo | `langage` (not `language`) is kept intentionally to match "the contract as given." | Confirm with the contract owner before more clients build against it. |
| 22 | Audit trail | No persisted, authenticated log of who triggered which call — `user_id` is free text. | Add a persisted call log once auth (P0 #1) exists. |
| 23 | Graceful shutdown | A `SIGTERM` mid-deploy can drop in-flight `_pending_calls` with no response ever sent. | Handle shutdown to fail pending futures cleanly instead of silently. |
| 24 | Capacity planning | No load/perf testing story for `api.py`'s single `uvicorn` process. | Load-test and define a capacity plan. |
| 25 | Concern escalation | Nothing actively alerts a caregiver on urgent content — only a passive summary field. (Product decision, not just engineering — echoed by #19 above.) | Decide on and implement an active escalation path. |
| 26 | License | No `LICENSE` file. | Add if relevant for this org. |
| 27 | Agent identity duplication | `"Avery-ff5"` is hardcoded independently in three files (`agent.py`, `api.py`, `place_call.py`). | Centralize as one shared constant. |
| 28 | Orchestration asymmetry | ✅ Applied — `place_call.py` now takes `--prompt`/`--helper-prompt`/`--language` flags, same override capability as `api.py`; still defaults to the built-in persona with no flags, so existing usage is unaffected. | — |
| 29 | Metadata parsing | `_parse_dial_info` only guards against invalid JSON, not valid-JSON-non-dict metadata — a bare string/array would throw `AttributeError` uncaught downstream. | Validate metadata shape defensively at the trust boundary. |
| 30 | Persona/brand mismatch | `BackgroundAudioPlayer` plays `OFFICE_AMBIENCE` under a persona introduced as "a caring family member... not a customer service representative." | Remove or replace with ambience that matches the persona. |
| 31 | Safety-rule verification | The (currently unused-in-prod) safety rules are prose-only — nothing checks the model actually followed them on a given call. | Add an output check/moderation pass once the guardrails are actually wired in (see top finding). |
| 32 | Multilingual coverage | `language` is threaded through the whole pipeline but only tested in English. | Add behavioral tests for at least one non-English language. |
| 33 | Closing-phase trigger | ✅ Applied — see #17: `begin_wrap_up()` is a real tool the model calls to trigger the handoff to `ClosingAgent`, not just prose asking it to "wrap up warmly." Offline-tested (`tests/test_agent.py`) that the handoff returns a `ClosingAgent` and that the tool is registered for both personas. | — |
| 34 | Deliverable artifacts in repo | A new untracked `Claude outputs/` folder now sits in the repo root, containing copies of these audit docs — the desktop app's own sync of files delivered in this chat, not a code change. | Add `Claude outputs/` to `.gitignore` (or relocate it) if you don't want AI-session deliverables tracked alongside application code. |

## How to use this

- Fixing the **top finding** and **#1–#4** together closes the biggest combined risk: an unauthenticated endpoint that can make the agent say anything, to anyone, with no safety floor.
- **#5–#7** are what stand between "works when I run it locally" and "survives a real deploy."
- **P1** is the general-availability bar; **P2** is ongoing hardening once the product is live.
