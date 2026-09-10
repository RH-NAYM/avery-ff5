# avery-ff5 — Agent Design Audit (Architecture, Behavior, Orchestration)

**Date:** 2026-09-08 (re-verified against the live repo later the same day — see verification notes)

This is a second pass, separate from the earlier infra/security audit — this one looks purely at the *agent* itself: how it's built, how it behaves, and how sessions get orchestrated. Reviewed against `src/agent.py`, `AGENTS.md`'s own stated guidance, and the installed `livekit-agents` framework internals (verified directly against the library source, not assumed).

## Headline finding: the safety guardrails only exist for the persona nobody actually uses in production

`DefaultAgent.__init__` builds its `instructions` string one of two ways:

- **No `prompt` passed** (only reachable from `place_call.py`, a manual CLI test script): the full elder-companion persona — warmth, "ask one question at a time," "don't diagnose," "encourage professional medical care if symptoms sound serious," how to close the call, what to summarize.
- **A `prompt` is passed** (the *only* path `src/api.py` — the actual product-facing outbound-call API — ever uses): `instructions` becomes just `prompt.strip()` + optional `helper_prompt` + `_voice_realism_instructions()` (pacing/formatting rules only — "keep replies short," "use `<break>` tags," etc.).

I checked whether the framework injects any base/system-level instructions underneath this — it doesn't. `Agent.__init__` in the installed `livekit-agents` package does `self._instructions = instructions` verbatim, with nothing added. So every guardrail that makes the default persona look well thought out — the "don't diagnose," the "encourage professional care," the graceful conversational closing — **applies to zero real outbound calls**, because real outbound calls always go through `api.py`, which always supplies a `prompt`, which always takes the branch that drops every one of those rules. What survives is only sentence-length and filler-word pacing.

Practically: today, whoever calls `POST /calls/outbound` fully controls Avery's entire persona and conduct for that call, with no floor underneath it — not "don't give medical advice," not "handle disclosed distress carefully," nothing. Combined with the missing auth on that endpoint (flagged in the earlier audit), this is the same underlying gap wearing two hats: no boundary between "what the operator configured" and "what any caller of the API can make the agent do or say."

**Fix**: extract the safety/conduct rules (no diagnosis, careful handling of health/distress disclosures, no pretending to be something it's not, whatever your compliance bar requires) into their own constant, and always prepend it — for both the default persona *and* any caller-supplied `prompt` — so it's a floor, not a feature of one branch.

## Architecture

**Single monolithic agent, despite the project's own guidance against that.** `AGENTS.md` (this repo's own contributor guide) says explicitly: *"it's important to design complex agents in a structured manner... You should make use of [handoffs and tasks] features, instead of writing long instruction prompts that cover multiple phases of a conversation."* `DefaultAgent`'s default persona is exactly that anti-pattern — one instructions block covering the opening, the exploratory-check-in phase, the health-concern-handling phase, and the closing/summary phase, all asked of a single `Agent` in one prompt with no state machine enforcing the transitions. Nothing structurally stops the model from summarizing at turn 2 or asking a "closing" question in the middle. For a scripted, multi-phase interaction like this (greet → check in → probe on concerns if any → wrap up), LiveKit's task/handoff primitives (which the project's own docs recommend) would give you actual phase boundaries instead of hoping the LLM infers "we're near the end now" from prompt text alone.

**Zero tools/function-calling.** `Agent.__init__` accepts a `tools=` list; `DefaultAgent` never passes one. This is a pure conversational agent with no way to *do* anything — it can't look up who it's calling, can't check a prior check-in's notes, can't log a symptom as structured data mid-call, can't page a caregiver, can't check a medication schedule. Everything it "knows" about the person is whatever free text `helper_prompt` contains, and everything it "reports" is a single free-text summary generated after the fact. For a product whose value is literally "notice if something's wrong with an elderly person and let their family know," having no mechanism to *act* on a concerning signal in real time — only a passive end-of-call paragraph — is the biggest behavioral gap here, bigger than any prompt wording issue.

**No structured output on the one thing that matters most.** `summarize_session()` asks a fresh LLM call to produce "2-4 sentences, factual and concise" — free text, no schema. For a wellness-check product, you want at minimum a structured result (e.g. `{summary, concern_level: none|watch|urgent, flagged_topics: [...]}`) that a backend can branch on programmatically, not a paragraph a human has to read to notice "grandma mentioned chest pain." Right now, an urgent disclosure and a pleasant chat about gardening produce the same shape of output, differing only in wording — nothing forces "urgent" to be detectable without an LLM (or a human) reading the prose.

**No call-length or idle guard, anywhere.** I grepped for it directly — the only `ctx.shutdown()` call in the whole file is on SIP failure. There's no max-duration timer, no idle-silence cutoff, nothing that ends a call on the agent's own initiative. `CALL_TIMEOUT_SECONDS` in `api.py` only bounds how long the *API* waits for a callback — it doesn't cap the phone call itself. A call runs exactly as long as the two parties keep talking (or until the far end hangs up), which is a real cost and product-experience risk with no backstop.

**Agent identity is duplicated, not shared.** `"Avery-ff5"` is hardcoded independently in `agent.py`'s `@server.rtc_session(agent_name=...)`, `api.py`'s `AGENT_NAME` constant, and `place_call.py`'s literal string. Fine today; a rename or multi-agent future breaks silently if only two of the three get updated.

**Two orchestration entry points, inconsistent capability.** `place_call.py` (manual CLI) can only ever trigger the hardcoded default persona — it doesn't send `prompt`, `helper_prompt`, or `language` at all. `api.py` (the real integration path) can override everything. That's a reasonable "quick manual test" vs. "real product" split in intent, but it's not documented as such, and it means the one persona with safety rules baked in (the default) is also the one nothing in production actually dispatches.

**Metadata parsing is fragile at the orchestration boundary.** `_parse_dial_info` catches `json.JSONDecodeError` but not the case where the metadata *is* valid JSON but not an object (e.g. a bare string or array) — `dial_info.get(...)` downstream would throw `AttributeError` uncaught. Low likelihood since `api.py` and `place_call.py` are the only current producers of that metadata and both always send a dict, but it's the trust boundary between an external dispatcher and the job, and it isn't defensive.

## Behavior / prompt design

What's genuinely good here, worth keeping:

- **Voice-realism instructions are well thought through** — sentence-length caps, one question at a time, filler words with `<break>` tags gated behind `_supports_ssml_breaks()` (correctly checks per-provider SSML support rather than assuming), mid-sentence self-corrections, rotating acknowledgments instead of repeating "oh that's nice." This is above-average attention to what actually makes a voice agent sound like a phone call instead of a chatbot read aloud.
- **The default persona's conversational rules are specific and well-calibrated** — "never ask questions like a form," "follow up on what they say," concrete example questions and example bad/good pairs. This is exactly the kind of few-shot-style grounding that makes LLM behavior more consistent than abstract instructions.
- **Turn-handling tuning is sophisticated**: dynamic endpointing (0.5–3.0s adapting to caller pause patterns), adaptive interruption detection (distinguishing barge-ins from "mm-hmm" backchannel), preemptive generation with preemptive TTS. This is real latency-engineering, not defaults left untouched.

What's missing or inconsistent:

- **Background ambience doesn't match the persona.** `BackgroundAudioPlayer` plays `BuiltinAudioClip.OFFICE_AMBIENCE`. The agent is introduced as "a warm, caring family companion... not a survey agent or customer service representative" — and then the caller hears office background noise the entire call. This reads as an unmodified template default rather than a deliberate choice, and it actively undercuts the persona it's paired with.
- **The default persona's safety rules are prose, not structure** — "if they mention a symptom... gently explore further," "do not diagnose," "encourage professional care if symptoms sound serious" are all instructions to the LLM's judgment, with nothing verifying the model actually follows them on a given call. There's no output check, no moderation pass, no post-hoc classifier — just trust in the prompt. Combined with the headline finding above, this whole safety layer is currently theoretical for real traffic anyway.
- **No language-specific behavior verification.** `language` is threaded through STT/LLM/TTS and into `_voice_realism_instructions` (which adds "speak only in {language}" for non-English), but nothing tests that the agent actually holds a coherent non-English conversation — the only behavioral tests that exist are English-only.
- **The closing behavior is aspirational, not enforced.** "At the end of the conversation: briefly summarize... end with warmth" is instruction text with no trigger. What tells the model "we're at the end" versus turn 3? Nothing structural — it's inferring this from conversational cues alone, same issue as the missing phase/task structure above.

## Orchestration

- **Prewarm is correctly scoped**: `silero.VAD.load()` runs once per worker process via `setup_fnc`, not per call — right call, avoids paying model-init latency on every job.
- **`llm.prewarm()` overlapping with the SIP dial-out** is a genuinely good latency trick — verified directly against the framework source that this is a real fire-and-forget task, not a dangling coroutine.
- **SIP failure handling distinguishes rejection from timeout/trunk failure** and reports back through the callback either way instead of leaving the API caller hanging until timeout — solid.
- **The one orchestration gap that matters most in practice**: nothing in the entrypoint enforces *who* is allowed to dispatch this agent with *what* instructions. Anyone able to call `agent_dispatch.create_dispatch` for `"Avery-ff5"` (or, once auth is added to `api.py`, anyone with a valid API key) can hand it literally any persona via `prompt`, and — per the headline finding — that persona runs with none of the conduct guardrails the default one has. The orchestration layer today has no concept of "trusted persona" vs. "arbitrary caller-supplied persona."

## What I'd prioritize, as an agent designer

1. **Make the safety/conduct rules a floor, not a persona feature** — always applied, regardless of whether `prompt` is caller-supplied. This is the single highest-leverage fix in this review.
2. **Give the end-of-call summary a schema**, not free text — at minimum a concern flag a backend can act on without an LLM-in-the-loop re-reading the paragraph.
3. **Split the monolithic persona into actual phases** using LiveKit's task/handoff pattern (which this project's own `AGENTS.md` already recommends) — greet → check-in → (conditional) concern-exploration → close — instead of one prompt asking the model to self-manage all four.
4. **Add a call-length ceiling** (idle timeout and/or hard max duration) so a call can't run indefinitely by construction.
5. **Fix or remove the background ambience** — it's actively working against the persona right now.
6. Once (1)–(4) exist, it's worth asking whether `DefaultAgent` should expose *tools* — e.g. a way to actually flag a concern in real time rather than only after the call ends — since that's what would let this stop being "a nice conversation that produces a paragraph" and start being "a check-in system that can actually alert someone."

## Verification notes

- Confirmed directly against the installed `livekit-agents` source (not assumed) that `Agent.__init__` stores `instructions` verbatim with no framework-level default/system prompt layered underneath it, and that `LLM.prewarm()` is genuinely synchronous/fire-and-forget by design.
- Grepped the full `src/` tree for `disconnect|max_duration|idle|timeout|hangup|shutdown|tools=` to confirm the absence of a call-length guard and of any tool/function registration — both came back empty except the one SIP-failure `ctx.shutdown()` call.
- Cross-checked the "safety rules only in the default branch" finding by re-reading `DefaultAgent.__init__` end to end and confirming `_voice_realism_instructions()` is the only content shared between the two branches.
- **Re-verified end to end** against the live repo: `git diff HEAD --stat` is empty and `HEAD` is the identical commit (`917b667...`) this audit was originally written against — `src/agent.py` (and every other tracked file) is unchanged, so every finding above still holds exactly as described. Nothing here is stale.
