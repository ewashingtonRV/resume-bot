# Refactor: Replace LangGraph with a single full-context LLM call

## Context

This repo is a Streamlit chatbot ("Remy") that answers questions about Eric Washington's resume and professional experience. Response logic currently lives in `src/` as a LangGraph pipeline:

- `src/graph.py` — StateGraph with two parallel classifier nodes feeding a QA node
- `src/nodes.py` — `IntentClassificationNode` and `GitHubStatsIntentClassificationNode` (each a gpt-4o call per user message), plus `QuestionAnsweringNode` which builds a `create_react_agent` per request
- `src/prompts.py` — intent categories, classifier prompts, and a system prompt template that loads reference text from `./data/projects/` and duplicates the Remy persona inline
- `src/state.py`, `src/tools.py` (GitHub stats tools), `src/utils.py`, `src/llm_base.py`
- `apps/streamlit_app.py` — Streamlit UI that invokes the compiled graph; `apps/fastapi_app.py` — FastAPI wrapper

Problems with the current design: 3+ LLM calls per turn, brittle intent routing (raises `ValueError` on unrecognized intent), grounding text is read from the stale `./data/` directory instead of the vault, and the persona file `.claude/soul.md` is never loaded.

## Goal

Delete the LangGraph pipeline and replace it with **one LLM call per user turn**, grounded in the full vault:

1. **System prompt = persona + entire wiki.** At app startup, build the system prompt by concatenating:
   - `.claude/soul.md` (the Remy persona — load from disk, do not copy its text into code)
   - All `*.md` files under `vault/`, **excluding `vault/sources/`** (immutable raw sources — never sent to the model) and excluding `vault/log.md`. Include `vault/index.md` first, then the remaining pages grouped by directory (`projects/`, `skills/`, `tools/`, `outcomes/`, `qa/`, plus `eric-washington.md`). Prefix each file's content with a header identifying its vault path so the model can attribute facts to pages.
   - The full wiki is ~85KB (~21K tokens), so this comfortably fits in context.
2. **No intent classification, no routing, no react agent.** The GitHub stats tools in `src/tools.py` (`get_user_stats`, `get_repo_stats`) should be kept and exposed as native tool/function definitions on the single chat completion call. The model decides when to call them; implement a plain tool-call loop (call model → if tool_calls, execute and append results → call model again → return final text). No LangChain `Tool`/`StructuredTool` wrappers.
3. **Keep OpenAI** as the provider (existing dependency and API-key flow), using the official `openai` SDK directly. Preserve the current behavior where an API key can be supplied at runtime (session/state) with fallback to the `OPENAI_API_KEY` env var. Use one configurable model name (default `gpt-4o-mini`) read from env/config. Structure the prompt so the large static portion (persona + vault) is the constant prefix, so provider-side prompt caching applies.
4. **Conversation memory** stays in Streamlit `st.session_state.messages` (list of role/content dicts). Drop LangGraph's `MemorySaver` and thread/checkpointer config. Pass the full message history plus the system prompt on each call.

## Implementation requirements

- New module (suggest `src/chat.py` or similar) exposing a small interface, e.g. `build_system_prompt() -> str` and `respond(messages: list[dict], api_key: str | None) -> str`. Keep context assembly (`build_system_prompt`) as a separate, swappable function — a later refactor may replace full-context stuffing with agentic file navigation.
- Cache the assembled system prompt in memory at startup (e.g. `@lru_cache` or module-level); add a way to rebuild it without restarting if cheap to do.
- Update `apps/streamlit_app.py` to call the new interface. Preserve the existing UI behavior: chat history rendering, the streaming-style `response_generator`, and session initialization. Actual token streaming from the API is a nice-to-have, not required.
- Update `apps/fastapi_app.py` to the same interface.
- Delete or quarantine now-dead code: `src/graph.py`, `src/nodes.py`, `src/state.py`, `src/prompts.py` (persona/classifier prompts), and the `./data/projects/` reference-text loading path. Keep `src/tools.py` (GitHub stats) — refactor it off LangChain if it imports any.
- Remove now-unused dependencies from `requirements.txt` and `pyproject.toml`: all `langgraph*`, `langchain*` packages (keep `langsmith` only if still used; otherwise remove).
- Error handling: if the OpenAI call fails, return a graceful in-persona message rather than raising; if a GitHub tool call fails, tell the model via the tool result so it can respond gracefully.
- Guardrail: `vault/sources/` and `vault/log.md` must never be included in the prompt. Add a unit test asserting this.

## Acceptance criteria

- One model call per turn for non-tool questions; at most one tool round-trip for GitHub stats questions.
- Remy answers project questions ("Tell me about RaaS", "What did smart links achieve?") using vault content, and persona/tone comes from `.claude/soul.md`.
- Cross-project questions ("Compare Eric's search work to his recommendation work") work — this was impossible under the old single-category routing.
- GitHub stats questions still trigger the tools and report the lookback period.
- Unknown-topic questions produce the in-persona knowledge-gap response defined in soul.md (no hallucinated experience).
- Streamlit app runs locally (`streamlit run apps/streamlit_app.py`) with no LangGraph/LangChain imports anywhere.
- Existing tests updated or replaced; new tests for `build_system_prompt` (includes index.md, excludes sources/ and log.md) and for the tool-call loop (mocked OpenAI client).

## Non-goals

- No embedding/vector retrieval, no chunking.
- No changes to vault content, structure, or the ingestion pipeline.
- No changes to Docker/deploy beyond dependency cleanup.
