"""Single-call chat interface for Remy.

Builds one system prompt (persona + full vault wiki) and answers each user
turn with a single Claude (Anthropic) message, plus an optional tool
round-trip for GitHub stats questions.
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from anthropic import Anthropic

from src.stats_provider import CachedGitHubStats

REPO_ROOT = Path(__file__).parent.parent
VAULT_DIR = REPO_ROOT / "vault"
SOUL_PATH = REPO_ROOT / ".claude" / "soul.md"

# Order in which vault directories are appended after index.md and the
# top-level eric-washington.md page.
VAULT_DIR_ORDER = ["projects", "skills", "tools", "outcomes", "qa"]

# Never sent to the model: immutable raw sources and the append-only log.
EXCLUDED_DIRS = {"sources"}
EXCLUDED_FILES = {"log.md"}

DEFAULT_MODEL = "claude-opus-4-8"
MAX_TOKENS = 16000
MAX_TOOL_ITERATIONS = 3

FALLBACK_MESSAGE = (
    "Hm, I hit a snag reaching my brain (the model API) just now. "
    "Give it another shot in a moment — or reach out to Eric Washington directly."
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _vault_page(path: Path) -> str:
    """Format one vault page with a header identifying its vault path."""
    rel = path.relative_to(REPO_ROOT)
    return f"===== {rel} =====\n{_read(path)}"


def list_vault_files() -> list[Path]:
    """Return the vault pages included in the prompt, in prompt order.

    Excludes vault/sources/ (immutable raw sources) and vault/log.md by
    construction — these must never be sent to the model.
    """
    files = [VAULT_DIR / "index.md", VAULT_DIR / "eric-washington.md"]
    for dir_name in VAULT_DIR_ORDER:
        files.extend(sorted((VAULT_DIR / dir_name).glob("*.md")))
    return [
        f for f in files
        if f.is_file()
        and f.name not in EXCLUDED_FILES
        and EXCLUDED_DIRS.isdisjoint(f.relative_to(VAULT_DIR).parts[:-1])
    ]


def build_system_prompt() -> str:
    """Assemble the full system prompt: persona (soul.md) + entire vault wiki.

    Deterministic output (stable file order) so the prompt is a constant
    prefix and provider-side prompt caching applies. Swappable: replace this
    function to move from full-context stuffing to retrieval later.
    """
    sections = [_read(SOUL_PATH)]
    sections.append(
        "\n---\n\nBelow is Eric Washington's full knowledge base (the vault wiki). "
        "Each page is preceded by a header with its vault path so you can "
        "attribute facts to pages."
    )
    sections.extend(_vault_page(path) for path in list_vault_files())
    return "\n\n".join(sections)


@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    """Cached system prompt. Call rebuild_system_prompt() to refresh."""
    return build_system_prompt()


def rebuild_system_prompt() -> str:
    """Rebuild the cached system prompt (e.g. after vault edits)."""
    get_system_prompt.cache_clear()
    return get_system_prompt()


GITHUB_CATEGORIES = [
    "automango", "raas", "medical taxonomy",
    "recommendation models", "smart links", "article tagging",
]

# Precomputed snapshot windows — must match the --windows used when running
# scripts/collect_github_stats.py.
GITHUB_LOOKBACK_WINDOWS = [7, 30, 90, 365]

TOOL_DEFINITIONS = [
    {
        "name": "github_user_stats",
        "description": (
            "Retrieves Eric Washington's overall GitHub statistics (commits, "
            "pull requests) across his tracked project repos for a time period. "
            "Call this when the user asks about Eric's overall coding or "
            "GitHub activity. Stats come from a periodically refreshed "
            "snapshot; the result includes an as_of_date. Always tell the "
            "user the lookback period and the as-of date the stats cover."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lookback_days": {
                    "type": "integer",
                    "enum": GITHUB_LOOKBACK_WINDOWS,
                    "description": "Days to look back. Only these precomputed windows exist. Defaults to 365.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "github_repo_stats",
        "description": (
            "Retrieves Eric Washington's GitHub statistics (commits, pull "
            "requests, code changes) for the repositories of a specific "
            "project category. Call this when the user asks about Eric's "
            "coding activity on a specific project. Stats come from a "
            "periodically refreshed snapshot; the result includes an "
            "as_of_date. Always tell the user the lookback period and the "
            "as-of date the stats cover."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intent_category_name": {
                    "type": "string",
                    "description": "Project category to get repo stats for.",
                    "enum": GITHUB_CATEGORIES,
                },
                "lookback_days": {
                    "type": "integer",
                    "enum": GITHUB_LOOKBACK_WINDOWS,
                    "description": "Days to look back. Only these precomputed windows exist. Defaults to 365.",
                },
            },
            "required": ["intent_category_name"],
        },
    },
]


def _execute_tool(name: str, arguments: dict) -> tuple[str, bool]:
    """Run one GitHub stats tool; return (result_text, is_error).

    Failures are reported as text (not raised) so the model can respond
    gracefully.
    """
    try:
        stats = CachedGitHubStats()  # lazy: raises here if GITHUB_STATS_BUCKET is missing
        if name == "github_user_stats":
            result = stats.get_user_stats(**arguments)
        elif name == "github_repo_stats":
            result = stats.get_repo_stats(**arguments)
        else:
            return f"Error: unknown tool '{name}'.", True
        return json.dumps(result), False
    except Exception as e:
        logging.error(f"Tool {name} failed: {e}")
        return (
            f"Error: the {name} tool failed ({e}). "
            "Let the user know the stats are unavailable right now.",
            True,
        )


def _extract_text(response) -> str:
    return "\n\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def respond(messages: list[dict], api_key: str | None = None) -> str:
    """Answer the conversation with one model call plus an optional tool loop.

    Args:
        messages: full chat history as {"role", "content"} dicts.
        api_key: runtime Anthropic key; falls back to ANTHROPIC_API_KEY env var.
    """
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
    try:
        client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        # Static system prompt with a cache breakpoint so Anthropic's prompt
        # caching serves the persona + vault prefix at ~0.1x cost.
        system = [{
            "type": "text",
            "text": get_system_prompt(),
            "cache_control": {"type": "ephemeral"},
        }]
        request_messages = list(messages)

        for iteration in range(MAX_TOOL_ITERATIONS + 1):
            # Last pass: forbid tools so the model must produce a text answer.
            tool_choice = (
                {"type": "none"} if iteration == MAX_TOOL_ITERATIONS else {"type": "auto"}
            )
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=system,
                messages=request_messages,
                tools=TOOL_DEFINITIONS,
                tool_choice=tool_choice,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                return _extract_text(response) or FALLBACK_MESSAGE

            # Echo the assistant turn (thinking + tool_use blocks) unchanged,
            # then answer every tool call in a single user message.
            request_messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tool_use in tool_uses:
                content, is_error = _execute_tool(tool_use.name, dict(tool_use.input))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": content,
                    "is_error": is_error,
                })
            request_messages.append({"role": "user", "content": tool_results})

        return FALLBACK_MESSAGE
    except Exception as e:
        logging.error(f"Anthropic call failed: {e}")
        return FALLBACK_MESSAGE
