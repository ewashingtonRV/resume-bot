import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src import chat
from src.chat import (
    FALLBACK_MESSAGE,
    build_system_prompt,
    list_vault_files,
    respond,
)


# --- build_system_prompt ---

def test_includes_soul_persona():
    prompt = build_system_prompt()
    soul_text = chat.SOUL_PATH.read_text(encoding="utf-8")
    assert soul_text in prompt
    # Persona loads first
    assert prompt.startswith(soul_text.split("\n")[0])


def test_index_first_then_project_pages():
    prompt = build_system_prompt()
    index_pos = prompt.index("===== vault/index.md =====")
    project_pos = prompt.index("===== vault/projects/")
    assert index_pos < project_pos


def test_includes_known_project_page():
    prompt = build_system_prompt()
    assert "===== vault/projects/raas.md =====" in prompt
    raas_content = (chat.VAULT_DIR / "projects" / "raas.md").read_text(encoding="utf-8")
    assert raas_content in prompt


def test_includes_all_wiki_directories():
    prompt = build_system_prompt()
    for dir_name in ["projects", "skills", "tools", "outcomes", "qa"]:
        assert f"===== vault/{dir_name}/" in prompt


# --- Guardrail: sources/ and log.md must never reach the model ---

def test_excludes_sources_and_log_from_file_list():
    for path in list_vault_files():
        rel_parts = path.relative_to(chat.VAULT_DIR).parts
        assert "sources" not in rel_parts[:-1]
        assert path.name != "log.md"


def test_excludes_sources_and_log_from_prompt():
    prompt = build_system_prompt()
    assert "===== vault/sources" not in prompt
    assert "===== vault/log.md =====" not in prompt


# --- respond: tool-use loop with mocked Anthropic client ---

def _text_block(text):
    return SimpleNamespace(type="text", text=text)

def _tool_use_block(block_id, name, arguments):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=arguments)

def _response(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


@pytest.fixture
def mock_anthropic():
    with patch("src.chat.Anthropic") as mock_cls:
        yield mock_cls.return_value


@pytest.fixture
def mock_github():
    with patch("src.chat.CachedGitHubStats") as mock_cls:
        yield mock_cls


def test_plain_answer_single_call(mock_anthropic):
    mock_anthropic.messages.create.return_value = _response(
        [_text_block("Eric built RaaS.")]
    )

    answer = respond([{"role": "user", "content": "Tell me about RaaS"}])

    assert answer == "Eric built RaaS."
    assert mock_anthropic.messages.create.call_count == 1
    kwargs = mock_anthropic.messages.create.call_args.kwargs
    # Persona + vault is the cached system prompt; history is untouched
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["system"][0]["text"] == chat.get_system_prompt()
    assert kwargs["messages"] == [{"role": "user", "content": "Tell me about RaaS"}]
    assert kwargs["model"] == chat.DEFAULT_MODEL


def test_tool_use_round_trip(mock_anthropic, mock_github):
    tool_use = _tool_use_block("toolu_1", "github_user_stats", {"lookback_days": 30})
    mock_anthropic.messages.create.side_effect = [
        _response([tool_use], stop_reason="tool_use"),
        _response([_text_block("Eric made 12 commits in the last 30 days.")]),
    ]
    user_stats = {
        "as_of_date": "2026-08-22", "lookback_days": 30,
        "total_commits": 12, "total_pull_requests": 5,
    }
    mock_github.return_value.get_user_stats.return_value = user_stats

    answer = respond([{"role": "user", "content": "How active is Eric on GitHub?"}])

    assert answer == "Eric made 12 commits in the last 30 days."
    mock_github.return_value.get_user_stats.assert_called_once_with(lookback_days=30)
    # Second call includes the tool result in a single user message
    second_messages = mock_anthropic.messages.create.call_args_list[1].kwargs["messages"]
    assert second_messages[1] == {"role": "assistant", "content": [tool_use]}
    tool_results = second_messages[2]["content"]
    assert second_messages[2]["role"] == "user"
    assert tool_results[0]["type"] == "tool_result"
    assert tool_results[0]["tool_use_id"] == "toolu_1"
    assert tool_results[0]["is_error"] is False
    assert json.loads(tool_results[0]["content"]) == user_stats


def test_repo_stats_tool_dispatch(mock_anthropic, mock_github):
    tool_use = _tool_use_block(
        "toolu_2", "github_repo_stats", {"intent_category_name": "raas"}
    )
    mock_anthropic.messages.create.side_effect = [
        _response([tool_use], stop_reason="tool_use"),
        _response([_text_block("done")]),
    ]
    mock_github.return_value.get_repo_stats.return_value = []

    respond([{"role": "user", "content": "GitHub activity on raas?"}])

    mock_github.return_value.get_repo_stats.assert_called_once_with(intent_category_name="raas")


def test_api_failure_returns_graceful_message(mock_anthropic):
    mock_anthropic.messages.create.side_effect = RuntimeError("api down")

    answer = respond([{"role": "user", "content": "hi"}])

    assert answer == FALLBACK_MESSAGE


def test_tool_failure_reported_to_model(mock_anthropic, mock_github):
    tool_use = _tool_use_block("toolu_3", "github_user_stats", {})
    mock_anthropic.messages.create.side_effect = [
        _response([tool_use], stop_reason="tool_use"),
        _response([_text_block("Stats are unavailable right now.")]),
    ]
    mock_github.side_effect = ValueError("GITHUB_STATS_BUCKET not set")

    answer = respond([{"role": "user", "content": "GitHub stats?"}])

    assert answer == "Stats are unavailable right now."
    second_messages = mock_anthropic.messages.create.call_args_list[1].kwargs["messages"]
    tool_result = second_messages[2]["content"][0]
    assert tool_result["is_error"] is True
    assert "Error" in tool_result["content"]


def test_tool_loop_cap_forces_text_answer(mock_anthropic, mock_github):
    tool_use = _tool_use_block("toolu_4", "github_user_stats", {})
    mock_anthropic.messages.create.side_effect = [
        _response([tool_use], stop_reason="tool_use"),
        _response([tool_use], stop_reason="tool_use"),
        _response([tool_use], stop_reason="tool_use"),
        _response([_text_block("Here's what I found.")]),
    ]
    mock_github.return_value.get_user_stats.return_value = {}

    answer = respond([{"role": "user", "content": "GitHub stats?"}])

    assert answer == "Here's what I found."
    # The final call forbids tools so the model must answer in text
    final_kwargs = mock_anthropic.messages.create.call_args_list[-1].kwargs
    assert final_kwargs["tool_choice"] == {"type": "none"}
