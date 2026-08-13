import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from apps.fastapi_app import app

tc = TestClient(app)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")


@pytest.fixture
def mock_respond():
    with patch("apps.fastapi_app.respond") as m:
        yield m


# --- Root and health ---

def test_root():
    response = tc.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Resume Bot API"
    assert "chat" in body["endpoints"]


def test_health():
    response = tc.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# --- /chat happy paths ---

def test_chat_returns_response(mock_respond):
    mock_respond.return_value = "I'm a resume bot!"

    response = tc.post("/chat", json={
        "messages": [{"role": "user", "content": "Tell me about yourself"}]
    })

    assert response.status_code == 200
    body = response.json()
    assert len(body["messages"]) == 2
    assert body["messages"][-1]["role"] == "assistant"
    assert body["messages"][-1]["content"] == "I'm a resume bot!"
    assert "thread_id" in body


def test_chat_uses_provided_thread_id(mock_respond):
    mock_respond.return_value = "Hello!"

    thread_id = "my-thread-123"
    response = tc.post("/chat", json={
        "messages": [{"role": "user", "content": "Hi"}],
        "thread_id": thread_id,
    })

    assert response.status_code == 200
    assert response.json()["thread_id"] == thread_id


def test_chat_generates_thread_id_when_absent(mock_respond):
    mock_respond.return_value = "Hi"

    response = tc.post("/chat", json={
        "messages": [{"role": "user", "content": "Hello"}]
    })

    assert response.status_code == 200
    thread_id = response.json()["thread_id"]
    assert len(thread_id) == 36  # UUID format


def test_chat_multi_turn_messages(mock_respond):
    mock_respond.return_value = "I can answer resume questions."

    response = tc.post("/chat", json={
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What can you do?"},
        ]
    })

    assert response.status_code == 200
    messages = response.json()["messages"]
    assert len(messages) == 4
    assert messages[-1]["content"] == "I can answer resume questions."
    # Full history is passed to respond
    history = mock_respond.call_args.args[0]
    assert [m["content"] for m in history[:3]] == ["Hi", "Hello!", "What can you do?"]


# --- /chat error handling ---

def test_chat_returns_500_on_error(mock_respond):
    mock_respond.side_effect = RuntimeError("model exploded")

    response = tc.post("/chat", json={
        "messages": [{"role": "user", "content": "Hi"}]
    })

    assert response.status_code == 500
    assert "model exploded" in response.json()["detail"]


def test_chat_empty_messages_skips_model(mock_respond):
    response = tc.post("/chat", json={"messages": []})

    assert response.status_code == 200
    assert response.json()["messages"] == []
    mock_respond.assert_not_called()


def test_chat_invalid_payload():
    response = tc.post("/chat", json={"wrong_field": "value"})
    assert response.status_code == 422
