import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage

from apps.fastapi_app import app

tc = TestClient(app)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("IS_LOCAL_TESTING", "True")


@pytest.fixture
def mock_graph():
    with patch("apps.fastapi_app.graph") as m:
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

def test_chat_returns_response(mock_graph):
    mock_graph.invoke.return_value = {
        "messages": [
            HumanMessage(content="Tell me about yourself"),
            AIMessage(content="I'm a resume bot!"),
        ]
    }

    response = tc.post("/chat", json={
        "messages": [{"role": "user", "content": "Tell me about yourself"}]
    })

    assert response.status_code == 200
    body = response.json()
    assert len(body["messages"]) == 2
    assert body["messages"][-1]["role"] == "assistant"
    assert body["messages"][-1]["content"] == "I'm a resume bot!"
    assert "thread_id" in body


def test_chat_uses_provided_thread_id(mock_graph):
    mock_graph.invoke.return_value = {
        "messages": [AIMessage(content="Hello!")]
    }

    thread_id = "my-thread-123"
    response = tc.post("/chat", json={
        "messages": [{"role": "user", "content": "Hi"}],
        "thread_id": thread_id,
    })

    assert response.status_code == 200
    assert response.json()["thread_id"] == thread_id

    call_config = mock_graph.invoke.call_args.kwargs["config"]
    assert call_config["configurable"]["thread_id"] == thread_id


def test_chat_generates_thread_id_when_absent(mock_graph):
    mock_graph.invoke.return_value = {"messages": [AIMessage(content="Hi")]}

    response = tc.post("/chat", json={
        "messages": [{"role": "user", "content": "Hello"}]
    })

    assert response.status_code == 200
    thread_id = response.json()["thread_id"]
    assert len(thread_id) == 36  # UUID format


def test_chat_passes_is_local_flag(mock_graph):
    mock_graph.invoke.return_value = {"messages": [AIMessage(content="ok")]}

    tc.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    call_config = mock_graph.invoke.call_args.kwargs["config"]
    assert call_config["metadata"]["is_local_testing"] is True
    assert "local" in call_config["tags"]


def test_chat_multi_turn_messages(mock_graph):
    mock_graph.invoke.return_value = {
        "messages": [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello!"),
            HumanMessage(content="What can you do?"),
            AIMessage(content="I can answer resume questions."),
        ]
    }

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


# --- /chat error handling ---

def test_chat_returns_500_on_graph_error(mock_graph):
    mock_graph.invoke.side_effect = RuntimeError("graph exploded")

    response = tc.post("/chat", json={
        "messages": [{"role": "user", "content": "Hi"}]
    })

    assert response.status_code == 500
    assert "graph exploded" in response.json()["detail"]


def test_chat_rejects_empty_messages(mock_graph):
    mock_graph.invoke.return_value = {"messages": []}

    response = tc.post("/chat", json={"messages": []})

    assert response.status_code == 200
    assert response.json()["messages"] == []


def test_chat_invalid_payload():
    response = tc.post("/chat", json={"wrong_field": "value"})
    assert response.status_code == 422
