# Resume Bot

A conversational AI chatbot ("Remy") that answers questions about my professional experience and skills. Built with the Anthropic API (Claude), FastAPI, and Streamlit.

## Features

- 🤖 Interactive chat interface for resume-based Q&A
- 📚 Grounded in a full knowledge-base wiki (`vault/`) loaded into the model's context
- 🛠️ Native tool calling for live GitHub contribution stats
- 💾 Prompt caching so the large static context is cheap after the first request
- 📊 Evaluation framework for testing and improving responses
- 🚀 FastAPI backend for scalable deployment
- 🌊 Streamlit frontend for easy interaction

## Project Structure

```
resume-bot/
├── vault/              # Knowledge-base wiki (projects, skills, tools, outcomes, qa)
├── data/               # Resume markdown (download button)
├── evals/              # Evaluation framework
│   ├── scripts/        # Evaluation scripts
│   └── src/            # Evaluation source code
├── apps/               # Application entry points
│   ├── fastapi_app.py  # FastAPI backend
│   └── streamlit_app.py# Streamlit frontend
├── src/                # Core bot logic
│   ├── chat.py         # System-prompt assembly + Claude tool-call loop
│   ├── tools.py        # GitHub stats tools
│   └── utils.py        # Helper functions
└── tests/              # Test suite
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-bot.git
cd resume-bot
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up environment variables in a `.env` file at the repo root:
```bash
ANTHROPIC_API_KEY=sk-ant-...   # required — powers all responses
GITHUB_TOKEN=ghp_...           # optional — enables the GitHub stats tools
ANTHROPIC_MODEL=claude-opus-4-8  # optional — defaults to claude-opus-4-8
```

## Running the Application Locally

### Streamlit app (chat UI)

```bash
uv run streamlit run apps/streamlit_app.py
```

Then open http://localhost:8501 in your browser. The full system prompt (persona + vault wiki) is assembled once at startup; each chat turn makes a single Claude call, plus one tool round-trip for GitHub stats questions.

### FastAPI backend

```bash
uv run uvicorn apps.fastapi_app:app --reload
```

The API serves on http://localhost:8000 (interactive docs at http://localhost:8000/docs).

#### Sample request

The API is stateless — send the full conversation history each time. The response echoes the history with the assistant's reply appended:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is Relevance as a Service and how does it work?"}
    ]
  }'
```

Sample response:

```json
{
  "messages": [
    {"role": "user", "content": "What is Relevance as a Service and how does it work?"},
    {"role": "assistant", "content": "RaaS is Eric's recommendation platform built on Vespa..."}
  ],
  "thread_id": "0d5e9a3c-8f27-4b1e-9c64-2a7d1f0b5e88"
}
```

For a multi-turn conversation, send the returned `messages` array back with your next user message appended. `thread_id` is optional metadata — pass your own or let the API generate one.

Health check:

```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

## Development

### Running Tests
```bash
uv run pytest tests/
```

### Evaluation
The `evals/` directory is based largely on this [Hamel course](https://maven.com/parlance-labs/evals) and contains scripts for:
- Generating synthetic conversations
- Creating the golden dataset
- Evaluating response quality using Auto Annotators
- Iterating over different prompts and configurations

The eval scripts have extra dependencies: `uv pip install -r evals/requirements.txt`

Run evaluations:
```bash
cd evals/scripts
python evaluate.py
```

## Architecture

Each user turn is answered by a single Claude (`claude-opus-4-8`) call:

1. **System prompt** — the Remy persona (`.claude/soul.md`) plus the entire `vault/` wiki (excluding `vault/sources/` and `vault/log.md`), assembled at startup and cached with Anthropic prompt caching.
2. **Tool-call loop** — the GitHub stats tools in `src/tools.py` are exposed as native tool definitions; the model decides when to call them and results are fed back in the same turn.
3. **Conversation memory** — chat history lives in the client (Streamlit session state, or the `messages` array API clients send), not on the server.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

Eric Washington - washe97@gmail.com
[LinkedIn Profile](www.linkedin.com/in/eric-washington-111a935a)
