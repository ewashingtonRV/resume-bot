# Resume Bot

A conversational AI chatbot ("Remy") that answers questions about my professional experience and skills. Built with the Anthropic API (Claude), FastAPI, and Streamlit.

## Features

- 🤖 Interactive chat interface for resume-based Q&A
- 📚 Grounded in a full knowledge-base wiki (`vault/`) loaded into the model's context
- 🛠️ Native tool calling for GitHub contribution stats, served from a periodically refreshed S3 snapshot (no GitHub token in the deployed app)
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
├── scripts/            # Local-only: collect GitHub stats + upload snapshot to S3
├── src/                # Core bot logic
│   ├── chat.py         # System-prompt assembly + Claude tool-call loop
│   ├── stats_provider.py # Serves GitHub stats from the S3 snapshot
│   ├── tools.py        # Live GitHub API client (used only by scripts/)
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
ANTHROPIC_API_KEY=sk-ant-...      # required — powers all responses
ANTHROPIC_MODEL=claude-opus-4-8   # optional — defaults to claude-opus-4-8
GITHUB_STATS_BUCKET=my-bucket     # optional — enables the GitHub stats tools (S3 snapshot)
GITHUB_STATS_PREFIX=github-stats/ # optional — S3 key prefix, defaults to github-stats/
GITHUB_TOKEN=ghp_...              # local only — used by scripts/collect_github_stats.py, never deployed
```

AWS credentials for reading the snapshot come from the default boto3 chain (env vars, `~/.aws`, or an IAM role when deployed on AWS).

## GitHub Stats Snapshot (S3)

The deployed app never holds a GitHub token. Stats are precomputed locally
for fixed lookback windows (7, 30, 90, 365 days), uploaded to a private S3
bucket, and served from there. Each answer cites the snapshot's as-of date.

### One-time AWS setup

```bash
aws s3 mb s3://<your-bucket>   # keep it private (default)
```

Minimal IAM policy for the **deployed app** (read-only):

```json
{"Version": "2012-10-17", "Statement": [{
  "Effect": "Allow",
  "Action": "s3:GetObject",
  "Resource": "arn:aws:s3:::<your-bucket>/github-stats/*"
}]}
```

The identity you upload with locally additionally needs `s3:PutObject` on the
same resource.

### Refreshing the snapshot

Run locally whenever you want fresher numbers (requires `GITHUB_TOKEN` and a
read-scoped fine-grained PAT is enough — the scripts only read commits/PRs):

```bash
uv run python scripts/collect_github_stats.py                  # writes data/github_stats/*.csv (gitignored)
uv run python scripts/upload_github_stats.py --bucket <bucket> # uploads to s3://<bucket>/github-stats/
```

The app caches the snapshot in-process — restart or redeploy to pick up a
refresh. The lookback windows written by the collect script must match
`GITHUB_LOOKBACK_WINDOWS` in `src/chat.py`.

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
2. **Tool-call loop** — the GitHub stats tools are exposed as native tool definitions; the model decides when to call them and results are fed back in the same turn. Results come from the S3 snapshot via `src/stats_provider.py` (see "GitHub Stats Snapshot" above); `src/tools.py` talks to the live GitHub API and is used only by the local collect script.
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
