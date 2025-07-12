# Resume Bot

A conversational AI chatbot that can answer questions about my professional experience and skills. Built with LangChain, FastAPI, and OpenAI.

## Features

- 🤖 Interactive chat interface for resume-based Q&A
- 🔄 Context-aware conversations with memory
- 🎯 Intent classification for targeted responses
- 📊 Evaluation framework for testing and improving responses
- 🚀 FastAPI backend for scalable deployment
- 🌊 Streamlit frontend for easy interaction

## Project Structure

```
resume-bot/
├── data/               # Resume and training data
├── evals/             # Evaluation framework
│   ├── scripts/       # Evaluation scripts
│   └── src/          # Evaluation source code
├── src/               # Core bot logic
│   ├── graph.py      # Conversation flow graph
│   ├── nodes.py      # Processing nodes
│   ├── prompts.py    # System prompts
│   ├── state.py      # State management
│   ├── tools.py      # Utility tools
│   └── utils.py      # Helper functions
├── fastapi_app.py     # FastAPI backend
└── streamlit_app.py   # Streamlit frontend
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-bot.git
cd resume-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and other configurations
```

## Running the Application

### Backend Server
```bash
uvicorn fastapi_app:app --reload
```

### Frontend Interface
```bash
streamlit run streamlit_app.py
```

## Development

### Running Tests
```bash
python -m pytest test_fastapi.py
```

### Evaluation
The `evals/` directory is based largely on this [Hamel course](https://maven.com/parlance-labs/evals) and contains scripts for:
- Generating synthetic conversations
- Creating the golden dataset
- Evaluating response quality using Auto Annotators
- Iterating over different prompts and configurations

Run evaluations:
```bash
cd evals/scripts
python evaluate.py
```

## Architecture

The bot uses a graph-based architecture with:
1. Intent Classification Node - Determines the type of question
2. Question Answering Node - Generates contextual responses
3. State Management - Maintains conversation context

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

Eric Washington - washe97@gmail.com
[LinkedIn Profile](www.linkedin.com/in/eric-washington-111a935a)
