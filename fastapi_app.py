from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging
import uuid
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the pre-configured graph
from src.graph import graph
from src.state import State

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify API key is loaded
openai_api_key = os.getenv("OPENAI_API_KEY")
logger.info(f"OPENAI_API_KEY: {openai_api_key}")

if openai_api_key:
    logger.info(f"✅ OPENAI_API_KEY loaded successfully (starts with: {openai_api_key[:7]}...)")
else:
    logger.error("❌ OPENAI_API_KEY not found in environment variables!")

app = FastAPI(title="Resume Bot API", description="API for resume bot conversation")

# Pydantic models for request/response
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    messages: List[Message]
    thread_id: str

def convert_to_langchain_messages(messages: List[Message]) -> List[BaseMessage]:
    """Convert Pydantic messages to LangChain messages."""
    langchain_messages = []
    for msg in messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
        else:
            logger.warning(f"Unknown message role: {msg.role}")
    return langchain_messages

def convert_from_langchain_messages(messages: List[BaseMessage]) -> List[Message]:
    """Convert LangChain messages to Pydantic messages."""
    pydantic_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            pydantic_messages.append(Message(role="user", content=msg.content))
        elif isinstance(msg, AIMessage):
            pydantic_messages.append(Message(role="assistant", content=msg.content))
        else:
            logger.warning(f"Unknown message type: {type(msg)}")
    return pydantic_messages

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a conversation with the resume bot.
    
    Args:
        request: ChatRequest containing messages and optional thread_id
        
    Returns:
        ChatResponse with updated messages and thread_id
    """
    try:
        # Generate thread_id if not provided
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Convert Pydantic messages to LangChain messages
        langchain_messages = convert_to_langchain_messages(request.messages)
        
        # Detect environment - simple check for localhost/local development
        is_local = os.getenv("IS_LOCAL_TESTING") == "True"
        
        # Create state for the graph
        state = {
            "messages": langchain_messages,
            "next": "intent_classification",
            "intent": None,
            "openai_api_key": openai_api_key  # Always use environment variable
        }        
        # Create config with thread_id for checkpointer
        config = {
            "configurable": {
                "thread_id": thread_id
            },
            "metadata": {
                "environment": "local" if is_local else "production",
                "is_local_testing": is_local
            },
            "tags": ["resume-bot", "fastapi", "local" if is_local else "production"]
        }
        
        # Invoke the graph
        result = graph.invoke(state, config=config)
        
        # Convert result messages back to Pydantic format
        response_messages = convert_from_langchain_messages(result["messages"])
        
        return ChatResponse(messages=response_messages, thread_id=thread_id)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Resume Bot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 