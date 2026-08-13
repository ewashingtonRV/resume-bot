from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import uuid
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.chat import respond

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.getenv("ANTHROPIC_API_KEY"):
    logger.info("✅ ANTHROPIC_API_KEY loaded successfully")
else:
    logger.error("❌ ANTHROPIC_API_KEY not found in environment variables!")

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
        # Thread ID is kept for API compatibility; conversation state is
        # carried entirely by the messages the client sends.
        thread_id = request.thread_id or str(uuid.uuid4())

        messages = [msg.model_dump() for msg in request.messages]
        if messages:
            answer = respond(messages)
            messages.append({"role": "assistant", "content": answer})

        response_messages = [Message(**msg) for msg in messages]
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
