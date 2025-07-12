from langgraph.graph import StateGraph
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage


class State(TypedDict):
    """State definition for the conversation workflow."""
    messages: List[BaseMessage]  
    next: str = None                 
    intent: str = None
    openai_api_key: Optional[str] = None  # Add API key field
