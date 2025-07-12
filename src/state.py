from typing import TypedDict, List, Optional, Annotated
from typing_extensions import Required
from langchain_core.messages import BaseMessage


class State(TypedDict, total=False):
    """State definition for the conversation workflow."""
    # Required fields
    messages: Required[List[BaseMessage]]  # Messages must always be present
    
    # Optional fields that can be updated in parallel
    intent: Annotated[Optional[str], "intent"]  # Intent from main classifier
    is_github_stats: Annotated[Optional[bool], "is_github_stats"]  # GitHub stats flag
    
    # Other optional fields
    next: Optional[str]
    openai_api_key: Optional[str]
    answer: Optional[str]
