from langgraph.graph import StateGraph, START, END
import logging
import os
from src.state import State
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import Tool
from langchain.tools import StructuredTool
from src.nodes import IntentClassificationNode, QuestionAnsweringNode, GitHubStatsIntentClassificationNode
from src.tools import GitHubStats
from pydantic import BaseModel, Field

class GitHubStatsInput(BaseModel):
    lookback_days: int = Field(default=365, description="Number of days to look back for statistics (e.g. 7 for a week, 30 for a month, 365 for a year)")

class GitHubRepoStatsInput(BaseModel):
    lookback_days: int = Field(default=365, description="Number of days to look back for statistics")
    intent_category_name: str = Field(description="Name of the intent category to get repo stats for (e.g. 'raas', 'medical taxonomy')")

# Set up Graph Builder with State
graph_builder = StateGraph(State)

# Add tools
github_stats = GitHubStats()
tool_github_user_stats = Tool(
        name="github_user_stats",
        func=github_stats.get_user_stats,
        description="""Retrieves Eric Washington's overall GitHub statistics for a specified time period.
        Requires only one parameter:
        - lookback_days: Number of days to look back (e.g. 7 for a week, 30 for a month, 365 for a year)""",
        args_schema=GitHubStatsInput
    )

tool_github_repo_stats = StructuredTool.from_function(
        func=github_stats.get_repo_stats,
        name="github_repo_stats",
        description="""Retrieves Eric Washington's GitHub statistics for specific repositories over a specified time period.
        Requires two parameters:
        - intent_category_name: Name of the intent category to get repo stats for (e.g. 'raas', 'medical taxonomy')
        - lookback_days: (Optional) Number of days to look back. If not specified, defaults to 365 days."""
    )

tools = [tool_github_user_stats, tool_github_repo_stats]

# Add nodes
graph_builder.add_node("intent_classification", IntentClassificationNode())
graph_builder.add_node("github_stats_classification", GitHubStatsIntentClassificationNode())
graph_builder.add_node("question_answering", QuestionAnsweringNode(tools=tools))

# Add edges for parallel execution of intent classification nodes
graph_builder.add_edge(START, "intent_classification")
graph_builder.add_edge(START, "github_stats_classification") 

# Both nodes feed into question answering
graph_builder.add_edge("intent_classification", "question_answering")
graph_builder.add_edge("github_stats_classification", "question_answering")
graph_builder.add_edge("question_answering", END)

# Compile the graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)