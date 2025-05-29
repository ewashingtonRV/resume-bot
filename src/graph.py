from langgraph.graph import StateGraph, START, END
import logging
from src.state import State
from langgraph.checkpoint.memory import MemorySaver
from src.nodes import IntentClassificationNode, QuestionAnsweringNode

# Set up Graph Builder with State
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("intent_classification", IntentClassificationNode())
graph_builder.add_node("question_answering", QuestionAnsweringNode())
graph_builder.add_node("end", END)

# Add edges
graph_builder.add_edge(START, "intent_classification")
graph_builder.add_edge("intent_classification", "question_answering", condition=lambda x: x["next"] == "question_answering")
graph_builder.add_edge("intent_classification", "end", condition=lambda x: x["next"] == "end")
graph_builder.add_edge("question_answering", "end")

# Compile the graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)