from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from typing import List, Dict
from src.state import State
import src.prompts as prompts
import logging

def get_last_human_message(messages: List[BaseMessage]) -> str:
    """Extract the last human message from the chat history."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


class IntentClassificationNode:
    """Node for classifying user intent."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0
        )

    def __call__(self, state: State) -> State:
        """Classify the intent of the last message."""
        logging.info(f"IntentClassificationNode: Processing state with keys: {state.keys()}")
        
        last_message = get_last_human_message(state["messages"])
        logging.info(f"IntentClassificationNode: Last message: {last_message}")
        
        # Classify intent using a structured prompt
        response = self.llm.invoke(
            [
                SystemMessage(content=prompts.intent_classification_system_prompt),
                HumanMessage(content=last_message)
            ]
        )
        
        intent = response.content.strip().lower()
        state["intent"] = intent
        logging.info(f"IntentClassificationNode: Classified intent: {intent}")
        
        # Route based on intent
        if intent in ["recommendation model question", "medical taxonomy question"]:
            state["next"] = "question_answering"
        else:
            state["next"] = "end"
            
        logging.info(f"IntentClassificationNode: Next node: {state['next']}")
        return state


class QuestionAnsweringNode:
    """Node for answering questions using chat history."""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name)

    def __call__(self, state: State) -> State:
        """Process questions using the full message history for context."""
        logging.info(f"QuestionAnsweringNode: Processing state with keys: {state.keys()}")
        logging.info(f"QuestionAnsweringNode: Intent: {state.get('intent', 'NOT SET')}")
        
        # Ensure intent exists
        if "intent" not in state or state["intent"] is None:
            logging.info("QuestionAnsweringNode: Warning - intent not set, using fallback")
            state["intent"] = "general question"

        # Select appropriate system prompt based on intent
        if state["intent"] == "recommendation model":
            system_msg = prompts.recommendation_system_prompt
        elif state["intent"] == "medical taxonomy":
            system_msg = prompts.medical_taxonomy_system_prompt
        elif state["intent"] == "smart links":
            system_msg = prompts.smart_links_system_prompt
        else:
            raise ValueError(f"Invalid intent: {state['intent']}")
        
        # Create messages list with system message and existing chat history
        messages = [SystemMessage(content=system_msg)] + state["messages"]
        
        # Generate answer
        response = self.llm.invoke(messages)
        
        # Update state
        state["answer"] = response.content
        state["next"] = "end"
        
        # Add assistant's response to message history
        state["messages"].append(AIMessage(content=response.content))
        
        logging.info(f"QuestionAnsweringNode: Generated answer of length {len(response.content)}")
        return state
