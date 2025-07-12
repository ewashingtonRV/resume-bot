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
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        if api_key:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                api_key=api_key
            )
        else:
            # Fallback to environment variable
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0
            )

    def __call__(self, state: State) -> State:
        """Classify the intent of the last message."""
        logging.info(f"IntentClassificationNode: Processing state with keys: {state.keys()}")
        
        # Check if API key is provided in state
        if state.get("openai_api_key"):
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                api_key=state["openai_api_key"]
            )
            logging.info("✅ Using API key from state")
        else:
            llm = self.llm
            logging.info("Using default LLM configuration")
        
        last_message = get_last_human_message(state["messages"])
        logging.info(f"IntentClassificationNode: Last message: {last_message}")
        
        # Classify intent using a structured prompt
        response = llm.invoke(
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
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        if api_key:
            self.llm = ChatOpenAI(model=model_name, api_key=api_key)
        else:
            # Fallback to environment variable
            self.llm = ChatOpenAI(model=model_name)

    def __call__(self, state: State) -> State:
        """Process questions using the full message history for context."""
        logging.info(f"QuestionAnsweringNode: Processing state with keys: {state.keys()}")
        logging.info(f"QuestionAnsweringNode: Intent: {state.get('intent', 'NOT SET')}")
        
        # Check if API key is provided in state
        if state.get("openai_api_key"):
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=state["openai_api_key"]
            )
            logging.info("✅ Using API key from state")
        else:
            llm = self.llm
            logging.info("Using default LLM configuration")
        
        # Ensure intent exists
        if "intent" not in state or state["intent"] is None:
            logging.info("QuestionAnsweringNode: Warning - intent not set, using fallback")
            state["intent"] = "general question"
        
        # Normalize intent and select appropriate system prompt
        intent = state["intent"].lower().strip()
        
        if "recommendation" in intent:
            system_msg = prompts.recommendation_system_prompt
            logging.info("QuestionAnsweringNode: Using recommendation system prompt with project data")
        elif "medical" in intent or "taxonomy" in intent:
            system_msg = prompts.medical_taxonomy_system_prompt
            logging.info("QuestionAnsweringNode: Using medical taxonomy system prompt with project data")
        elif "smart" in intent or "link" in intent:
            system_msg = prompts.smart_links_system_prompt
            logging.info("QuestionAnsweringNode: Using smart links system prompt with project data")
        elif "article" in intent or "tagging" in intent:
            system_msg = prompts.article_tagging_system_prompt
            logging.info("QuestionAnsweringNode: Using article tagging system prompt with project data")
        elif "raas" in intent:
            system_msg = prompts.raas_system_prompt
            logging.info("QuestionAnsweringNode: Using raas system prompt with project data")
        elif "ds-lead" in intent:
            system_msg = prompts.ds_lead_system_prompt
            logging.info("QuestionAnsweringNode: Using ds-lead system prompt with project data")
        else:
            raise ValueError(f"Invalid intent: {state['intent']}")
        
        # Create messages list with system message and existing chat history
        messages = [SystemMessage(content=system_msg)] + state["messages"]
        
        # Generate answer
        response = llm.invoke(messages)
        
        # Update state
        state["answer"] = response.content
        state["next"] = "end"
        
        # Add assistant's response to message history
        state["messages"].append(AIMessage(content=response.content))
        
        logging.info(f"QuestionAnsweringNode: Generated answer of length {len(response.content)}")
        return state
