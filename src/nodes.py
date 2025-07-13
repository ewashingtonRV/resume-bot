from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from typing import List, Dict
from src.state import State
import src.prompts as prompts
import logging
from langgraph.prebuilt import create_react_agent

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

    def __call__(self, state: State) -> Dict:
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
        logging.info(f"IntentClassificationNode: Classified intent: {intent}")
            
        # Return only the intent update
        return {"intent": intent}


class GitHubStatsIntentClassificationNode:
    """Node for determining if a question is asking about GitHub coding activity."""
    
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

    def __call__(self, state: State) -> Dict:
        """Determine if the question is about GitHub activity."""
        logging.info("GitHubStatsIntentClassificationNode: Processing state")
        
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
        
        # Use the GitHub stats classification prompt to determine if this is a GitHub stats question
        response = llm.invoke(
            [
                SystemMessage(content=prompts.github_stats_classification_system_prompt),
                HumanMessage(content=last_message)
            ]
        )
        
        # Convert response to boolean
        is_github_stats = response.content.strip().lower() == "true"
        logging.info(f"GitHubStatsIntentClassificationNode: Is GitHub stats question: {is_github_stats}")
        return {"is_github_stats": is_github_stats}


class QuestionAnsweringNode:
    """Node for answering questions using chat history."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None, tools: list = None):
        self.model_name = model_name
        self.api_key = api_key
        self.tools = tools
        
        # Initialize the LLM once
        if api_key:
            self.llm = ChatOpenAI(model=model_name, api_key=api_key)
        else:
            # Fallback to environment variable
            self.llm = ChatOpenAI(model=model_name)

    def create_agent(self, llm: ChatOpenAI, tools: list, system_message: str):
        """Create a new agent with the specified configuration using LangGraph."""
        return create_react_agent(llm, tools, prompt=system_message)

    def __call__(self, state: State) -> State:
        """Process questions using the full message history for context."""
        logging.info(f"QuestionAnsweringNode: Processing state with keys: {state.keys()}")
        logging.info(f"QuestionAnsweringNode: Intent: {state.get('intent', 'NOT SET')}")
        
        # Use state-provided API key if available
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
        
        # Normalize intent and get GitHub stats flag
        intent = state["intent"].lower().strip()
        is_github_stats = state.get("is_github_stats", False)
        logging.info(f"QuestionAnsweringNode: Is GitHub stats: {is_github_stats}")
        
        # Select the appropriate system prompt based on intent
        if "github stats" in intent:
            system_msg = prompts.github_system_prompt
            logging.info("QuestionAnsweringNode: Using github system prompt")
        elif intent in prompts.REFERENCE_PATHS:
            # Dynamically create system prompt with current GitHub stats flag
            system_msg = prompts.create_system_prompt(
                category_name=intent,
                reference_text_path=prompts.REFERENCE_PATHS[intent],
                is_github_stats=is_github_stats
            )
            logging.info(f"QuestionAnsweringNode: Created dynamic system prompt for {intent} with GitHub stats: {is_github_stats}")
        else:
            raise ValueError(f"Invalid intent: {state['intent']}")

        # Create agent with current configuration
        agent = self.create_agent(
            llm=llm,
            tools=self.tools if is_github_stats else [],
            system_message=system_msg
        )
        
        # Get the last human message
        last_message = get_last_human_message(state["messages"])
        logging.info(f"QuestionAnsweringNode: Using input message: {last_message}")
        
        try:
            # Run agent with the query
            response = agent.invoke({"messages": [HumanMessage(content=last_message)]})
            logging.info(f"QuestionAnsweringNode: Agent response: {response}")
            # Extract the response content from the agent's response
            if response and "messages" in response and response["messages"]:
                last_ai_message = response["messages"][-1]
                if hasattr(last_ai_message, 'content'):
                    state["answer"] = last_ai_message.content
                else:
                    state["answer"] = str(last_ai_message)
            else:
                state["answer"] = "I apologize, but I couldn't generate a response."
        except Exception as e:
            logging.error(f"Error in agent execution: {str(e)}")
            # Create messages list with system message and existing chat history
            messages = [SystemMessage(content=system_msg)] + state["messages"]
            # Fall back to regular chat completion
            response = llm.invoke(messages)
            state["answer"] = response.content
        
        # Add assistant's response to message history
        state["messages"].append(AIMessage(content=state["answer"]))
        
        logging.info(f"QuestionAnsweringNode: Generated answer of length {len(state['answer'])}")
        return state