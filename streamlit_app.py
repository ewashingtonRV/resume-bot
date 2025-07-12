import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import logging
import uuid
import time

# Import the pre-configured graph
from src.graph import graph
from src.state import State

# Configure logging
logging.basicConfig(level=logging.INFO)

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        # Generate a unique thread ID for this session
        st.session_state.thread_id = str(uuid.uuid4())

def invoke_graph_with_question(question: str):
    """Invoke the graph with a new question and return the response."""
    # Use existing messages from session state (user message already added)
    state = {
        "messages": st.session_state.messages.copy(),
        "next": "intent_classification",
        "intent": None,
        "context": None,
        "answer": None
    }
    
    # Create config with thread_id for checkpointer
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }
    
    # Invoke the graph
    try:
        result = graph.invoke(state, config=config)
        
        # Update session state with the complete result
        st.session_state.messages = result["messages"]
        
        return result["messages"][-1]  # Return the assistant's response
        
    except Exception as e:
        st.error(f"Error invoking graph: {str(e)}")
        logging.error(f"Graph invocation error: {str(e)}")
        return None

def display_chat_history():
    """Display the existing chat history."""
    for message in st.session_state.messages:
        # Convert LangChain messages to dictionary format
        if isinstance(message, HumanMessage):
            msg_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            msg_dict = {"role": "assistant", "content": message.content}
        else:
            continue  # Skip unknown message types
        
        with st.chat_message(msg_dict["role"]):
            st.markdown(msg_dict["content"])

def main():
    st.title("Resume Bot Chat")
    st.markdown("Ask me questions about my experience and projects! \
                The bot is currently geared towards providing specifics about the RVOH bullets in my resume")
    
    # Initialize session state
    initialize_session_state()
    
    # Display existing chat history
    display_chat_history()
    
    # Chat input at the bottom
    if prompt := st.chat_input("Ask me anything about my experience..."):
        # Add user message to session state first
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = invoke_graph_with_question(prompt)
            
            if response:
                # Stream the response
                st.write_stream(response_generator(response.content))
            else:
                st.error("Sorry, I encountered an error processing your question.")
    
    # Sidebar with additional info
    with st.sidebar:
        st.header("Chat Controls")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            # Generate new thread_id when clearing history
            st.session_state.thread_id = str(uuid.uuid4())
            st.success("Chat history cleared!")
            st.rerun()
        
        st.header("Debug Info")
        st.text(f"Total messages: {len(st.session_state.messages)}")
        st.text(f"Thread ID: {st.session_state.thread_id}")
        
        if st.checkbox("Show raw messages"):
            st.json([{"type": type(msg).__name__, "content": msg.content} 
                    for msg in st.session_state.messages])

if __name__ == "__main__":
    main() 