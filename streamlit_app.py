import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import logging
import uuid
import time
import re
from src.utils import MarkdownReader

# Import the pre-configured graph
from src.graph import graph
from src.state import State

# Configure logging
logging.basicConfig(level=logging.INFO)

def response_generator(response):
    """Generator that yields chunks of text while preserving markdown formatting."""
    # Split by sentences or logical chunks instead of words to preserve formatting
    
    # Split by sentences but keep the delimiters
    sentences = re.split(r'(\n\n|\n|\.|\!|\?)', response)
    
    current_chunk = ""
    for part in sentences:
        current_chunk += part
        # Yield on sentence endings or line breaks
        if part in ['\n\n', '\n'] or (part in ['.', '!', '?'] and len(current_chunk.strip()) > 20):
            yield current_chunk
            current_chunk = ""
            time.sleep(0.1)  # Slower for better readability
    
    # Yield any remaining content
    if current_chunk.strip():
        yield current_chunk

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
    
    # Detect environment - simple check for localhost/local development
    is_local = False
    # Create config with thread_id for checkpointer
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        },
        "metadata": {
            "environment": "local" if is_local else "production",
            "is_local_testing": is_local,
            "interface": "streamlit"
        },
        "tags": ["resume-bot", "streamlit", "local" if is_local else "production"]
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
    st.title("Eric's Resume Bot")
    # Read resume markdown
    mdr = MarkdownReader()
    resume_text = mdr.read_markdown_files('./data/resume.md')
    
    # Display resume and bot introduction
    st.markdown("""Hi I'm Remy, Eric's resume bot! I am designed to answer questions about the RVOH bullets on Eric's resume.

Here are some sample questions you can ask me:
* Can you tell me more about how Eric's Medical Taxonomy Enrichment service works?
* What is Relevance as a Service and how does it work? 
* What are Eric's code contributions to the AI Agent projects in the last 30 days?
""")
    
    # Add download button for resume
    st.download_button(
        label="📄 Download Eric's Resume",
        data=resume_text,
        file_name="Eric_Washington_Resume.md",
        mime="text/markdown",
        help="Click to download Eric's resume as a Markdown file"
    )
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
                # Create a placeholder for streaming
                response_placeholder = st.empty()
                full_response = ""
                
                # Stream the response chunk by chunk
                for chunk in response_generator(response.content):
                    full_response += chunk
                    # Use markdown for proper formatting during streaming
                    response_placeholder.markdown(full_response + "▌")  # Add cursor
                
                # Final update without cursor
                response_placeholder.markdown(full_response)
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