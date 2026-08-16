import streamlit as st
import logging
import time
import re
from src.utils import MarkdownReader
import os

from src.chat import respond

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

def get_response(question: str) -> str | None:
    """Get Remy's answer for the current conversation."""
    try:
        answer = respond(
            st.session_state.messages,
            api_key=st.session_state.get("anthropic_api_key"),
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        logging.error(f"Response error: {str(e)}")
        return None

def display_chat_history():
    """Display the existing chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.title("Eric's Resume Bot")
    # Read resume markdown
    mdr = MarkdownReader()
    resume_text = mdr.read_markdown_files(os.path.join(os.path.dirname(__file__), '..', 'data', 'resume.md'))

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
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response with streaming
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(prompt)

            if response:
                # Create a placeholder for streaming
                response_placeholder = st.empty()
                full_response = ""

                # Stream the response chunk by chunk
                for chunk in response_generator(response):
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
            st.success("Chat history cleared!")
            st.rerun()

        st.header("Debug Info")
        st.text(f"Total messages: {len(st.session_state.messages)}")

        if st.checkbox("Show raw messages"):
            st.json(st.session_state.messages)

if __name__ == "__main__":
    main()
