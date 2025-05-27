from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import pandas as pd
import numpy as np
# import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langsmith import utils

import uuid
import time
# openai_api_key = os.getenv('OPENAI_API_KEY')
# openai = OpenAI()
model_name = "gpt-4o"
stream = True

llm = ChatOpenAI(model=model_name)
# utils.tracing_is_enabled()


def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.set_page_config(layout="wide", page_title="API Test Client")
st.title("Chat Completion API Test Client")

# --- Session State Initialization ---
# Use session state to keep track of the conversation
if "messages" not in st.session_state:
    st.session_state.messages = [] # Store message history like {role: "user"/"assistant", content: "..."}
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_request" not in st.session_state:
    st.session_state.last_request = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "resume" not in st.session_state:
    st.session_state.resume = None

# Start the conversation
st.subheader("Conversation")
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("What is up?"):
    pass  # If user input comes from chat input, no need to change anything
    # Predefined questions
# questions = ["What's the weather like today?", "Tell me a joke", "What can you do?"]
# cols = st.columns(len(questions))
# # Create a button in each column
# question_buttons = [cols[i].button(q) for i, q in enumerate(questions)]

# if not user_input and any(question_buttons):
#     user_input = questions[question_buttons.index(True)]

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.resume = user_input  # Still track locally for UI continuity
    st.session_state.last_request = user_input
    st.session_state.last_response = None # Clear previous response
    response = llm.invoke(st.session_state.messages)
    assistant_output = response.content
    with st.chat_message("assistant"):
            st.write_stream(response_generator(assistant_output))
            st.session_state.messages.append({"role": "assistant", "content": assistant_output})