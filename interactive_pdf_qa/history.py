"""Chat history helpers for session-scoped histories in Streamlit."""
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def ensure_store():
    """Ensure a place in session_state to keep per-session chat histories."""
    if "store" not in st.session_state:
        st.session_state.store = {}


def get_session_history(session: str) -> BaseChatMessageHistory:
    """Get or initialize the chat history object for a given session id."""
    ensure_store()
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


def build_conversational_chain(rag_chain):
    """Wrap the RAG chain so it can read/write chat history automatically."""
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
