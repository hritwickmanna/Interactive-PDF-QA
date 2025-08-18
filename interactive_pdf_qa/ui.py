"""UI helper components for the Streamlit app."""
import streamlit as st


def render_header():
    st.title("Interactive PDF QA: RAG with Conversation Memory")
    st.write("Upload PDF(s) and ask questions with chat history.")


def render_api_key_input() -> str:
    return st.text_input("Enter your Groq API key:", type="password")


def render_session_input() -> str:
    return st.text_input("Session ID", value="default_session")


def render_file_uploader():
    return st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)


def render_question_input() -> str:
    return st.text_input("Ask a question:")


def render_agent_toggle() -> bool:
    """Whether to enable external web tools (Wikipedia/Arxiv/Search) via an agent."""
    return st.checkbox("Enable web tools (Wikipedia, Arxiv, Web search)")
