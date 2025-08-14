## RAG Q&A conversation with PDF Including Chat History (Modularized)
# Entry point for the Streamlit app. Orchestrates the flow using modular helpers.

from typing import List

import streamlit as st

# Local modules
from config import load_env
from resources import get_embeddings, get_llm
from pdf_utils import save_uploaded_pdfs, load_documents_from_pdfs, split_documents
from rag import (
    build_retriever,
    build_contextualize_prompt,
    build_qa_prompt,
    build_history_aware_retriever,
    build_rag_chain,
)
from history import ensure_store, get_session_history, build_conversational_chain
from ui import (
    render_header,
    render_api_key_input,
    render_session_input,
    render_file_uploader,
    render_question_input,
)

# Load environment variables
load_env()


# ---------------------------
# Main app flow
# ---------------------------

def main():
    # Header
    render_header()

    # API key + basic inputs
    api_key = render_api_key_input()
    if not api_key:
        st.warning("Please enter your Groq API key to use the application.")
        return

    # Initialize core resources
    llm = get_llm(api_key)
    embeddings = get_embeddings()

    # Chat session id and state store
    session_id = render_session_input()
    ensure_store()

    # Ingest PDFs
    uploaded_files = render_file_uploader()
    if not uploaded_files:
        # Don't proceed until files are provided
        return

    with st.spinner("Processing documents..."):
        pdf_paths = save_uploaded_pdfs(uploaded_files)
        documents = load_documents_from_pdfs(pdf_paths)
        splits = split_documents(documents)

        # Build retriever and RAG pipeline
        retriever = build_retriever(splits, embeddings)
        contextualize_prompt = build_contextualize_prompt()
        history_aware_retriever = build_history_aware_retriever(llm, retriever, contextualize_prompt)
        qa_prompt = build_qa_prompt()
        rag_chain = build_rag_chain(llm, history_aware_retriever, qa_prompt)
        conversational_rag_chain = build_conversational_chain(rag_chain)

    # Ask question and display answer
    user_input = render_question_input()
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {
                    "session_id": session_id,
                }
            },
        )
        # Optional: Inspect the raw store for debugging
        st.write(st.session_state.store)

        st.write("Assistant:", response["answer"])
        st.write("Chat History:", session_history.messages)


if __name__ == "__main__":
    main()
