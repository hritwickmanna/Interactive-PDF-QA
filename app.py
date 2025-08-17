## RAG Q&A conversation with PDF Including Chat History.
# Entry point for the Streamlit app. Orchestrates the flow using modular helpers.

import streamlit as st
from contextlib import suppress

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
    render_agent_toggle,
)

# New modular helpers
from agents import build_web_agent, compute_index_sig, compute_agent_sig
from synthesis import synthesize_combined_answer

# Keep Streamlit callback handler for agent thoughts display
from langchain.callbacks import StreamlitCallbackHandler

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

    # Toggle for enabling external tools/agent
    use_tools = render_agent_toggle()

    # Ingest PDFs
    uploaded_files = render_file_uploader()

    # If nothing uploaded and no cached chain, stop early unless tools are enabled
    if not uploaded_files and "conversational_rag_chain" not in st.session_state and not use_tools:
        return

    # If files uploaded, rebuild only when they change
    if uploaded_files:
        sig = compute_index_sig(uploaded_files)
        if st.session_state.get("index_sig") != sig:
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
                # Store both wrapped and unwrapped versions
                st.session_state["rag_chain_unwrapped"] = rag_chain
                st.session_state["conversational_rag_chain"] = build_conversational_chain(rag_chain)
                st.session_state["index_sig"] = sig
                # Clear any previous agent so it can be rebuilt with the new PDF tool
                st.session_state.pop("web_agent", None)
                st.session_state.pop("agent_sig", None)
                st.success("Document index is ready.")

    conversational_rag_chain = st.session_state.get("conversational_rag_chain")

    # Build agent (optionally) using web tools and PDF QA tool (when available)
    if use_tools:
        # Create a signature so we only rebuild when inputs change
        agent_sig = compute_agent_sig(api_key, session_id)

        if st.session_state.get("agent_sig") != agent_sig:
            st.session_state["web_agent"] = build_web_agent(llm, session_id)
            st.session_state["agent_sig"] = agent_sig

    # If neither PDFs nor tools are available
    if not conversational_rag_chain and not use_tools:
        st.info("Upload a PDF to start.")
        return

    # Ask question and display answer
    user_input = render_question_input()
    if user_input:
        # Tools-enabled path
        if use_tools and st.session_state.get("web_agent") is not None:
            with st.spinner("Thinking across PDFs and web tools..."):
                st_cb = None
                with suppress(Exception):
                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)

                # Get a PDF answer (side-effect free; do not auto-write history)
                pdf_answer = None
                if st.session_state.get("rag_chain_unwrapped") is not None:
                    session_history = get_session_history(session_id)
                    pdf_result = st.session_state["rag_chain_unwrapped"].invoke(
                        {"input": user_input, "chat_history": session_history.messages}
                    )
                    pdf_answer = pdf_result.get("answer")

                # Get an answer from web tools agent
                agent_answer = st.session_state["web_agent"].run(
                    user_input, callbacks=[st_cb] if st_cb else None
                )

                # Synthesize a single combined answer
                final_answer = synthesize_combined_answer(
                    llm, user_input, pdf_answer, agent_answer
                )

                # Always write one combined history turn for this session
                session_history = get_session_history(session_id)
                session_history.add_user_message(user_input)
                session_history.add_ai_message(final_answer)

                # Update agent memory as well so future tool calls are stateful
                agent_exec = st.session_state.get("web_agent")
                agent_mem = getattr(agent_exec, "memory", None)
                chat_mem = getattr(agent_mem, "chat_memory", None)
                if chat_mem is not None:
                    chat_mem.add_user_message(user_input)
                    chat_mem.add_ai_message(final_answer)

                # Display only the final combined answer
                st.write("Assistant:", final_answer)

                # Optional: Inspect the raw store for debugging
                st.write(st.session_state.store)
                session_history = get_session_history(session_id)
                st.write("Chat History:", session_history.messages)
        else:
            # RAG-only path
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            # Optional: Inspect the raw store for debugging
            st.write(st.session_state.store)

            st.write("Assistant:", response["answer"])
            st.write("Chat History:", session_history.messages)


if __name__ == "__main__":
    main()
