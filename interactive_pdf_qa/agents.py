from __future__ import annotations

from typing import List, Optional
from importlib import import_module

import streamlit as st

from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

from .history import get_session_history


def compute_index_sig(uploaded_files: List) -> str:
    """
    Create a signature for uploaded files so we rebuild the index only when they change.
    """
    return "|".join(
        f"{f.name}:{getattr(f, 'size', None) or len(f.getvalue())}" for f in uploaded_files
    )


def compute_agent_sig(api_key: str, session_id: str | int) -> str:
    parts: List[str] = [api_key[-4:] if len(api_key) >= 4 else "key", "v1", str(session_id)]
    if st.session_state.get("index_sig"):
        parts.append(st.session_state["index_sig"])  # tie to PDF index
    return "|".join(parts)


def _create_external_tools() -> List:
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    )
    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    )
    search = DuckDuckGoSearchRun(name="Search")
    return [wiki, arxiv, search]


def _create_pdf_tool(session_id: str | int) -> Optional[Tool]:
    if st.session_state.get("rag_chain_unwrapped") is None:
        return None

    def _pdf_tool_call(q: str) -> str:
        session_history = get_session_history(session_id)
        res = st.session_state["rag_chain_unwrapped"].invoke(
            {"input": q, "chat_history": session_history.messages}
        )
        return res.get("answer", "")

    return Tool(
        name="pdf_qa",
        description=(
            "Answer questions using the uploaded PDFs with conversation memory. "
            "Prefer this for questions about the uploaded documents."
        ),
        func=_pdf_tool_call,
    )


def _seed_agent_memory(session_id: str | int):
    ConversationBufferMemory = None
    try:
        ConversationBufferMemory = getattr(
            import_module("langchain.memory"), "ConversationBufferMemory"
        )
    except (ModuleNotFoundError, ImportError, AttributeError):
        ConversationBufferMemory = None

    if ConversationBufferMemory is None:
        return None

    agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Seed from existing session chat history
    session_history_obj = get_session_history(session_id)
    for m in getattr(session_history_obj, "messages", []):
        mtype = getattr(m, "type", "")
        content = getattr(m, "content", None)
        if not content:
            continue
        if mtype == "human":
            agent_memory.chat_memory.add_user_message(content)
        elif mtype == "ai":
            agent_memory.chat_memory.add_ai_message(content)

    return agent_memory


def build_web_agent(llm, session_id: str | int):
    tools = _create_external_tools()

    pdf_tool = _create_pdf_tool(session_id)
    if pdf_tool is not None:
        tools = [pdf_tool] + tools

    agent_memory = _seed_agent_memory(session_id)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=agent_memory,  # may be None; agent works without it
        handle_parsing_errors=True,
    )
    return agent
