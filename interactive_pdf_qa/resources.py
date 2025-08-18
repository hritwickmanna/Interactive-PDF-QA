"""Cached resources such as embeddings and LLM clients."""
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from .config import MODEL_NAME, EMBEDDING_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create and cache the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str) -> ChatGroq:
    """Create and cache the LLM client for a given API key."""
    return ChatGroq(
        groq_api_key=api_key,
        model_name=MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
