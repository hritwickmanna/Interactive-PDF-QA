"""Utilities for handling PDFs: saving uploads, loading and splitting documents."""
import tempfile
from typing import List

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from .config import CHUNK_SIZE, CHUNK_OVERLAP


def save_uploaded_pdfs(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    """Persist uploaded PDFs to temporary files and return file paths."""
    file_paths: List[str] = []
    for uf in uploaded_files:
        suffix = ".pdf" if not uf.name.lower().endswith(".pdf") else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}{suffix}") as tmp:
            tmp.write(uf.getvalue())
            file_paths.append(tmp.name)
    return file_paths


def load_documents_from_pdfs(pdf_paths: List[str]):
    """Load documents from a list of PDF file paths using PyPDFLoader."""
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
    return documents


def split_documents(documents):
    """Split raw documents into chunks suitable for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)
