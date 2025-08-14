"""Global configuration and environment setup for the Streamlit RAG app."""
import os
from dotenv import load_dotenv

# ---------------------------
# App constants and settings
# ---------------------------
MODEL_NAME = "gemma2-9b-it"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500


def load_env() -> None:
    """Load environment variables from .env/Streamlit secrets and export tokens.

    Ensures dependencies (e.g., Hugging Face) can read tokens from env.
    Looks for HF tokens in this order: Streamlit secrets -> OS env -> .env
    and sets common env names used by different libraries.
    """
    # Load from .env first
    load_dotenv()

    # Read from OS env first (may be set by deployment platform)
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )

    # Prefer Streamlit secrets if available
    try:
        import streamlit as st  # type: ignore

        secrets = getattr(st, "secrets", None)
        if secrets:
            token = (
                secrets.get("HF_TOKEN", token)
                or secrets.get("HUGGINGFACEHUB_API_TOKEN", token)
                secrets.get("HF_TOKEN") or token
                or secrets.get("HUGGINGFACEHUB_API_TOKEN") or token
                or secrets.get("HUGGINGFACE_HUB_TOKEN") or token
            )
    except (ImportError, AttributeError):
        # Streamlit not available or secrets not configured; ignore
        pass

    # Normalize across common env var names
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
