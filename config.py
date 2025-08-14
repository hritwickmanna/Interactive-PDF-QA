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
    # Load from .env first (lowest priority when Streamlit secrets are present)
    load_dotenv()

    # Start with any existing env values
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    # Find the first environment variable that is set (even if empty string)
    for var in ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:
        value = os.getenv(var)
        if value is not None:
            token = value
            break
    else:
        token = None

    # Prefer Streamlit secrets if available
    try:
        import streamlit as st  # type: ignore

        secrets = getattr(st, "secrets", None)
        if secrets:
            token = (
                secrets.get("HF_TOKEN", token)
                or secrets.get("HUGGINGFACEHUB_API_TOKEN", token)
                or secrets.get("HUGGINGFACE_HUB_TOKEN", token)
            # Use explicit None checks for secrets fallback
            if secrets.get("HF_TOKEN", None) is not None:
                token = secrets.get("HF_TOKEN")
            elif secrets.get("HUGGINGFACEHUB_API_TOKEN", None) is not None:
                token = secrets.get("HUGGINGFACEHUB_API_TOKEN")
            elif secrets.get("HUGGINGFACE_HUB_TOKEN", None) is not None:
                token = secrets.get("HUGGINGFACE_HUB_TOKEN")
            # else, keep previous token value
    except (ImportError, AttributeError):
        # Streamlit not available or secrets not configured; ignore
        pass

    # Normalize across common env var names
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
