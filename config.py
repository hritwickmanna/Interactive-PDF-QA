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
    """Load environment variables from .env and export any required tokens.

    This ensures dependencies like Hugging Face can read tokens from env.
    """
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
