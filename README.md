# RAG Q&A with PDFs (Streamlit + LangChain)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://interactive-pdf.streamlit.app/)
Live app: https://interactive-pdf.streamlit.app/

Conversational question answering over your PDFs using a Retrieval-Augmented Generation (RAG) pipeline. The app supports chat history so follow-up questions are contextualized properly.


## Features
- Upload one or more PDF files and query them
- History-aware retrieval (follow-up questions work)
- Modular codebase for easy extension
- Caching of LLM and Embeddings for performance


## Tech Stack
- Streamlit for UI
- LangChain for retrieval and chaining
- Chroma as in-memory vector store
- HuggingFace embeddings (`all-MiniLM-L6-v2`)
- Groq LLM (`gemma2-9b-it` by default)


## Folder Structure
```
4.1 - RAG Q&A/
├── app.py                 # Streamlit entry point (orchestrates the flow)
├── config.py              # Global constants and environment loader
├── resources.py           # Cached resources (LLM and Embeddings)
├── pdf_utils.py           # PDF saving, loading, and splitting
├── rag.py                 # Builders for retriever, prompts, and RAG chain
├── history.py             # Session chat history management
├── ui.py                  # Streamlit UI helper components
└── README.md              # This file
```


## Prerequisites
- Python 3.10+
- A Groq API Key (required)
- Optional: HF_TOKEN if you use gated/private HF models


## Setup
1) Create and activate a virtual environment (recommended)

macOS/Linux (zsh):
```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

If you have a project-wide `requirements.txt`, you can do:
```
pip install -r requirements.txt
```

Otherwise, install the needed packages directly:
```
pip install -U \
  streamlit python-dotenv \
  langchain langchain-community langchain-text-splitters \
  langchain-chroma chromadb \
  langchain-groq groq \
  langchain-huggingface huggingface-hub sentence-transformers \
  pypdf
```

3) Configure environment variables

Option A: Use a `.env` file in this folder:
```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=optional_hf_token_here
```

Option B: Export in your shell (zsh):
```
export GROQ_API_KEY="your_groq_api_key_here"
export HF_TOKEN="optional_hf_token_here"
```


## Run the App
From this folder:
```
streamlit run app.py
```


## How to Use
- Enter your Groq API key in the UI
- (Optional) Provide a custom Session ID
- Upload one or more PDFs
- Ask a question; follow up with additional questions using chat history


## Configuration
Edit `config.py` to change defaults:
- `MODEL_NAME` (Groq model)
- `EMBEDDING_MODEL` (HuggingFace)
- `CHUNK_SIZE`, `CHUNK_OVERLAP` (document splitting)


## Notes & Troubleshooting
- Run Streamlit from this folder so relative imports work.
- If you change dependencies, restart Streamlit after `pip install`.
- Clear cache if needed: from the Streamlit menu > “Clear cache”.
- Current setup uses an in-memory Chroma vector store. For persistence, switch to a persistent directory and initialize Chroma with `persist_directory`.


## Extending
- Add custom prompts in `rag.py`.
- Swap models or parameters in `config.py`.
- Add new UI elements in `ui.py`.
- Implement persistence or advanced retrieval strategies in `rag.py`.


---

## Issues faced and how we addressed them

- Issues faced:
  - Inconsistent Hugging Face token handling between local and Streamlit Cloud.
  - Chroma/SQLite errors on Streamlit Cloud during vector store initialization.
  - Slow first question on large PDFs due to heavy indexing/embedding.
  - Streamlit reruns dropping state (e.g., “No PDF uploaded” after asking a question).

- Fixes applied in this repo:
  - Normalized secrets loading in `config.py`: reads Streamlit Secrets first, then env/.env; aligns `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, `HUGGINGFACE_HUB_TOKEN` for library compatibility.
  - Replaced Chroma with FAISS-only vector store in the implementation to avoid SQLite on Cloud; removed Chroma/SQLite deps from `requirements.txt` used by the app.
  - Cached embeddings and LLM; persisted the built RAG chain in `st.session_state` and rebuild only when uploads change to prevent rerun issues.
  - Tuned defaults for speed/consistency: `CHUNK_OVERLAP=100`, `RETRIEVAL_K=3`, `LLM_MAX_TOKENS=512`, `LLM_TEMPERATURE=0.2`.

Notes:
- If you prefer the original Chroma behavior locally, you can keep it; for Streamlit Cloud or portability, use FAISS as implemented.
- The README above reflects the original design; the fixes listed here describe how we made it reliable on both local and Cloud.
