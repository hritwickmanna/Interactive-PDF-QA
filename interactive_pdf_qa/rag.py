"""Builders for RAG components: retriever, prompts, chains."""
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .config import RETRIEVAL_K


def build_retriever(splits, embeddings: HuggingFaceEmbeddings):
    """Build a vector store retriever from document splits and embeddings.

    Uses FAISS (fast, CPU-friendly, no sqlite dependency).
    """
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


def build_contextualize_prompt() -> ChatPromptTemplate:
    """Prompt that converts a context-dependent question into a standalone one."""
    contextualize_q_system_prompt = (
        "Given a chat history and latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. Do not answer the question, "
        "just return the formulated question if it is needed and otherwise return it as is."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def build_qa_prompt() -> ChatPromptTemplate:
    """Prompt used for answering the question with retrieved context."""
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "If the question is not answerable with the context, respond with 'I don't know'."
        "Answer concisely and accurately.\n\n{context}"
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def build_history_aware_retriever(llm: ChatGroq, retriever, contextualize_prompt: ChatPromptTemplate):
    """Wrap the base retriever with a step that rewrites questions using chat history."""
    return create_history_aware_retriever(llm, retriever, contextualize_prompt)


def build_rag_chain(llm: ChatGroq, history_aware_retriever, qa_prompt: ChatPromptTemplate):
    """Create the end-to-end RAG chain (retrieval + question answering)."""
    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
    )
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
