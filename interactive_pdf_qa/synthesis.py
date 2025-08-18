from __future__ import annotations

from typing import Optional


def synthesize_combined_answer(llm, question: str, pdf_answer: Optional[str], web_answer: Optional[str]) -> str:
    pdf_part = pdf_answer if pdf_answer else "N/A"
    web_part = web_answer if web_answer else "N/A"
    synthesis_prompt = (
        "You are a helpful assistant. Combine the following answers into one clear, concise response "
        "to the user's question. Prefer precise facts from the PDF answer when available, and "
        "augment with web information only if it adds non-conflicting useful context. Do not mention "
        "sources or that you combined answers.\n\n"
        f"Question: {question}\n"
        f"PDF answer: {pdf_part}\n"
        f"Web answer: {web_part}\n\n"
        "Final combined answer:"
    )
    combined_msg = llm.invoke(synthesis_prompt)
    return getattr(combined_msg, "content", str(combined_msg))
