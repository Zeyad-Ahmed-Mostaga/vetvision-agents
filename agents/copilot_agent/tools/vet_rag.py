"""
agents/copilot_agent/tools/vet_rag.py — Veterinary RAG Search Tool
====================================================================
Searches the VetVision veterinary knowledge base via Advanced RAG.
"""

import logging
from typing import Literal

from langchain_core.tools import tool

from rag.retrieval import advanced_rag_retrieve

logger = logging.getLogger(__name__)


@tool
def vet_rag_search(
    question: str,
    animal_type: Literal["cat", "dog", "horse", "other"],
) -> str:
    """
    Search the VetVision veterinary knowledge base using Advanced RAG.
    Use this tool to find veterinary information about diseases, diet, toxins,
    behavior, or general pet care.

    RULES:
    - 'question' MUST be in English (translate from Arabic before calling).
    - 'animal_type' must be confirmed from the user BEFORE calling this tool.
    - Do NOT call if animal type is unknown — ask first.

    Args:
        question:    English description of the veterinary question or symptoms.
        animal_type: Type of animal — 'cat', 'dog', 'horse', or 'other'.

    Returns:
        Top 3 ranked veterinary knowledge chunks as a formatted string.
    """
    logger.info("[vet_rag_search] animal=%s | question=%.60s...", animal_type, question)

    try:
        docs = advanced_rag_retrieve(question=question, animal_type=animal_type)
    except Exception as exc:
        logger.error("[vet_rag_search] Unexpected error: %s", exc, exc_info=True)
        return (
            f"RAG RESULT: Tool error — {exc}. "
            "Consider using the tavily_search tool as a fallback."
        )

    if not docs:
        return (
            "RAG RESULT: No relevant information found in the VetVision knowledge base "
            "for the given question and animal type. "
            "Consider using the tavily_search tool as a fallback."
        )

    formatted = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        source      = m.get("source", "VetVision KB")
        animal      = m.get("animal_type", animal_type)
        topic       = m.get("Topic", "")
        subtopic    = m.get("SubTopic", "")
        chunk_index = m.get("chunk_index", "?")

        header = f"--- Chunk {i} | Animal: {animal} | Source: {source} | Chunk#: {chunk_index}"
        if topic:
            header += f" | Topic: {topic}"
        if subtopic:
            header += f" | SubTopic: {subtopic}"
        header += " ---"

        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(formatted)
