"""
agents/user_agent/tools.py — Tool Definitions for the User Agent (فيتو)
=========================================================================
Defines the two tools available to the VetVision user-facing agent:

  1. vet_rag_search  — Advanced RAG over the four veterinary books.
  2. tavily_search   — Web search fallback (clinics, drugs, real-world data).
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from rag.retrieval import advanced_rag_retrieve
from config import settings

logger = logging.getLogger(__name__)


# ── Schema: VetVision RAG Search ──────────────────────────────────────────────
class VetRAGSearchInput(BaseModel):
    """Input schema for the vet_rag_search tool."""

    question: str = Field(
        ...,
        description=(
            "The veterinary question or symptom description to search for. "
            "MUST be written in English — translate from Arabic before calling. "
            "Be specific and include relevant clinical details "
            "(e.g., 'cat vomiting after eating grass' instead of just 'cat sick')."
        ),
    )
    animal_type: Literal["cat", "dog", "horse", "other"] = Field(
        ...,
        description=(
            "The type of animal the question is about. Must be one of: "
            "'cat', 'dog', 'horse', or 'other'. "
            "This MUST be confirmed from the user BEFORE calling — "
            "do NOT guess or assume the animal type."
        ),
    )


# ── Schema: Tavily Web Search ─────────────────────────────────────────────────
class TavilyWebSearchInput(BaseModel):
    """Input schema for the tavily_search web search tool."""

    query: str = Field(
        ...,
        description=(
            "The web search query. Formulate in the language most appropriate for "
            "the search target — use Arabic for Egyptian locations/clinics, English "
            "for scientific or medical topics. Rephrase into a concise, keyword-rich "
            "search query. For location searches, explicitly include the city and "
            "neighborhood (e.g., 'القاهرة المعادي عيادة بيطرية')."
        ),
    )


# ── Tool 1: VetVision Advanced RAG Search ────────────────────────────────────
@tool(args_schema=VetRAGSearchInput)
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
        question:    English description of the user's veterinary question or symptoms.
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


# ── Tool 2: Tavily Web Search ─────────────────────────────────────────────────
tavily_search = TavilySearch(
    max_results=4,
    topic="general",
    include_answer=True,
    tavily_api_key=settings.tavily_api_key,
    args_schema=TavilyWebSearchInput,
    description=(
        "Web search tool. Use this for: "
        "1. Fallback when 'vet_rag_search' fails or returns irrelevant results. "
        "2. Finding real-world locations (e.g., nearest veterinary clinics, hospitals in specific Egyptian cities). "
        "3. Looking up specific commercial medications, prices, or current real-world data. "
        "CRITICAL RULES: "
        "- You MUST formulate the search query in good way and the language of user. "
        "- Rephrase the user's input into a concise, keyword-rich search query. "
        "- For locations, explicitly include the city and neighborhood (e.g., 'القاهرة المعادي عيادة بيطرية')."
    ),
)

# Exported tool list (imported by graph.py)
TOOLS = [vet_rag_search, tavily_search]
