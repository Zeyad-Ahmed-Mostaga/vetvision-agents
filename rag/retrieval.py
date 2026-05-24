"""
rag/retrieval.py — 6-step Advanced RAG Pipeline
================================================
Exact replication of the notebook's retrieval logic with full fault tolerance.

Steps:
  1. Query Enhancement  — translate + rewrite into clinical English (Groq)
  2. HyDE              — hypothetical textbook passage (Groq)
  3. Metadata Filter   — Qdrant filter: animal_type == requested type
  4. Dual Retrieval    — similarity_search with query + HyDE, k=8 each
  5. Deduplication     — merge, dedup by first 200 chars of content
  6. Cohere Reranking  — ContextualCompressionRetriever → top 3

Fault tolerance:
  - Groq fails on Step 1 → use raw original query
  - Groq fails on Step 2 → skip HyDE, use only the enhanced query
  - Cohere fails on Step 6 → return top 3 from merged list (unranked)
  - Any unexpected error → logged, returns [] (caller handles gracefully)

Public API:
    advanced_rag_retrieve(question, animal_type, k=8) → list[Document]
"""

import logging
import time
from typing import List

from langchain_groq import ChatGroq
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import settings
from rag.store import get_vector_store, get_fallback_vector_store

logger = logging.getLogger(__name__)

# ── Lazy singletons — avoid heavy SDK loads at import time ────────────────────
_utility_llm = None
_enhance_query_chain = None
_hyde_chain = None
_cohere_reranker = None


def _get_utility_llm():
    global _utility_llm
    if _utility_llm is None:
        _utility_llm = ChatGroq(
            model=settings.utility_model,
            temperature=settings.utility_temperature,
            groq_api_key=settings.groq_api_key,
        )
    return _utility_llm


def _get_enhance_query_chain():
    global _enhance_query_chain
    if _enhance_query_chain is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful veterinary assistant. "
             "Translate (if necessary) and rewrite the following query into a clear, simple English search query "
             "for a veterinary medical knowledge base. "
             "CRITICAL: The target animal is a '{animal_type}'. You MUST explicitly include '{animal_type}' in your query. "
             "Focus on the core issue (symptoms, diet, toxins, general care). "
             "Return ONLY the search query — no preamble, no explanation."),
            ("human", "{query}")
        ])
        _enhance_query_chain = prompt | _get_utility_llm() | StrOutputParser()
    return _enhance_query_chain


def _get_hyde_chain():
    global _hyde_chain
    if _hyde_chain is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a veterinary expert. Write a SHORT (3-5 sentence) hypothetical answer or relevant passage "
             "that directly addresses the following veterinary query. "
             "Be specific, factual, and informative. English only. Return ONLY the passage."),
            ("human", "{enhanced_query}")
        ])
        _hyde_chain = prompt | _get_utility_llm() | StrOutputParser()
    return _hyde_chain


def _get_cohere_reranker():
    global _cohere_reranker
    if _cohere_reranker is None:
        _cohere_reranker = CohereRerank(
            model=settings.cohere_rerank_model,
            top_n=settings.rerank_top_n,
            cohere_api_key=settings.cohere_api_key,
        )
    return _cohere_reranker


class StaticRetriever(BaseRetriever):
    """
    Passes a static document list through to ContextualCompressionRetriever.
    Exact copy from notebook — required because CohereRerank wraps a retriever,
    not a raw document list.
    """
    docs: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.docs


def advanced_rag_retrieve(
    question: str,
    animal_type: str,
    k: int | None = None,
) -> List[Document]:
    """
    Execute the full 6-step Advanced RAG pipeline.

    Args:
        question:    The user's veterinary question (any language).
        animal_type: 'cat', 'dog', 'horse', or 'other'.
        k:           Docs per retrieval leg. Defaults to settings.retrieval_k.

    Returns:
        Top 3 reranked Document chunks, or [] if nothing found.
    """
    if k is None:
        k = settings.retrieval_k

    vector_store, _ = get_vector_store()

    # ── Step 1: Query Enhancement ─────────────────────────────────────────────
    try:
        enhanced_query = _get_enhance_query_chain().invoke({
            "query": question,
            "animal_type": animal_type,
        })
        logger.info("[RAG Step 1] Enhanced query: %.100s...", enhanced_query)
    except Exception as exc:
        logger.warning("[RAG Step 1] Groq enhancement failed (%s) — using raw query.", exc)
        enhanced_query = question

    # ── Step 2: HyDE ──────────────────────────────────────────────────────────
    try:
        hyde_text = _get_hyde_chain().invoke({"enhanced_query": enhanced_query})
        logger.info("[RAG Step 2] HyDE passage generated (%d chars).", len(hyde_text))
    except Exception as exc:
        logger.warning("[RAG Step 2] HyDE generation failed (%s) — skipping HyDE leg.", exc)
        hyde_text = None

    # ── Step 3: Metadata filter ───────────────────────────────────────────────
    metadata_filter = Filter(
        must=[FieldCondition(key="metadata.animal_type", match=MatchValue(value=animal_type))]
    )

    # ── Step 4: Dual retrieval (with retry for transient Jina/Qdrant errors) ──
    def _search_with_retry(query_text: str, retries: int = 3) -> List[Document]:
        fallback_store = get_fallback_vector_store()
        for attempt in range(1, retries + 1):
            try:
                return vector_store.similarity_search(query_text, k=k, filter=metadata_filter)
            except Exception as exc:
                if attempt == retries:
                    if fallback_store:
                        logger.warning("[RAG Step 4] Primary search failed (%s). Trying fallback store...", exc)
                        try:
                            return fallback_store.similarity_search(query_text, k=k, filter=metadata_filter)
                        except Exception as fallback_exc:
                            logger.error("[RAG Step 4] Both primary and fallback searches failed: %s", fallback_exc)
                            return []
                    else:
                        logger.error("[RAG Step 4] Search failed after %d attempts: %s", retries, exc)
                        return []
                wait = 2 * attempt
                logger.warning(
                    "[RAG Step 4] Search attempt %d/%d failed (%s) — retrying in %ds...",
                    attempt, retries, exc, wait,
                )
                time.sleep(wait)
        return []  # unreachable, but satisfies type checker

    docs_query = _search_with_retry(enhanced_query)
    docs_hyde: List[Document] = []
    if hyde_text:
        docs_hyde = _search_with_retry(hyde_text)
    logger.info(
        "[RAG Step 4] Retrieved %d (query) + %d (HyDE).",
        len(docs_query), len(docs_hyde),
    )

    # ── Step 5: Deduplication ─────────────────────────────────────────────────
    seen: set = set()
    merged: List[Document] = []
    for doc in docs_query + docs_hyde:
        fp = doc.page_content.strip()[:200]
        if fp not in seen:
            seen.add(fp)
            merged.append(doc)
    logger.info("[RAG Step 5] After dedup: %d unique chunks.", len(merged))

    if not merged:
        return []

    # ── Step 6: Cohere Reranking ──────────────────────────────────────────────
    try:
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=_get_cohere_reranker(),
            base_retriever=StaticRetriever(docs=merged),
        )
        final_docs = compression_retriever.invoke(enhanced_query)
        logger.info("[RAG Step 6] After Cohere reranking: %d chunks.", len(final_docs))
        return final_docs
    except Exception as exc:
        logger.warning(
            "[RAG Step 6] Cohere reranking failed (%s) — returning top %d unranked.",
            exc, settings.rerank_top_n,
        )
        return merged[: settings.rerank_top_n]
