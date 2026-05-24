"""
rag/embeddings.py — Jina Embeddings initializer
================================================
Returns a configured JinaEmbeddings instance (jina-embeddings-v3, 1024-dim).
Imported by both the indexing script and the retrieval pipeline at runtime.
"""

from langchain_community.embeddings import JinaEmbeddings
from config import settings


def get_embeddings() -> JinaEmbeddings:
    """
    Return a configured JinaEmbeddings instance.
    The JINA_API_KEY is already set in the environment by settings.__post_init__.
    """
    return JinaEmbeddings(
        model_name=settings.jina_model,
        jina_api_key=settings.jina_api_key,
        dimensions=settings.jina_dimensions,
    )
