"""
rag/store.py — Qdrant client + vector store initialization
===========================================================
Implements the hybrid Qdrant strategy based on QDRANT_MODE.

Public API:
    get_vector_store()          → (QdrantVectorStore, str)  (returns primary)
    get_fallback_vector_store() → QdrantVectorStore | None
    get_all_vector_stores()     → list[QdrantVectorStore]
    ensure_collection(...)      → creates collection if absent (idempotent)
    get_collection_point_counts() → dict
"""

import logging
from typing import Tuple, List, Optional, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
from langchain_qdrant import QdrantVectorStore

from config import settings
from rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Module-level cache
_cloud_client: QdrantClient | None = None
_local_client: QdrantClient | None = None
_cloud_store: QdrantVectorStore | None = None
_local_store: QdrantVectorStore | None = None
_initialized: bool = False


def init_clients() -> None:
    global _cloud_client, _local_client, _initialized
    if _initialized:
        return

    # 1. Cloud Client
    if settings.qdrant_cloud_url and settings.qdrant_cloud_api_key:
        try:
            logger.info("Connecting to Qdrant Cloud: %s", settings.qdrant_cloud_url)
            c = QdrantClient(
                url=settings.qdrant_cloud_url,
                api_key=settings.qdrant_cloud_api_key,
                timeout=10,
            )
            c.get_collections()
            logger.info("✅ Qdrant Cloud connected.")
            _cloud_client = c
        except Exception as exc:
            logger.warning("⚠️ Qdrant Cloud connection failed: %s", exc)
            _cloud_client = None

    # 2. Local Client
    try:
        from pathlib import Path as _Path
        _Path(settings.qdrant_local_path).mkdir(parents=True, exist_ok=True)
        logger.info("Connecting to local Qdrant at: %s", settings.qdrant_local_path)
        lc = QdrantClient(path=settings.qdrant_local_path)
        lc.get_collections()
        logger.info("✅ Local Qdrant ready.")

        # Auto-create the collection if it doesn't exist (clean-slate boot)
        if not lc.collection_exists(settings.collection_name):
            from rag.embeddings import get_embeddings
            _dim = settings.jina_dimensions  # 1024 — avoid an API call just to probe
            lc.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(size=_dim, distance=Distance.COSINE),
            )
            logger.info("✅ Created empty local collection '%s' (%d-dim COSINE).",
                         settings.collection_name, _dim)

        _local_client = lc
    except Exception as exc:
        logger.error("❌ Local Qdrant connection failed: %s", exc)
        _local_client = None

    _initialized = True


def _get_store(client: QdrantClient | None, store_cache: QdrantVectorStore | None) -> QdrantVectorStore | None:
    if store_cache is not None:
        return store_cache
    if client is None:
        return None
    
    embeddings = get_embeddings()
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )


def get_vector_store() -> Tuple[QdrantVectorStore, str]:
    """Return the PRIMARY (QdrantVectorStore, mode) based on QDRANT_MODE."""
    global _cloud_store, _local_store
    init_clients()
    
    mode = settings.qdrant_mode
    
    if mode == "local":
        _local_store = _get_store(_local_client, _local_store)
        if _local_store:
            return _local_store, "local"
        raise RuntimeError("QDRANT_MODE is 'local' but local client failed to initialize.")
        
    if mode == "cloud":
        _cloud_store = _get_store(_cloud_client, _cloud_store)
        if _cloud_store:
            return _cloud_store, "cloud"
        # Fallback if cloud fails but mode was cloud
        _local_store = _get_store(_local_client, _local_store)
        if _local_store:
            logger.warning("Cloud failed in 'cloud' mode. Falling back to local disk.")
            return _local_store, "local (fallback)"
        raise RuntimeError("Both Cloud and Local clients failed to initialize.")
        
    # Hybrid (default)
    _cloud_store = _get_store(_cloud_client, _cloud_store)
    if _cloud_store:
        return _cloud_store, "cloud (hybrid)"
    
    _local_store = _get_store(_local_client, _local_store)
    if _local_store:
        logger.warning("Cloud failed in 'hybrid' mode. Primary is now local disk.")
        return _local_store, "local (hybrid fallback)"
        
    raise RuntimeError("Both Cloud and Local clients failed to initialize.")


def get_fallback_vector_store() -> Optional[QdrantVectorStore]:
    """Return the FALLBACK QdrantVectorStore if available (only in hybrid mode when cloud is primary)."""
    global _cloud_store, _local_store
    init_clients()
    
    mode = settings.qdrant_mode
    if mode != "hybrid":
        return None
        
    # In hybrid mode, if cloud is primary (because it succeeded), local is fallback
    if _cloud_client is not None and _local_client is not None:
        _local_store = _get_store(_local_client, _local_store)
        return _local_store
        
    return None


def get_all_vector_stores() -> List[QdrantVectorStore]:
    """Return a list of all active vector stores (for dual-writes)."""
    global _cloud_store, _local_store
    init_clients()
    
    stores = []
    mode = settings.qdrant_mode
    
    if mode in ("cloud", "hybrid") and _cloud_client is not None:
        _cloud_store = _get_store(_cloud_client, _cloud_store)
        if _cloud_store:
            stores.append(_cloud_store)
            
    if mode in ("local", "hybrid") and _local_client is not None:
        _local_store = _get_store(_local_client, _local_store)
        if _local_store:
            stores.append(_local_store)
            
    return stores


def ensure_collection(client: QdrantClient, vector_dim: int) -> None:
    """Create the collection if absent and ensure payload indexes exist."""
    if not client.collection_exists(settings.collection_name):
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        logger.info("✅ Collection '%s' created (%d-dim COSINE).", settings.collection_name, vector_dim)
    else:
        logger.info("ℹ️  Collection '%s' already exists.", settings.collection_name)

    # Ensure keyword index on metadata.animal_type (required for Qdrant Cloud filtering)
    try:
        client.create_payload_index(
            collection_name=settings.collection_name,
            field_name="metadata.animal_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("✅ Payload index created on 'metadata.animal_type'.")
    except Exception as exc:
        if "already exists" in str(exc).lower():
            logger.info("ℹ️  Payload index on 'metadata.animal_type' already exists.")
        else:
            logger.warning("⚠️  Could not create payload index: %s", exc)


def get_collection_point_counts() -> Dict[str, dict]:
    """Return the statuses and point counts for both backends."""
    init_clients()
    res = {}
    
    def _get_count(client: QdrantClient | None) -> int:
        if not client: return -1
        try:
            return client.get_collection(settings.collection_name).points_count or 0
        except Exception:
            return -1

    if settings.qdrant_mode in ("cloud", "hybrid"):
        count = _get_count(_cloud_client)
        res["cloud"] = {"status": "ok" if count >= 0 else "error", "points": count}
        
    if settings.qdrant_mode in ("local", "hybrid"):
        count = _get_count(_local_client)
        res["local"] = {"status": "ok" if count >= 0 else "error", "points": count}
        
    return res
