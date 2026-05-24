"""
scripts/build_index.py — Run-Once Indexing Pipeline
=====================================================
Loads all four pre-parsed Markdown veterinary books, runs the two-stage
chunking pipeline, and indexes the chunks into Qdrant (cloud or disk).

Idempotent: if the collection already has vectors, indexing is skipped.
Delete ./data/qdrant_db (or clear the cloud collection) to force re-indexing.

Usage:
    python scripts/build_index.py

Rate-limit handling:
    - 50 chunks per batch
    - 25s sleep between batches (Jina free tier limit)
    - Retry with exponential backoff on errors
"""

import sys
import time
import logging
import pickle
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings, MARKDOWN_SOURCES
from rag.chunking import chunk_markdown_file
from rag.embeddings import get_embeddings
from rag.store import get_all_vector_stores, ensure_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vetvision.build_index")

CHECKPOINT_FILE = Path("data/indexing_checkpoint.pkl")


def main() -> None:
    logger.info("=" * 60)
    logger.info("VetVision — Build Index Script")
    logger.info("=" * 60)

    # ── Step 1 & 2: Probing embedding dimension ──────────────────────────────
    logger.info("Probing embedding dimension...")
    embeddings = get_embeddings()
    sample_vec = embeddings.embed_query("dimension probe")
    vector_dim = len(sample_vec)
    logger.info("Embedding dimension: %d", vector_dim)

    # ── Step 3: Get stores & Create collection if absent ─────────────────────
    stores = get_all_vector_stores()
    if not stores:
        logger.error("No vector stores available!")
        sys.exit(1)
        
    for store in stores:
        ensure_collection(store.client, vector_dim)

    # ── Step 4: Check if already indexed ─────────────────────────────────────
    # We check the first active store
    collection_info = stores[0].client.get_collection(settings.collection_name)
    existing_count  = collection_info.points_count or 0

    if existing_count > 0:
        logger.info(
            "ℹ️  Collection already has %d vectors — skipping indexing.\n"
            "   To force re-index: clear the collection from Qdrant Cloud dashboard\n"
            "   (or delete '%s'), delete 'data/indexing_checkpoint.pkl', then re-run.",
            existing_count,
            settings.qdrant_local_path,
        )
        # Remove any stale checkpoint so future re-index starts from scratch
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("🗑️  Stale checkpoint file removed.")
        return

    # ── Step 5: Chunk all markdown files ─────────────────────────────────────
    logger.info("Loading and chunking %d markdown files...", len(MARKDOWN_SOURCES))
    all_chunks = []
    for animal_type, filepath in MARKDOWN_SOURCES.items():
        try:
            chunks = chunk_markdown_file(filepath, animal_type)
            all_chunks.extend(chunks)
            logger.info("  '%s': %d chunks", animal_type, len(chunks))
        except FileNotFoundError as exc:
            logger.error("  ❌ %s — skipping.", exc)
            continue

    if not all_chunks:
        logger.error("No chunks produced. Check your MD_BASE_PATH in .env.")
        sys.exit(1)

    # ── Step 6: Load Checkpoint ──────────────────────────────────────────────
    start_index = 0
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "rb") as f:
            start_index = pickle.load(f)
        logger.info("🔄 Resuming from chunk index: %d", start_index)

    # ── Step 7: Batch index with rate-limit handling ──────────────────────────
    batch_size = 50
    doc_ids    = []

    for i in range(start_index, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        batch_end = i + len(batch)
        logger.info("⏳ Indexing chunks %d–%d ...", i, batch_end)

        retries = 0
        while retries < 5:
            try:
                batch_ids = None
                for store in stores:
                    batch_ids = store.add_documents(batch)
                
                doc_ids.extend(batch_ids or [])
                
                # Update checkpoint
                with open(CHECKPOINT_FILE, "wb") as f:
                    pickle.dump(batch_end, f)
                
                logger.info("  ✅ %d / %d indexed to %d store(s).", batch_end, len(all_chunks), len(stores))
                break
            except Exception as exc:
                retries += 1
                wait = 2 ** retries * 5
                logger.warning("  ⚠️  Error indexing batch: %s. Retrying in %ds...", exc, wait)
                time.sleep(wait)
        else:
            logger.error("  ❌ Batch failed after retries.")
            sys.exit(1)

        # Inter-batch sleep to respect Jina's rate limits
        if batch_end < len(all_chunks):
            logger.info("  ⏸️  Sleeping %.0fs before next batch...", settings.index_batch_sleep)
            time.sleep(settings.index_batch_sleep)

    logger.info("=" * 60)
    logger.info("✅ Indexing complete — %d vectors stored in '%s'.", len(doc_ids), settings.collection_name)
    logger.info("=" * 60)

    # Clean up checkpoint — full indexing succeeded, no need to resume
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("🗑️  Checkpoint file removed (indexing fully complete).")


if __name__ == "__main__":
    main()
