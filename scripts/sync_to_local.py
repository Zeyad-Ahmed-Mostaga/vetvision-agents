"""
scripts/sync_to_local.py — One-Time Cloud → Local Vector Sync
================================================================
Copies every point (vectors + payloads + IDs) from the Qdrant Cloud
collection to the local on-disk Qdrant instance.

Why this over re-indexing:
  - No Jina API calls → finishes in seconds, not hours.
  - Exact fidelity — identical point IDs, embeddings, and metadata.
  - Cloud is read-only — zero mutation risk.
  - Source Markdown files are NOT required.

What it does:
  1. Connects to Qdrant Cloud (reads QDRANT_CLOUD_URL + QDRANT_CLOUD_API_KEY).
  2. Reads the collection's vector config (dimension, distance).
  3. Creates the same collection locally if it doesn't exist.
  4. Paginates through all cloud points via scroll() with vectors=True.
  5. Upserts each batch into the local collection.
  6. Recreates the payload index on 'metadata.animal_type'.
  7. Verifies the sync by comparing point counts.

Usage:
    python scripts/sync_to_local.py

Options:
    --force    Recreate the local collection even if it already has points.
    --batch    Batch size for scroll + upsert (default: 200).

Prerequisites:
    - .env with QDRANT_CLOUD_URL, QDRANT_CLOUD_API_KEY, QDRANT_LOCAL_PATH,
      and COLLECTION_NAME filled in.
    - qdrant-client installed in the active venv.
"""

import sys
import time
import logging
import argparse
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
    PointStruct,
)

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vetvision.sync_to_local")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync Qdrant Cloud collection → local on-disk instance."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the local collection if it already exists.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=200,
        help="Number of points per scroll/upsert batch (default: 200).",
    )
    return parser.parse_args()


def connect_cloud() -> QdrantClient:
    """Connect to Qdrant Cloud and verify the target collection exists."""
    url = settings.qdrant_cloud_url
    api_key = settings.qdrant_cloud_api_key

    if not url or not api_key:
        logger.error(
            "QDRANT_CLOUD_URL and/or QDRANT_CLOUD_API_KEY are not set in .env. "
            "Cannot sync without a source Cloud instance."
        )
        sys.exit(1)

    logger.info("Connecting to Qdrant Cloud: %s", url)
    client = QdrantClient(url=url, api_key=api_key, timeout=30)

    # Verify the collection exists
    if not client.collection_exists(settings.collection_name):
        logger.error(
            "Collection '%s' does not exist on the Cloud cluster. Nothing to sync.",
            settings.collection_name,
        )
        sys.exit(1)

    info = client.get_collection(settings.collection_name)
    count = info.points_count or 0
    logger.info(
        "✅ Cloud connected — collection '%s' has %d points.",
        settings.collection_name,
        count,
    )

    if count == 0:
        logger.warning("⚠️  Cloud collection is empty. Nothing to sync.")
        sys.exit(0)

    return client


def connect_local() -> QdrantClient:
    """Open the local on-disk Qdrant instance."""
    local_path = settings.qdrant_local_path
    logger.info("Opening local Qdrant at: %s", local_path)

    # Ensure the parent directory exists
    Path(local_path).mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=local_path)
    logger.info("✅ Local Qdrant opened.")
    return client


def prepare_local_collection(
    cloud_client: QdrantClient,
    local_client: QdrantClient,
    force: bool,
) -> None:
    """
    Create the local collection with the same vector config as Cloud.
    If the collection already has points and --force is not set, abort.
    """
    name = settings.collection_name

    # Read the vector config from Cloud so local matches exactly
    cloud_info = cloud_client.get_collection(name)
    vectors_config = cloud_info.config.params.vectors

    # Extract dimension and distance from the Cloud config
    if isinstance(vectors_config, dict):
        # Named vectors — pick the first one (shouldn't happen in this project
        # since ensure_collection() creates an unnamed vector)
        first_key = next(iter(vectors_config))
        vec_params = vectors_config[first_key]
        logger.info("Cloud uses named vector '%s'.", first_key)
    else:
        # Single unnamed vector (VectorParams directly)
        vec_params = vectors_config

    vector_dim = vec_params.size
    vector_distance = vec_params.distance
    logger.info(
        "Cloud vector config: %d-dim, distance=%s",
        vector_dim,
        vector_distance.name if hasattr(vector_distance, "name") else vector_distance,
    )

    # Check if local collection already exists
    if local_client.collection_exists(name):
        local_info = local_client.get_collection(name)
        local_count = local_info.points_count or 0

        if local_count > 0 and not force:
            logger.error(
                "Local collection '%s' already has %d points. "
                "Use --force to delete and resync, or delete data/qdrant_db manually.",
                name,
                local_count,
            )
            sys.exit(1)

        if force:
            logger.warning("--force: deleting existing local collection '%s'.", name)
            local_client.delete_collection(name)

    # Create the collection with matching config
    local_client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_dim, distance=vector_distance),
    )
    logger.info(
        "✅ Local collection '%s' created (%d-dim, %s).",
        name,
        vector_dim,
        vector_distance.name if hasattr(vector_distance, "name") else vector_distance,
    )


def sync_points(
    cloud_client: QdrantClient,
    local_client: QdrantClient,
    batch_size: int,
) -> int:
    """
    Scroll through all Cloud points and upsert them into the local collection.
    Returns the total number of points synced.
    """
    name = settings.collection_name
    total_synced = 0
    offset = None  # None = start from the beginning
    batch_num = 0
    start_time = time.time()

    logger.info("Starting sync (batch_size=%d)...", batch_size)

    while True:
        # scroll() returns (points, next_offset)
        # with_vectors=True is critical — without it we'd get payloads only
        points, next_offset = cloud_client.scroll(
            collection_name=name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        # Convert ScrollPoint objects to PointStruct for upsert
        point_structs = [
            PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
            for point in points
        ]

        local_client.upsert(
            collection_name=name,
            points=point_structs,
        )

        batch_num += 1
        total_synced += len(points)
        logger.info(
            "  ⏳ Batch %d: synced %d points (total: %d)",
            batch_num,
            len(points),
            total_synced,
        )

        # If next_offset is None, we've reached the end
        if next_offset is None:
            break

        offset = next_offset

    elapsed = time.time() - start_time
    logger.info(
        "✅ Sync complete — %d points transferred in %.1f seconds.",
        total_synced,
        elapsed,
    )
    return total_synced


def create_payload_index(local_client: QdrantClient) -> None:
    """Recreate the payload index on metadata.animal_type (matches store.py)."""
    name = settings.collection_name
    try:
        local_client.create_payload_index(
            collection_name=name,
            field_name="metadata.animal_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("✅ Payload index created on 'metadata.animal_type'.")
    except Exception as exc:
        if "already exists" in str(exc).lower():
            logger.info("ℹ️  Payload index already exists.")
        else:
            logger.warning("⚠️  Could not create payload index: %s", exc)


def verify_sync(
    cloud_client: QdrantClient,
    local_client: QdrantClient,
) -> None:
    """Compare point counts between Cloud and Local."""
    name = settings.collection_name

    cloud_info = cloud_client.get_collection(name)
    local_info = local_client.get_collection(name)

    cloud_count = cloud_info.points_count or 0
    local_count = local_info.points_count or 0

    logger.info("=" * 50)
    logger.info("Verification:")
    logger.info("  Cloud points:  %d", cloud_count)
    logger.info("  Local points:  %d", local_count)

    if cloud_count == local_count:
        logger.info("  ✅ Counts match — sync verified!")
    else:
        logger.warning(
            "  ⚠️  Count mismatch! Cloud=%d, Local=%d. "
            "Difference: %d points.",
            cloud_count,
            local_count,
            abs(cloud_count - local_count),
        )
    logger.info("=" * 50)


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("VetVision — Cloud → Local Vector Sync")
    logger.info("=" * 60)

    # Step 1: Connect to both instances
    cloud_client = connect_cloud()
    local_client = connect_local()

    # Step 2: Prepare local collection (create or force-recreate)
    prepare_local_collection(cloud_client, local_client, force=args.force)

    # Step 3: Scroll + upsert all points
    total = sync_points(cloud_client, local_client, batch_size=args.batch)

    if total == 0:
        logger.warning("No points were synced. Check Cloud collection state.")
        return

    # Step 4: Recreate payload indexes
    create_payload_index(local_client)

    # Step 5: Verify
    verify_sync(cloud_client, local_client)

    logger.info("")
    logger.info("Done! Local Qdrant at '%s' is now a mirror of Cloud.", settings.qdrant_local_path)
    logger.info("You can set QDRANT_MODE=local or QDRANT_MODE=hybrid in .env.")


if __name__ == "__main__":
    main()
