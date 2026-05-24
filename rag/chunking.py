"""
rag/chunking.py — Two-stage Markdown chunking pipeline
=======================================================
Exact replication of the notebook's chunking logic.

Stage 1: MarkdownHeaderTextSplitter on ## (Topic) and ### (SubTopic),
         strip_headers=False so the embedding captures the section name.
Stage 2: RecursiveCharacterTextSplitter with chunk_size=1250,
         chunk_overlap=250, is_separator_regex=True, and the exact
         universal_separators list from the notebook (medical-specific).

Context prefix "Animal: {type} | Topic: {topic} | SubTopic: {subtopic}"
is prepended to every chunk's content so the embedding captures
section-level semantics, not just body text.

Chunks shorter than min_chunk_length (default 100) are skipped.

Public API:
    chunk_markdown_file(filepath, animal_type) → list[Document]
"""

import logging
from pathlib import Path
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from config import settings

logger = logging.getLogger(__name__)

# ── Stage 1: Header splitter config ──────────────────────────────────────────
_HEADERS_TO_SPLIT_ON = [
    ("##",  "Topic"),
    ("###", "SubTopic"),
]

_markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=_HEADERS_TO_SPLIT_ON,
    strip_headers=False,   # keep ## headers in content so Jina embeds the topic name
)

# ── Stage 2: Universal separators (copied verbatim from notebook) ─────────────
# These are carefully tuned to split on medical-Markdown boundaries first,
# then fall back to generic text boundaries. Do not reorder or remove.
UNIVERSAL_SEPARATORS = [
    r"\n---\n",
    r"\n(?=\*{0,2}(?:Treatment|TREATMENT)\*{0,2}\s*:)",
    r"\n(?=\*{0,2}(?:Prevention|PREVENTION)\*{0,2}\s*:)",
    r"\n(?=\*{0,2}(?:Diagnosis|DIAGNOSIS)\*{0,2}\s*:)",
    r"\n(?=\*{0,2}(?:Symptoms?|SYMPTOMS?)\*{0,2}\s*:)",
    r"\n(?=\*{0,2}(?:Signs?|SIGNS?)\*{0,2}\s*:)",
    r"\n(?=\*{0,2}(?:Causes?|CAUSES?)\*{0,2}\s*:)",
    r"\n(?=\*{0,2}(?:Prognosis|PROGNOSIS)\*{0,2}\s*:)",
    r"\n(?=\*{0,2}(?:Vaccination|VACCINATION)\*{0,2}\s*:)",
    r"\n(?=\*{0,2}(?:Management|MANAGEMENT)\*{0,2}\s*:)",
    r"\n(?=\*{0,2}(?:Emergency\s+Care|EMERGENCY\s+CARE)\*{0,2}\s*:)",
    r"\n(?=\*{0,2}(?:Public\s+Health|PUBLIC\s+HEALTH)\*{0,2}\s*:)",
    r"\n(?=\*{0,2}(?:Zoonotic\s+Risk|ZOONOTIC\s+RISK)\*{0,2}\s*:)",
    r"\n(?=(?:Note|NOTE)\s*:)",
    r"\n(?=(?:Warning|WARNING)\s*:)",
    r"\n(?=(?:Important|IMPORTANT)\s*:)",
    r"\n(?=(?:Caution|CAUTION)\s*:)",
    r"\n\n",
    r"\n(?=\d+\.\s)",
    r"\n(?=-\s)",
    r"\n(?=\*\s)",
    r"\n(?=·\s)",
    r"\n(?=•\s)",
    r"(?<!Dr)(?<!Mr)(?<!St)(?<!vs)(?<=\.) +(?=[A-Z])",
    r"(?<=\?) +(?=[A-Z])",
    r"(?<=!) +(?=[A-Z])",
    r"\n",
    r"(?<=\.)\s",
    r"(?<=\?)\s",
    r"(?<=!)\s",
    r";\s",
    r"\s",
    r"",
]

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=UNIVERSAL_SEPARATORS,
    is_separator_regex=True,
    add_start_index=True,
)


def build_context_prefix(metadata: dict) -> str:
    """
    Build a plain-text prefix from Topic / SubTopic / animal_type metadata.
    Returns an empty string if all fields are absent.

    Example output:
        Animal: cat | Topic: Wound Irrigation | SubTopic: Antiseptic Solutions
    """
    parts = []
    animal   = metadata.get("animal_type", "")
    topic    = metadata.get("Topic", "")
    subtopic = metadata.get("SubTopic", "")
    if animal:
        parts.append(f"Animal: {animal}")
    if topic:
        parts.append(f"Topic: {topic}")
    if subtopic:
        parts.append(f"SubTopic: {subtopic}")
    if not parts:
        return ""
    return " | ".join(parts)


def chunk_markdown_file(filepath: str, animal_type: str) -> list[Document]:
    """
    Run the two-stage chunking pipeline on a single Markdown file.

    Args:
        filepath:    Absolute or relative path to the .md file.
        animal_type: One of 'cat', 'dog', 'horse', 'other'.

    Returns:
        List of Document chunks, each with metadata injected and a
        context prefix prepended to page_content.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {filepath}")

    logger.info("📖 Chunking '%s' → animal_type='%s'", path.name, animal_type)
    raw_text = path.read_text(encoding="utf-8")

    source_metadata = {"source": path.name, "animal_type": animal_type}

    # Stage 1 — header-aware split
    md_chunks = _markdown_splitter.split_text(raw_text)

    # Stage 2 — size-limited split
    final_chunks = _text_splitter.split_documents(md_chunks)

    chunks: list[Document] = []
    for chunk in final_chunks:
        content = chunk.page_content.strip()

        # Skip empty and stub chunks
        if not content or len(content) < settings.min_chunk_length:
            continue

        # Re-inject source-level metadata (source, animal_type).
        # Topic and SubTopic are already present from Stage 1.
        chunk.metadata.update(source_metadata)

        # Assign a sequential index within this file for traceability
        chunk.metadata["chunk_index"] = len(chunks)

        # Prepend context prefix so the embedding vector captures
        # section-level semantics, not just body text.
        prefix = build_context_prefix(chunk.metadata)
        if prefix:
            prefix_block = f"[{prefix}]\n\n"
            if not content.startswith(prefix_block.strip()):
                chunk.page_content = prefix_block + content

        chunks.append(chunk)

    logger.info("  ✓ '%s': %d chunks produced.", animal_type, len(chunks))
    return chunks
