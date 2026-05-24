"""
scripts/add_pdf.py — Add a New PDF to the Existing Qdrant Collection
======================================================================
Extracts text from a PDF using PyMuPDF (fitz), chunks it with default
RecursiveCharacterTextSplitter separators (NOT the medical Markdown-specific
ones — those are tuned for the pre-parsed Docling output), then embeds
and upserts into the existing Qdrant collection.

Usage:
    python scripts/add_pdf.py <pdf_path> <animal_type>

    animal_type: cat | dog | horse | other

Example:
    python scripts/add_pdf.py "C:/data/NewPetGuide.pdf" cat
"""

import sys
import logging
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import settings
from rag.store import get_all_vector_stores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vetvision.add_pdf")

# Default separators — NOT the medical Markdown ones (those are Docling-specific)
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1250,
    chunk_overlap=250,
    add_start_index=True,
)

_VALID_ANIMAL_TYPES = {"cat", "dog", "horse", "other"}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF using PyMuPDF (fast, no ML required)."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/add_pdf.py <pdf_path> <animal_type>")
        print(f"  animal_type: {' | '.join(sorted(_VALID_ANIMAL_TYPES))}")
        sys.exit(1)

    pdf_path    = sys.argv[1]
    animal_type = sys.argv[2].lower()

    if animal_type not in _VALID_ANIMAL_TYPES:
        logger.error(
            "Invalid animal_type '%s'. Must be one of: %s",
            animal_type, ", ".join(sorted(_VALID_ANIMAL_TYPES)),
        )
        sys.exit(1)

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    logger.info("📄 Loading PDF: %s", pdf_file.name)
    raw_text = extract_text_from_pdf(str(pdf_file))
    logger.info("  Extracted %d characters.", len(raw_text))

    # Wrap in a Document so the text splitter can process it
    source_doc = Document(
        page_content=raw_text,
        metadata={"source": pdf_file.name, "animal_type": animal_type},
    )

    chunks = _text_splitter.split_documents([source_doc])
    # Filter very short chunks
    chunks = [c for c in chunks if len(c.page_content.strip()) >= settings.min_chunk_length]
    logger.info("  %d chunks produced after filtering.", len(chunks))

    if not chunks:
        logger.warning("No usable chunks found in '%s'. Aborting.", pdf_file.name)
        sys.exit(0)

    logger.info("⏳ Upserting into collection '%s'...", settings.collection_name)
    stores = get_all_vector_stores()
    if not stores:
        logger.error("No vector stores available!")
        sys.exit(1)
        
    for store in stores:
        doc_ids = store.add_documents(chunks)
        
    logger.info(
        "✅ Upserted %d chunks from '%s' (animal_type='%s') into %d Qdrant store(s).",
        len(chunks), pdf_file.name, animal_type, len(stores),
    )


if __name__ == "__main__":
    main()
