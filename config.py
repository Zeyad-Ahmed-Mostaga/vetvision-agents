"""
config.py — VetVision Unified Central Configuration
=====================================================
Single source of truth for ALL environment variables, paths, model names,
and tunable parameters for both the User Agent and the Vet Copilot.

Import `settings` from this module everywhere.

Usage:
    from config import settings
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env from the project root (same directory as this file)
_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env")

logger = logging.getLogger(__name__)


def _require(key: str) -> str:
    """Return env var value or raise a clear error at startup."""
    value = os.getenv(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is missing. "
            f"Add it to your .env file (see .env.example)."
        )
    return value


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Markdown source paths (pre-parsed by Docling — used by indexing scripts only)
# ─────────────────────────────────────────────────────────────────────────────
_MD_BASE = _optional(
    "MD_BASE_PATH",
    str(_ROOT / "data" / "markdown_sources"),
)

MARKDOWN_SOURCES: dict[str, str] = {
    "cat":   _optional("MD_CAT",   str(Path(_MD_BASE) / "Cat.md")),
    "dog":   _optional("MD_DOG",   str(Path(_MD_BASE) / "dog.md")),
    "horse": _optional("MD_HORSE", str(Path(_MD_BASE) / "Horse.md")),
    "other": _optional(
        "MD_OTHER",
        str(Path(_MD_BASE) / "Veterinary_Guide_Animal_Owners.md"),
    ),
}


@dataclass
class Settings:
    # ── API Keys (required) ──────────────────────────────────────────────────
    groq_api_key:         str = field(default_factory=lambda: _require("GROQ_API_KEY"))
    jina_api_key:         str = field(default_factory=lambda: _require("JINA_API_KEY"))
    cohere_api_key:       str = field(default_factory=lambda: _require("COHERE_API_KEY"))
    tavily_api_key:       str = field(default_factory=lambda: _require("TAVILY_API_KEY"))
    openrouter_api_key:   str = field(default_factory=lambda: _require("OPENROUTER_API_KEY"))
    google_api_key:       str = field(default_factory=lambda: _require("GOOGLE_API_KEY"))

    # ── Agent LLM: Primary (OpenRouter) ─────────────────────────────────────
    openrouter_model:     str = field(default_factory=lambda: _optional("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free"))
    openrouter_base_url:  str = "https://openrouter.ai/api/v1"
    agent_temperature:    float = 0.1

    # ── Agent LLM: Fallback (Gemini) — used by User Agent only ───────────────
    gemini_model:         str = field(default_factory=lambda: _optional("GEMINI_MODEL", "gemini-2.0-flash"))

    # ── Utility / Helper LLM (Groq) ──────────────────────────────────────────
    # Used for: query enhancement, HyDE, Arabic translation, report content
    groq_model:           str = field(default_factory=lambda: _optional("GROQ_MODEL", "llama-3.3-70b-versatile"))
    utility_model:        str = "llama-3.3-70b-versatile"
    utility_temperature:  float = 0.0

    # ── Embeddings ───────────────────────────────────────────────────────────
    jina_model:           str = "jina-embeddings-v3"
    jina_dimensions:      int = 1024

    # ── Qdrant Mode ───────────────────────────────────────────────────────────
    qdrant_mode:          str = field(default_factory=lambda: _optional("QDRANT_MODE", "local").lower())

    # ── Qdrant Cloud (primary) ────────────────────────────────────────────────
    qdrant_cloud_url:     str = field(default_factory=lambda: _optional("QDRANT_CLOUD_URL"))
    qdrant_cloud_api_key: str = field(default_factory=lambda: _optional("QDRANT_CLOUD_API_KEY"))

    # ── Qdrant On-Disk (fallback) ─────────────────────────────────────────────
    qdrant_local_path:    str = field(default_factory=lambda: _optional("QDRANT_LOCAL_PATH", "./data/qdrant_db"))
    collection_name:      str = field(default_factory=lambda: _optional("COLLECTION_NAME", "vetvision"))

    # ── Chunking (used by indexing scripts) ───────────────────────────────────
    chunk_size:           int = 1250
    chunk_overlap:        int = 250
    min_chunk_length:     int = 100

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_k:          int = 8       # docs per leg (query + HyDE)
    rerank_top_n:         int = 3       # Cohere top-N after reranking
    cohere_rerank_model:  str = "rerank-v3.5"

    # ── Agent context window ──────────────────────────────────────────────────
    context_window_messages: int = 10   # trim to last N messages (copilot uses 10)

    # ── Indexing (used by build_index.py) ─────────────────────────────────────
    index_batch_size:     int = 200
    index_batch_sleep:    float = 25.0
    index_ratelimit_sleep: float = 90.0

    # ── Database (Vet Copilot — SQLite patient records) ───────────────────────
    sqlite_db_path:       str = field(default_factory=lambda: _optional("SQLITE_DB_PATH", "./data/patients.db"))

    # ── Reports (Vet Copilot — PDF generation) ───────────────────────────────
    reports_dir:          str = field(default_factory=lambda: _optional("REPORTS_DIR", "./data/reports"))
    report_templates_dir: str = field(default_factory=lambda: _optional("REPORT_TEMPLATES_DIR", "./data/templates"))
    report_fonts_dir:     str = field(default_factory=lambda: _optional("REPORT_FONTS_DIR", "./data/fonts"))
    report_llm_retries:   int = 3          # LLM retry attempts per report phase
    report_rag_min_chars: int = 300        # RAG sufficiency threshold for web search fallback

    # ── LangSmith (optional) ──────────────────────────────────────────────────
    langsmith_api_key:    str = field(default_factory=lambda: _optional("LANGSMITH_API_KEY"))
    langsmith_project:    str = field(default_factory=lambda: _optional("LANGSMITH_PROJECT", "VetVision-Unified"))

    def __post_init__(self) -> None:
        """Set env vars so all LangChain integrations pick them up automatically."""
        os.environ["GROQ_API_KEY"]   = self.groq_api_key
        os.environ["JINA_API_KEY"]   = self.jina_api_key
        os.environ["COHERE_API_KEY"] = self.cohere_api_key
        os.environ["TAVILY_API_KEY"] = self.tavily_api_key
        os.environ["GOOGLE_API_KEY"] = self.google_api_key

        # Ensure data directories exist at startup
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_templates_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_fonts_dir).mkdir(parents=True, exist_ok=True)

        if self.langsmith_api_key:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
            logger.info("LangSmith tracing enabled — project: %s", self.langsmith_project)
        else:
            os.environ.pop("LANGSMITH_TRACING", None)


# Singleton — import this everywhere
settings = Settings()
