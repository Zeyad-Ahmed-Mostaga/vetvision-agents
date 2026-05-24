"""
main.py — VetVision Unified FastAPI Application
================================================
Single entry point for the unified VetVision AI backend.

Mounts:
  /chat         — User-facing AI agent (فيتو) for pet owners
  /health       — User agent health check
  /copilot/*    — Vet Copilot for veterinary doctors
  /docs         — Swagger UI (auto-generated)
  /redoc        — ReDoc UI (auto-generated)

Run locally:
  uvicorn main:app --reload --port 8000

Deploy (Render / Heroku):
  web: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Load config first — sets all env vars before any LangChain import ─────────
from config import settings  # noqa: F401 — side effect: sets env vars

# ── Routers ───────────────────────────────────────────────────────────────────
from routers.user_chat import router as user_router
from routers.copilot import router as copilot_router

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("vetvision.main")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks for both agents."""
    # ── STARTUP ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("VetVision Unified — Starting up")
    logger.info("=" * 60)

    # Warm up vector store connection
    try:
        from rag.store import get_vector_store, get_collection_point_counts
        _, mode = get_vector_store()
        counts = get_collection_point_counts()
        logger.info("Qdrant ready | mode=%s | collection=%s | points=%s",
                    mode, settings.collection_name, counts)
    except Exception as exc:
        logger.warning("Qdrant warmup warning (non-fatal): %s", exc)

    # Compile both agents
    try:
        from agents.user_agent.graph import get_agent
        get_agent()
        logger.info("User Agent (فيتو) ready")
    except Exception as exc:
        logger.error("User Agent failed to compile: %s", exc, exc_info=True)

    try:
        from agents.copilot_agent.graph.builder import get_copilot
        get_copilot()
        logger.info("Vet Copilot ready")
    except Exception as exc:
        logger.error("Vet Copilot failed to compile: %s", exc, exc_info=True)

    logger.info("=" * 60)
    logger.info("VetVision Unified is ready — Swagger: http://localhost:8000/docs")
    logger.info("=" * 60)

    yield  # ── SHUTDOWN ─────────────────────────────────────────────────────

    logger.info("VetVision Unified — Shutting down...")
    try:
        from agents.copilot_agent.tools.report.pipeline import shutdown_browser
        shutdown_browser()
        logger.info("Playwright browser closed.")
    except Exception as exc:
        logger.warning("Browser shutdown warning: %s", exc)
    logger.info("VetVision Unified — Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VetVision AI — Unified Backend",
    description=(
        "Production-ready FastAPI backend for the VetVision veterinary platform.\n\n"
        "## Agents\n"
        "- **User Agent (فيتو)** — AI assistant for pet owners: `/chat`\n"
        "- **Vet Copilot** — AI assistant for veterinary doctors: `/copilot/chat`\n\n"
        "## Features\n"
        "- Advanced RAG over veterinary knowledge base (Qdrant + Jina + Cohere)\n"
        "- LangGraph agent orchestration with MemorySaver conversation persistence\n"
        "- SQLite patient record management (Vet Copilot)\n"
        "- Bilingual Arabic/English PDF report generation via Playwright (Vet Copilot)\n"
        "- Server-Sent Events (SSE) streaming for real-time responses"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ───────────────────────────────────────────────────────────
app.include_router(user_router)
app.include_router(copilot_router)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service":  "VetVision AI — Unified Backend",
        "version":  "2.0.0",
        "docs":     "/docs",
        "agents": {
            "user_agent":  "/chat",
            "vet_copilot": "/copilot/chat",
        },
        "health": {
            "user_agent":  "/health",
            "vet_copilot": "/copilot/health",
        },
    }
