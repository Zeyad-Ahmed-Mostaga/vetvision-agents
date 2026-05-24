"""
routers/copilot.py — Vet Copilot Endpoints (Doctor-Facing)
===========================================================
Endpoints:
  POST /copilot/chat                — SSE streaming chat for veterinary doctors
  GET  /copilot/patient/{animal_id} — Patient history lookup by 6-char Animal ID
  POST /copilot/generate-report     — Direct PDF report generation (bypass agent)
  GET  /copilot/reports/{filename}  — Serve generated PDF files
  GET  /copilot/health              — Service health check

These endpoints serve veterinary doctors on the VetVision platform.
"""

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator

from config import settings
from agents.copilot_agent.graph.builder import get_copilot, DEFAULT_THREAD_ID
from db.crud import get_patient_history as db_get_history
from rag.store import get_vector_store, get_collection_point_counts
from agents.copilot_agent.tools.report.pipeline import REPORTS_DIR, shutdown_browser

logger = logging.getLogger("vetvision.copilot")

router = APIRouter(prefix="/copilot", tags=["Vet Copilot — Doctor Agent"])

# ── Request Models ────────────────────────────────────────────────────────────
_MAX_MSG_LEN = 3000


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty.")
        if len(v) > _MAX_MSG_LEN:
            v = v[:_MAX_MSG_LEN]
        return v


class ReportRequest(BaseModel):
    animal_name:  str
    animal_type:  str
    owner_name:   str
    weight_kg:    float
    diagnosis:    str
    treatment:    str
    doctor_name:  str          = "Doctor"
    doctor_notes: str          = ""


# ── POST /copilot/chat — Streaming ───────────────────────────────────────────
@router.post("/chat", summary="Stream copilot response (SSE)")
async def chat(request: ChatRequest):
    """
    Stream the copilot's response token-by-token as Server-Sent Events.

    Events:
      {"type": "token",  "content": "<text>"}
      {"type": "done",   "thread_id": "<id>"}
      {"type": "error",  "content": "<message>"}
    """
    thread_id = DEFAULT_THREAD_ID

    logger.info("CHAT | thread=%s | msg_len=%d", thread_id, len(request.message))

    copilot = get_copilot()
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            # Padding to bypass Cloudflare/NGINX initial buffering
            yield f": {' ' * 8192}\n\n"
            await asyncio.sleep(0)

            chunk_count = 0
            logger.info("[Copilot Stream] Starting to stream from %s...", settings.openrouter_model)

            async for msg_chunk, metadata in copilot.astream(
                {"messages": [("user", request.message)]},
                config=config,
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "")
                if node != "router":
                    continue
                if not hasattr(msg_chunk, "content") or not msg_chunk.content:
                    continue
                if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls and not msg_chunk.content:
                    continue

                payload = json.dumps({"type": "token", "content": msg_chunk.content}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                chunk_count += 1

                # Log every 10th chunk to track streaming progress
                if chunk_count % 10 == 0:
                    logger.info("[Copilot Stream] Streamed %d chunks so far", chunk_count)

                # Force event loop to flush the chunk immediately
                await asyncio.sleep(0)

            logger.info("[Copilot Stream] Completed with %d chunks", chunk_count)

            done = json.dumps({"type": "done", "thread_id": thread_id}, ensure_ascii=False)
            yield f"data: {done}\n\n"
            await asyncio.sleep(0)

        except Exception as exc:
            logger.error("Streaming error: %s", exc, exc_info=True)
            error = json.dumps(
                {"type": "error", "content": "عذراً، حصل خطأ غير متوقع. حاول تاني. 🐾"},
                ensure_ascii=False,
            )
            yield f"data: {error}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── GET /copilot/patient/{animal_id} ─────────────────────────────────────────
@router.get("/patient/{animal_id}", summary="Get patient history by Animal ID")
async def patient_history(animal_id: str):
    """Retrieve full patient history by 6-character Animal ID."""
    animal_id = animal_id.strip().upper()
    data = db_get_history(animal_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Patient not found: {animal_id}")
    return data


# ── POST /copilot/generate-report ────────────────────────────────────────────
@router.post("/generate-report", summary="Generate PDF report directly (bypass agent)")
async def generate_report(request: ReportRequest):
    """Generate a PDF report directly, bypassing the chat agent."""
    from agents.copilot_agent.tools.report.pipeline import generate_report_pipeline

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate_report_pipeline(
                animal_name=request.animal_name,
                animal_type=request.animal_type,
                owner_name=request.owner_name,
                weight_kg=request.weight_kg,
                diagnosis=request.diagnosis,
                treatment=request.treatment,
                doctor_name=request.doctor_name,
                doctor_notes=request.doctor_notes,
            )
        )
        return {"status": "ok", "result": result}
    except Exception as exc:
        logger.error("Report generation failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(exc)},
        )


# ── GET /copilot/reports/{filename} ──────────────────────────────────────────
@router.get("/reports/{filename}", summary="Download a generated PDF report")
async def serve_report(filename: str):
    """Serve a generated PDF report file."""
    filepath = REPORTS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail=f"Report not found: {filename}")
    return FileResponse(
        path=str(filepath),
        media_type="application/pdf",
        filename=filename,
    )


# ── GET /copilot/health ──────────────────────────────────────────────────────
@router.get("/health", summary="Vet Copilot health check")
async def health():
    """Service health check with Qdrant status."""
    try:
        _, mode = get_vector_store()
        counts = get_collection_point_counts()
        return {
            "status":        "ok",
            "agent":         "vet-copilot (doctor agent)",
            "qdrant_mode":   settings.qdrant_mode,
            "qdrant_active": mode,
            "collection":    settings.collection_name,
            "points":        counts,
            "agent_model":   settings.openrouter_model,
            "thread_id":     DEFAULT_THREAD_ID,
            "timestamp":     datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        logger.error("Health check error: %s", exc)
        return JSONResponse(
            status_code=200,
            content={
                "status":    "degraded",
                "agent":     "vet-copilot (doctor agent)",
                "error":     str(exc),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
