"""
routers/user_chat.py — User-Facing Agent Endpoints
==========================================================
Endpoints:
  POST /chat        — SSE streaming chat for pet owners
  GET  /health      — Service health check (Qdrant + user agent status)

These endpoints serve the user-facing VetVision app (pet owners).
"""

import json
import logging
import uuid
import asyncio
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import settings
from agents.user_agent.graph import get_agent
from rag.store import get_vector_store, get_collection_point_counts

logger = logging.getLogger("vetvision.user_chat")

router = APIRouter(tags=["User Agent"])

# ── Request / Response models ─────────────────────────────────────────────────
_MAX_MESSAGE_LEN = 2000


class ChatRequest(BaseModel):
    message:   str  = Field(..., min_length=1)
    thread_id: str  = Field(default_factory=lambda: str(uuid.uuid4()))
    reset:     bool = False

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty.")
        if len(v) > _MAX_MESSAGE_LEN:
            v = v[:_MAX_MESSAGE_LEN]
        return v


# ── POST /chat — Streaming ────────────────────────────────────────────────────
@router.post("/chat", summary="Stream user-agent response (SSE)")
async def chat(request: ChatRequest):
    """
    Stream the user agent's response token-by-token as Server-Sent Events.

    Each event is a JSON object:
      {"type": "token",  "content": "<text>"}
      {"type": "done",   "thread_id": "<id>"}
      {"type": "error",  "content": "<error message>"}
    """
    thread_id = str(uuid.uuid4()) if request.reset else request.thread_id

    logger.info(
        "CHAT | thread=%s | reset=%s | msg_len=%d | ts=%s",
        thread_id[:8], request.reset, len(request.message),
        datetime.utcnow().isoformat(),
    )

    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            # Padding to bypass Cloudflare/NGINX initial buffering
            yield f": {' ' * 8192}\n\n"
            await asyncio.sleep(0)

            async for msg_chunk, metadata in agent.astream(
                {"messages": [("user", request.message)]},
                config=config,
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "")
                if node != "agent":
                    continue
                if not hasattr(msg_chunk, "content") or not msg_chunk.content:
                    continue
                if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls and not msg_chunk.content:
                    continue

                payload = json.dumps({"type": "token", "content": msg_chunk.content}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                
                # Force event loop to flush the chunk immediately
                await asyncio.sleep(0)

            done_payload = json.dumps({"type": "done", "thread_id": thread_id}, ensure_ascii=False)
            yield f"data: {done_payload}\n\n"
            await asyncio.sleep(0)

        except Exception as exc:
            logger.error("Streaming error for thread %s: %s", thread_id, exc, exc_info=True)
            error_payload = json.dumps(
                {"type": "error", "content": "عذراً، حصل خطأ غير متوقع. حاول تاني بعد شوية. 🐾"},
                ensure_ascii=False,
            )
            yield f"data: {error_payload}\n\n"
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


# ── GET /health ───────────────────────────────────────────────────────────────
@router.get("/health", summary="User agent health check")
async def health():
    """Returns service status, active Qdrant backend, and indexed vector count."""
    try:
        _, mode = get_vector_store()
        counts = get_collection_point_counts()
        return {
            "status":        "ok",
            "agent":         "user-agent",
            "qdrant_mode":   settings.qdrant_mode,
            "qdrant_active": mode,
            "collection":    settings.collection_name,
            "points":        counts,
            "agent_model":   settings.openrouter_model,
            "timestamp":     datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        logger.error("Health check error: %s", exc)
        return JSONResponse(
            status_code=200,
            content={
                "status":    "degraded",
                "agent":     "user-agent",
                "error":     str(exc),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )