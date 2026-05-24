"""
agents/copilot_agent/graph/builder.py — Graph Compilation
===========================================================
Builds and compiles the Vet Copilot LangGraph agent.

Architecture:
    router → (tool_calls?) → tools → router → ... → END

Nodes:
    1. router — LLM call with tools bound (the brain)
    2. tools  — ToolNode executes tool calls

Memory:
    MemorySaver with per-session thread_id.

Public API:
    build_copilot()  → CompiledStateGraph
    get_copilot()    → CompiledStateGraph (cached singleton)
"""

import logging

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agents.copilot_agent.graph.state import CopilotState
from agents.copilot_agent.graph.nodes import router_node
from agents.copilot_agent.graph.edges import should_continue
from agents.copilot_agent.tools import ALL_TOOLS
from config import settings

logger = logging.getLogger(__name__)

# ── Thread ID ─────────────────────────────────────────────────────────────────
# TODO: Replace with per-user session management (JWT → thread_id mapping).
# For the single-user MVP, all conversations share one thread.
DEFAULT_THREAD_ID = "default-copilot-thread"


def build_copilot():
    """
    Build and compile the Vet Copilot LangGraph agent.

    Graph topology:
        __start__ → router → tools → router → ... → __end__

    The router IS the agent — it thinks, decides, and routes.
    The tools node executes whatever the router asks for.
    handle_tool_errors=True ensures tool failures return structured
    error messages instead of crashing the graph.
    """
    # Tool node with error handling — never crashes
    tool_node = ToolNode(ALL_TOOLS, handle_tool_errors=True)

    builder = StateGraph(CopilotState)
    builder.add_node("router", router_node)
    builder.add_node("tools", tool_node)

    builder.set_entry_point("router")

    # router → tools (if tool_calls) OR → END (final answer)
    builder.add_conditional_edges("router", should_continue)

    # tools always loops back to router for next decision
    builder.add_edge("tools", "router")

    # MemorySaver: in-memory checkpointer for conversation persistence
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info(
        "✅ Vet Copilot compiled | model=%s | tools=%s | window=%d msgs",
        settings.openrouter_model,
        [t.name for t in ALL_TOOLS],
        settings.context_window_messages,
    )
    return graph


# ── Singleton ─────────────────────────────────────────────────────────────────
_copilot = None


def get_copilot():
    """Return the cached compiled copilot, building it on first call."""
    global _copilot
    if _copilot is None:
        _copilot = build_copilot()
    return _copilot
