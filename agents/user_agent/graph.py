"""
agents/user_agent/graph.py — LangGraph User Agent
===================================================
Builds the VetVision ReAct agent graph for the user-facing chatbot (فيتو).

Architecture:
  AgentState → agent_node → tools_condition → ToolNode → agent_node → …

Primary LLM:  OpenRouter via ChatOpenAI (model configurable via settings)
Fallback LLM: ChatGoogleGenerativeAI (Gemini) — transparent to the graph;
              switched automatically if OpenRouter raises an error.

Memory:  MemorySaver (per-thread conversation history)
Context: Sliding window of last N messages (trim_messages, strategy="last")

Public API:
    build_agent()   → CompiledStateGraph  (call once at startup)
    get_agent()     → CompiledStateGraph  (cached singleton)
"""

import logging
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, trim_messages
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from config import settings
from agents.user_agent.tools import TOOLS
from agents.user_agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ── Typed State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── LLM setup ─────────────────────────────────────────────────────────────────
def _make_primary_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openrouter_model,
        temperature=settings.agent_temperature,
        openai_api_base=settings.openrouter_base_url,
        openai_api_key=settings.openrouter_api_key,
        streaming=True,
        default_headers={
            "HTTP-Referer": "https://vetvision.app",
            "X-Title": "VetVision",
        },
    )


def _make_fallback_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.agent_temperature,
        streaming=True,
    )


_primary_llm_with_tools  = _make_primary_llm().bind_tools(TOOLS)
_fallback_llm_with_tools = _make_fallback_llm().bind_tools(TOOLS)


# ── Agent Node ────────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> AgentState:
    """
    Core agent node.
    - Trims messages to last N (settings.context_window_messages) — sliding window.
    - Prepends the system prompt before each LLM call.
    - Tries the primary LLM (OpenRouter); falls back to Gemini on any error.
    """
    trimmed = trim_messages(
        state["messages"],
        max_tokens=settings.context_window_messages,
        token_counter=len,        # count by number of messages, not tokens
        strategy="last",
        include_system=False,     # system message is added separately below
        allow_partial=False,
    )

    messages_to_send = [SystemMessage(content=SYSTEM_PROMPT)] + trimmed

    # Primary LLM
    try:
        logger.info("[User Agent] Invoking primary LLM (OpenRouter): %s", settings.openrouter_model)
        response = _primary_llm_with_tools.invoke(messages_to_send)
        logger.info("[User Agent] Primary LLM response received, length=%d", len(response.content) if response.content else 0)
        return {"messages": [response]}
    except Exception as exc:
        logger.warning(
            "Primary LLM (OpenRouter) failed (%s) — switching to Gemini fallback.", exc
        )

    # Fallback LLM
    try:
        response = _fallback_llm_with_tools.invoke(messages_to_send)
        return {"messages": [response]}
    except Exception as exc:
        logger.error("Fallback LLM (Gemini) also failed: %s", exc, exc_info=True)
        raise


# ── Graph factory ─────────────────────────────────────────────────────────────
def build_agent():
    """Build and compile the LangGraph ReAct agent. Call once at startup."""
    tool_node = ToolNode(TOOLS)

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.set_entry_point("agent")

    # agent → tools (if tool call present) OR → END
    builder.add_conditional_edges("agent", tools_condition)

    # tools always loops back to agent
    builder.add_edge("tools", "agent")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info(
        "✅ User Agent (فيتو) compiled | model=%s | tools=%s | window=%d msgs",
        settings.openrouter_model,
        [t.name for t in TOOLS],
        settings.context_window_messages,
    )
    return graph


# ── Singleton ─────────────────────────────────────────────────────────────────
_agent = None


def get_agent():
    """Return the cached compiled agent, building it on first call."""
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent
