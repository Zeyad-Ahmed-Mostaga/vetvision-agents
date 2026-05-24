"""
agents/copilot_agent/graph/state.py — Agent State Definition
==============================================================
CopilotState carries conversation messages only.

The doctor's name is NOT stored in state — the agent learns it
conversationally and it persists naturally in the message history,
which MemorySaver checkpoints between turns.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class CopilotState(TypedDict):
    """State for the Vet Copilot agent graph."""
    messages: Annotated[list, add_messages]
