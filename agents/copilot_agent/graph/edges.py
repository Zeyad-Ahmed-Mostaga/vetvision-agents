"""
agents/copilot_agent/graph/edges.py — Conditional Routing Logic
================================================================
Determines whether the LLM wants to call tools or deliver a final answer.
"""

from langgraph.graph import END

from agents.copilot_agent.graph.state import CopilotState


def should_continue(state: CopilotState) -> str:
    """
    Conditional edge: check if the last message contains tool calls.

    Returns:
        "tools"   — if the LLM emitted tool_calls → route to ToolNode
        "__end__" — if no tool_calls → final answer, end the graph
    """
    last_message = state["messages"][-1]

    # Check for tool calls on the message
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END
