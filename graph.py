"""
graph.py – LangGraph conversation graph.

Topology: START → agent_node → output_node → END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from .graph_state import ConversationState
from .nodes import agent_node, output_node


def build_graph():
    graph = StateGraph(ConversationState)
    graph.add_node("agent_node",  agent_node)
    graph.add_node("output_node", output_node)
    graph.set_entry_point("agent_node")
    graph.add_edge("agent_node",  "output_node")
    graph.add_edge("output_node", END)
    return graph.compile()


# Module-level singleton imported by consumers.py
compiled_graph = build_graph()
