"""LangGraph wiring: Query -> Screening -> Synthesis."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.query_agent import query_agent_node
from agents.screening_agent import screening_agent_node
from agents.state import ResearchState
from agents.synthesis_agent import synthesis_agent_node


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("query_agent", query_agent_node)
    graph.add_node("screening_agent", screening_agent_node)
    graph.add_node("synthesis_agent", synthesis_agent_node)

    graph.add_edge(START, "query_agent")
    graph.add_edge("query_agent", "screening_agent")
    graph.add_edge("screening_agent", "synthesis_agent")
    graph.add_edge("synthesis_agent", END)

    return graph.compile()
