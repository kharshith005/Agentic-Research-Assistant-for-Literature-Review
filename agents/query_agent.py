"""Query Generation Agent: research question -> search queries + paper fetch."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm import get_llm, invoke_llm
from agents.state import ResearchState
from agents.utils import extract_json
from tools.semantic_scholar import search_papers

SYSTEM_PROMPT = """You are an academic search expert.
Given a research question, produce:
1. Exactly 2 short keyword queries optimized for the Semantic Scholar API
   (plain keywords / boolean-friendly phrases, not full sentences).
2. Explicit inclusion/exclusion criteria distilled from the question.

Respond with ONLY valid JSON in this shape:
{
  "search_queries": ["query1", "query2"],
  "inclusion_criteria": "bullet-style criteria text"
}
"""

PAPERS_PER_QUERY = 6


def _dedupe_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for paper in papers:
        pid = paper.get("paperId") or paper.get("title") or ""
        if not pid or pid in seen:
            continue
        seen.add(pid)
        unique.append(paper)
    return unique


def query_agent_node(state: ResearchState) -> dict[str, Any]:
    question = state["research_question"]
    llm = get_llm(temperature=0)
    response = invoke_llm(
        llm,
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Research question:\n{question}"),
        ],
    )
    parsed = extract_json(response.content)
    queries = [q.strip() for q in parsed.get("search_queries", []) if str(q).strip()]
    criteria = str(parsed.get("inclusion_criteria", "")).strip()

    if not queries:
        queries = [question]
    if not criteria:
        criteria = question

    raw_papers: list[dict[str, Any]] = []
    for query in queries:
        raw_papers.extend(search_papers(query, limit=PAPERS_PER_QUERY))

    return {
        "search_queries": queries,
        "inclusion_criteria": criteria,
        "raw_papers": _dedupe_papers(raw_papers),
    }
