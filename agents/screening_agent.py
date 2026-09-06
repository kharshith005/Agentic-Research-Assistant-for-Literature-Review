"""Screening Agent: filter papers by inclusion/exclusion criteria."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm import get_llm, invoke_llm
from agents.state import ResearchState
from agents.utils import extract_json

BATCH_SIZE = 10

SYSTEM_PROMPT = """You are a ruthless academic literature screener.
For each paper, decide INCLUDE or EXCLUDE based ONLY on the given criteria
and the paper's title + abstract. Be strict: if the abstract does not clearly
satisfy the criteria, EXCLUDE it.

Respond with ONLY valid JSON:
{
  "decisions": [
    {"index": 0, "decision": "INCLUDE", "reason": "one short sentence"},
    {"index": 1, "decision": "EXCLUDE", "reason": "one short sentence"}
  ]
}
Use the paper index numbers provided. decision must be INCLUDE or EXCLUDE.
"""


def _format_batch(papers: list[dict[str, Any]], offset: int) -> str:
    blocks: list[str] = []
    for i, paper in enumerate(papers):
        authors = ", ".join(
            a.get("name", "") for a in (paper.get("authors") or [])[:3]
        )
        abstract = (paper.get("abstract") or "No abstract available.").strip()
        blocks.append(
            f"[{offset + i}] Title: {paper.get('title', 'Untitled')}\n"
            f"Year: {paper.get('year', 'N/A')} | Authors: {authors or 'N/A'}\n"
            f"Abstract: {abstract}"
        )
    return "\n\n".join(blocks)


def screening_agent_node(state: ResearchState) -> dict[str, Any]:
    papers = state.get("raw_papers") or []
    criteria = state.get("inclusion_criteria") or state.get("research_question", "")

    if not papers:
        return {"screened_papers": [], "rejected_papers": []}

    llm = get_llm(temperature=0)
    screened: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for start in range(0, len(papers), BATCH_SIZE):
        batch = papers[start : start + BATCH_SIZE]
        human = (
            f"Inclusion/exclusion criteria:\n{criteria}\n\n"
            f"Papers to screen:\n{_format_batch(batch, start)}"
        )
        response = invoke_llm(
            llm,
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=human),
            ],
        )
        try:
            parsed = extract_json(response.content)
            decisions = parsed.get("decisions") or []
        except (json.JSONDecodeError, ValueError, TypeError):
            for paper in batch:
                rejected.append({**paper, "screen_reason": "Parse failure — excluded"})
            continue

        decision_map = {
            int(d["index"]): d
            for d in decisions
            if isinstance(d, dict) and "index" in d
        }

        for i, paper in enumerate(batch):
            global_index = start + i
            decision = decision_map.get(global_index) or decision_map.get(i) or {}
            label = str(decision.get("decision", "EXCLUDE")).upper()
            reason = str(decision.get("reason", "No reason provided")).strip()
            annotated = {**paper, "screen_reason": reason}
            if label == "INCLUDE":
                screened.append(annotated)
            else:
                rejected.append(annotated)

        time.sleep(1.0)

    return {"screened_papers": screened, "rejected_papers": rejected}
