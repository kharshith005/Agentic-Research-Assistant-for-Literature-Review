"""Synthesis Agent: build a markdown literature review comparison matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm import get_llm, invoke_llm
from agents.state import ResearchState
from agents.utils import message_text

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "review_matrix.md"

SYSTEM_PROMPT = """You are an academic synthesis assistant.
Create a literature review comparison matrix as a Markdown table.
Columns (exactly): Title | Year | Methodology | Dataset | Key Findings | Relevance Score (1-5)
One row per paper. Infer Methodology, Dataset, and Key Findings from the abstract.
If information is missing, write "Not stated". Keep cell text concise.
Also add a short intro paragraph (2-3 sentences) before the table summarizing
how many papers were included and the overall theme.
Return ONLY markdown (no code fences).
"""


def _format_papers(papers: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, paper in enumerate(papers, start=1):
        authors = ", ".join(
            a.get("name", "") for a in (paper.get("authors") or [])[:5]
        )
        abstract = (paper.get("abstract") or "No abstract available.").strip()
        blocks.append(
            f"Paper {i}:\n"
            f"Title: {paper.get('title', 'Untitled')}\n"
            f"Year: {paper.get('year', 'N/A')}\n"
            f"Authors: {authors or 'N/A'}\n"
            f"Citations: {paper.get('citationCount', 'N/A')}\n"
            f"Screen reason: {paper.get('screen_reason', 'N/A')}\n"
            f"Abstract: {abstract}"
        )
    return "\n\n".join(blocks)


def synthesis_agent_node(state: ResearchState) -> dict[str, Any]:
    papers = state.get("screened_papers") or []
    question = state.get("research_question", "")

    if not papers:
        matrix = (
            f"# Literature Review Matrix\n\n"
            f"**Research question:** {question}\n\n"
            "No papers passed screening. Try broadening the question or criteria.\n"
        )
    else:
        llm = get_llm(temperature=0.2)
        response = invoke_llm(
            llm,
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Research question:\n{question}\n\n"
                        f"Included papers ({len(papers)}):\n{_format_papers(papers)}"
                    )
                ),
            ],
        )
        matrix = message_text(response.content).strip()
        if matrix.startswith("```"):
            import re

            fenced = re.search(r"```(?:markdown|md)?\s*(.*?)\s*```", matrix, re.DOTALL | re.IGNORECASE)
            if fenced:
                matrix = fenced.group(1).strip()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = f"# Literature Review Matrix\n\n**Research question:** {question}\n\n"
    if not matrix.startswith("#"):
        matrix = header + matrix
    OUTPUT_FILE.write_text(matrix, encoding="utf-8")

    return {"review_matrix": matrix}
