"""Shared LangGraph state for the literature review pipeline."""

from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    research_question: str
    inclusion_criteria: str
    search_queries: list[str]
    raw_papers: list[dict[str, Any]]
    screened_papers: list[dict[str, Any]]
    rejected_papers: list[dict[str, Any]]
    review_matrix: str
