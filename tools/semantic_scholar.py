"""Thin Semantic Scholar Academic Graph API client."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = "paperId,title,abstract,year,authors,citationCount,fieldsOfStudy"
RATE_LIMIT_SECONDS = 1.1


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    api_key = os.getenv("S2_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """Perform an HTTP request with rate limiting and retries on 429/5xx."""
    url = f"{BASE_URL}{path}"
    last_error: Exception | None = None
    for attempt in range(5):
        time.sleep(RATE_LIMIT_SECONDS * (attempt + 1))
        response = requests.request(method, url, headers=_headers(), timeout=60, **kwargs)
        if response.status_code in (429, 500, 502, 503):
            last_error = requests.HTTPError(
                f"{response.status_code} for {url}", response=response
            )
            continue
        response.raise_for_status()
        return response.json()
    assert last_error is not None
    raise last_error


def search_papers(query: str, limit: int = 50) -> list[dict[str, Any]]:
    """Search papers via /paper/search/bulk and return up to `limit` results.

    Bulk returns up to 1,000 matches per call and is more reliable without an
    API key than the relevance endpoint. Results are truncated to `limit`.
    """
    params = {
        "query": query,
        "fields": DEFAULT_FIELDS,
    }
    data = _request("GET", "/paper/search/bulk", params=params)
    papers = data.get("data") or []
    return papers[:limit]
