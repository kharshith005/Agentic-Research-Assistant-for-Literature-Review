"""CLI entry point for the agentic literature review pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from graph import build_graph

load_dotenv()

console = Console()
OUTPUT_FILE = Path(__file__).resolve().parent / "output" / "review_matrix.md"


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        console.print(
            "[red]Missing GOOGLE_API_KEY.[/red] "
            "Copy .env.example to .env and add your Gemini key."
        )
        sys.exit(1)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = console.input("[bold]Enter your research question:[/bold] ").strip()

    if not question:
        console.print("[red]A research question is required.[/red]")
        sys.exit(1)

    console.print(
        Panel.fit(question, title="Research Question", border_style="cyan")
    )

    app = build_graph()
    merged: dict = {"research_question": question}

    console.print(
        "\n[bold cyan][Query Agent][/bold cyan] Generating queries and fetching papers..."
    )

    for event in app.stream(merged, stream_mode="updates"):
        for node_name, update in event.items():
            merged.update(update)

            if node_name == "query_agent":
                queries = update.get("search_queries") or []
                papers = update.get("raw_papers") or []
                console.print(
                    f"[bold cyan][Query Agent][/bold cyan] "
                    f"Generated {len(queries)} search queries:"
                )
                for q in queries:
                    console.print(f"  • {q}")
                console.print(
                    f"[bold cyan][Query Agent][/bold cyan] "
                    f"Found {len(papers)} unique papers from Semantic Scholar"
                )
                console.print(
                    "\n[bold yellow][Screening Agent][/bold yellow] "
                    "Evaluating papers against criteria..."
                )

            elif node_name == "screening_agent":
                passed = update.get("screened_papers") or []
                rejected = update.get("rejected_papers") or []
                console.print(
                    f"[bold yellow][Screening Agent][/bold yellow] "
                    f"{len(passed)} papers passed, {len(rejected)} rejected"
                )
                console.print(
                    "\n[bold green][Synthesis Agent][/bold green] "
                    "Building review matrix..."
                )

            elif node_name == "synthesis_agent":
                console.print(
                    f"[bold green][Synthesis Agent][/bold green] "
                    f"Matrix saved to {OUTPUT_FILE}"
                )

    raw = merged.get("raw_papers") or []
    passed = merged.get("screened_papers") or []
    console.print()
    console.print(
        Panel.fit(
            f"Found {len(raw)} papers · {len(passed)} passed screening\n"
            f"Saved → {OUTPUT_FILE}",
            title="Done",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
