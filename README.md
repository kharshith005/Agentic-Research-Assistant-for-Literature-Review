# Agentic Research Assistant for Literature Review

Turns a research question into a filtered literature review matrix using three LangGraph agents and the Semantic Scholar API.

1. **Query Agent** — generates search queries and fetches papers  
2. **Screening Agent** — keeps or discards papers from title/abstract  
3. **Synthesis Agent** — builds a comparison matrix  

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your `GOOGLE_API_KEY` in `.env` ([get one here](https://aistudio.google.com/apikey)).

## Run

```bash
python main.py "Graph neural networks applied to healthcare data after 2020"
```

Result: `output/review_matrix.md`
