---
title: VetVision AI
emoji: 🐾
colorFrom: teal
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# VetVision AI — Unified Backend

Production-ready FastAPI backend for the VetVision veterinary platform.
Unifies two previously separate projects into one modular, extensible service.

## Agents

For full details on agent workflows, tools, and decision engines, see the [Agents Documentation](AGENTS.md).

| Agent | Route | Audience |
|---|---|---|
| **User Agent (فيتو)** | `POST /chat` | Pet owners |
| **Vet Copilot** | `POST /copilot/chat` | Veterinary doctors |

## Architecture

```
main.py
├── routers/user_chat.py      → /chat, /health
└── routers/copilot.py        → /copilot/*

agents/
├── user_agent/               → LangGraph ReAct agent (OpenRouter + Gemini fallback)
│   ├── graph.py
│   ├── tools.py              → vet_rag_search, tavily_search
│   └── prompts.py
└── copilot_agent/
    ├── graph/                → LangGraph agent (router → tools)
    └── tools/                → 6 tools including patient DB + PDF reports

rag/                          → Shared Advanced RAG pipeline (Qdrant + Jina + Cohere)
db/                           → SQLite patient records (SQLAlchemy)
scripts/                      → Indexing utilities
data/                         → Runtime data (Qdrant DB, SQLite, PDFs, templates, fonts)
```

## Setup

### 1. Prerequisites

```bash
pip install -r requirements.txt
playwright install chromium    # Required for PDF report generation
```

### 2. Environment Variables

```bash
cp .env.example .env
# Fill in all required keys in .env
```

Required keys: `GROQ_API_KEY`, `JINA_API_KEY`, `COHERE_API_KEY`, `TAVILY_API_KEY`,
`OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, `QDRANT_CLOUD_URL`, `QDRANT_CLOUD_API_KEY`

### 3. Run Locally

```bash
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | User agent streaming chat (SSE) |
| `GET` | `/health` | User agent health check |
| `POST` | `/copilot/chat` | Vet copilot streaming chat (SSE) |
| `GET` | `/copilot/patient/{id}` | Get patient history by Animal ID |
| `POST` | `/copilot/generate-report` | Generate PDF report directly |
| `GET` | `/copilot/reports/{filename}` | Download generated PDF |
| `GET` | `/copilot/health` | Vet copilot health check |

## Indexing (one-time setup)

```bash
python scripts/build_index.py   # Index all markdown sources into Qdrant
python scripts/add_pdf.py <path> cat   # Add a new PDF
```

## Deployment

```bash
# Render / Heroku
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Make sure to run `playwright install chromium` in your build command.
