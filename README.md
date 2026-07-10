---
title: VetVision AI
emoji: 🐾
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🐾 VetVision AI — Unified Backend

**Production-grade, multi-agent AI platform for veterinary care in Egypt.**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-purple)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-HF_Spaces-blue?logo=docker)](https://huggingface.co/docs/hub/spaces-sdks-docker)

*Serving two distinct AI agents — one for pet owners, one for veterinary doctors — on a single FastAPI backend with shared RAG infrastructure.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Agents](#agents)
  - [فيتو — User Agent (Pet Owner Chatbot)](#فيتو--user-agent-pet-owner-chatbot)
  - [Vet Copilot — Doctor Agent](#vet-copilot--doctor-agent)
- [Advanced RAG Pipeline](#advanced-rag-pipeline)
- [PDF Report Generation](#pdf-report-generation)
- [Data Model](#data-model)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Setup & Local Development](#setup--local-development)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)
- [Indexing the Knowledge Base](#indexing-the-knowledge-base)

---

## Overview

VetVision is a bilingual (Arabic/English) AI veterinary platform built for the Egyptian market. It consolidates what were originally two separate projects into a single, modular FastAPI service (`v2.0.0`).

**Two users, two agents, one backend:**

| Audience | Agent | Route |
|---|---|---|
| Pet owners | **فيتو (Vito)** — warm, conversational assistant | `POST /chat` |
| Veterinary doctors | **Vet Copilot** — clinical decision support + EHR | `POST /copilot/chat` |

**Core capabilities:**

- 🔍 **Advanced RAG** — 6-step retrieval pipeline over 4 veterinary knowledge bases (Qdrant + Jina embeddings + Cohere reranking)
- 🤖 **LangGraph agents** — ReAct loop for فيتو; router→tools loop for Vet Copilot; both with sliding-window conversation memory
- 🗃️ **Electronic Health Records** — SQLite-backed patient registry with 6-char Animal IDs, per-visit weight tracking, and audit-safe write rules
- 📄 **Bilingual PDF reports** — Arabic-RTL + English medical reports via Jinja2 → Playwright/Chromium
- 📡 **SSE streaming** — token-by-token response delivery for both agents
- 🔁 **LLM fallback** — فيتو automatically switches from OpenRouter → Gemini on failure; all agents include graceful error recovery

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                          │
│                           (main.py)                                 │
│                                                                     │
│   ┌───────────────────────┐   ┌─────────────────────────────────┐  │
│   │   /chat, /health      │   │  /copilot/*                     │  │
│   │   routers/user_chat   │   │  routers/copilot                │  │
│   └──────────┬────────────┘   └───────────────┬─────────────────┘  │
└──────────────┼────────────────────────────────┼────────────────────┘
               │                                │
               ▼                                ▼
  ┌────────────────────────┐    ┌───────────────────────────────────┐
  │   User Agent (فيتو)    │    │         Vet Copilot               │
  │   LangGraph ReAct      │    │   LangGraph Router → Tools        │
  │   ─────────────────    │    │   ─────────────────────────────   │
  │   Tools:               │    │   Tools:                          │
  │   • vet_rag_search     │    │   • vet_rag_search                │
  │   • tavily_search      │    │   • tavily_search                 │
  │                        │    │   • register_first_visit (WRITE)  │
  │   Primary LLM:         │    │   • log_returning_visit  (WRITE)  │
  │     OpenRouter         │    │   • get_patient_history  (READ)   │
  │   Fallback:            │    │   • generate_patient_report       │
  │     Gemini 2.0 Flash   │    │                                   │
  └──────────┬─────────────┘    └───────────────┬───────────────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            │
                            ▼
         ┌─────────────────────────────────────┐
         │         Shared RAG Module           │
         │   rag/retrieval.py (6-step pipeline)│
         │   ─────────────────────────────     │
         │   1. Query Enhancement  (OpenRouter) │
         │   2. HyDE Generation    (OpenRouter) │
         │   3. Metadata Filter    (Qdrant)     │
         │   4. Dual Retrieval     (Jina+Qdrant)│
         │   5. Deduplication                  │
         │   6. Cohere Reranking               │
         └─────────────────────────────────────┘
```

### Key Design Decisions

- **Shared RAG, separate agents**: Both agents call the same `advanced_rag_retrieve()` function, avoiding code duplication while keeping agent behavior fully independent.
- **Singleton pattern everywhere**: LLMs, vector stores, compiled graphs — all initialized once at startup via lazy singletons, then reused.
- **Config as single source of truth**: All tunable parameters (model names, chunk sizes, retrieval k, reranking top-n, etc.) live in `config.py:Settings` and are overridable via environment variables.
- **Audit-safe writes**: The Vet Copilot's patient record tools carry explicit `WRITE` markers in their docstrings and system prompt rules — the LLM is instructed never to write without explicit doctor confirmation.

---

## Agents

### فيتو — User Agent (Pet Owner Chatbot)

**File**: `agents/user_agent/`

A friendly, bilingual AI assistant for pet owners. Responds in the same language the user writes in (Egyptian Arabic, English, or MSA). Built on a LangGraph **ReAct** loop.

#### Decision Engine (from system prompt)

| Mode | Trigger | Action |
|---|---|---|
| General conversation | Greeting / thanks | Respond warmly, no tools |
| Missing info gateway | Medical Q without animal type; location Q without area; drug Q without name | Ask for the missing detail — never guess |
| Ready to search | All info present | `vet_rag_search` for medical/diet; `tavily_search` for clinics/drugs |
| RAG self-correction | RAG returns empty or irrelevant | Auto-fallback to `tavily_search` |

#### Tools

| Tool | When Used | Notes |
|---|---|---|
| `vet_rag_search` | Medical symptoms, diet, toxins, behavior, general care | Question **must** be in English; `animal_type` must be confirmed before calling |
| `tavily_search` | Clinics, drug prices, real-world data; RAG fallback | Egypt-specific queries formulated in Arabic |

#### LLM Strategy

```
Primary:  OpenRouter (DeepSeek Chat V3 by default, configurable)
Fallback: Google Gemini 2.0 Flash (automatic — transparent to the graph)
```

Both LLMs are bound to the same tool list. On any OpenRouter exception, the node retries with Gemini without surfacing the error to the user.

---

### Vet Copilot — Doctor Agent

**Files**: `agents/copilot_agent/`

A clinical-grade AI assistant for veterinary doctors. Manages patient intake, visit logging, medical knowledge lookup, and PDF report generation. Built on a LangGraph **router→tools** loop.

#### Capabilities

1. **Medical Knowledge Lookup** — `vet_rag_search` over the same shared knowledge base
2. **Web Search** — `tavily_search` for drug availability, latest protocols
3. **Patient Registration** — `register_first_visit` (WRITE) creates new patient + first visit, returns a unique 6-char Animal ID
4. **Visit Logging** — `log_returning_visit` (WRITE) appends to existing patient record
5. **History Retrieval** — `get_patient_history` (READ-ONLY) by Animal ID
6. **PDF Reports** — `generate_patient_report` triggers the 3-phase report pipeline

#### Write Safety Protocol

The copilot enforces strict write-safety rules in its system prompt:

- `register_first_visit` and `log_returning_visit` are **only called** when the doctor uses explicit save/register intent. Questions, summaries, and lookups **never** trigger a write.
- The system prompt uses `WRITE` annotations in tool usage rules to reinforce this.
- When in doubt, the agent is instructed: **ask, never write**.

#### Medical Text Formalization

Before any database write or report generation, the agent automatically:
- Converts informal/Arabic input into structured, bullet-formatted clinical English
- Applies formal veterinary terminology (e.g., `"Feline Upper Respiratory Infection (Herpesvirus)"` not `"cat has cold"`)
- Includes dosage/frequency/duration in treatment records

#### Patient Visit Flow

```
Doctor sends message
        │
        ▼
First turn? ──YES──► Ask for doctor's name (MANDATORY)
        │
        ▼ (name known)
First time patient?
   ├─ YES ──► Collect: name, type, owner, diagnosis, treatment, date, weight?, notes?
   │           ──► register_first_visit() ──► Return Animal ID (e.g. "A3X7K9")
   └─ NO  ──► Ask for existing Animal ID
               ──► log_returning_visit() ──► Confirm logged
```

---

## Advanced RAG Pipeline

**File**: `rag/retrieval.py`

Both agents use the same 6-step pipeline. Each step has fault-tolerant fallbacks so a single service failure never breaks the retrieval.

```
User Question (any language)
         │
         ▼
┌─ Step 1: Query Enhancement ─────────────────────────────────────────┐
│  OpenRouter LLM: translate + rewrite into clinical English          │
│  Fault: if LLM fails → use raw question                            │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Step 2: HyDE (Hypothetical Document Embeddings) ───────────────────┐
│  OpenRouter LLM: generate a 3-5 sentence hypothetical answer        │
│  This passage is embedded alongside the query for richer retrieval  │
│  Fault: if LLM fails → skip HyDE leg entirely                      │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Step 3: Metadata Filter ───────────────────────────────────────────┐
│  Qdrant filter: animal_type == {cat|dog|horse|other}               │
│  Prevents cross-species retrieval contamination                    │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Step 4: Dual Retrieval (k=8 per leg) ──────────────────────────────┐
│  similarity_search(enhanced_query) + similarity_search(HyDE)       │
│  Embeddings: Jina jina-embeddings-v3 (1024-dim)                    │
│  Fault: retry x3 with backoff → fallback to on-disk Qdrant         │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Step 5: Deduplication ─────────────────────────────────────────────┐
│  Deduplicate by first 200 chars of page_content                    │
│  Merges up to 16 raw candidates into unique set                    │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Step 6: Cohere Reranking ──────────────────────────────────────────┐
│  CohereRerank (rerank-v3.5) → top 3 documents                      │
│  Fault: if Cohere fails → return top 3 from merged list (unranked) │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
  Top 3 ranked Document chunks → Agent tool response
```

### Knowledge Base

The vector store (`vetvision` collection in Qdrant) is indexed from pre-parsed Markdown sources:

| Collection Key | Source |
|---|---|
| `cat` | Cat veterinary guide |
| `dog` | Dog veterinary guide |
| `horse` | Horse veterinary guide |
| `other` | General animal owners guide |

Each chunk stores `animal_type`, `source`, `Topic`, `SubTopic`, and `chunk_index` as metadata.

**Chunking config**: `chunk_size=1250`, `chunk_overlap=250`, `min_chunk_length=100`

---

## PDF Report Generation

**Files**: `agents/copilot_agent/tools/report/`

The report pipeline has 3 phases:

```
Phase 1 — Research (no LLM)
  ├── Multi-query RAG: diagnosis + treatment → vet_rag_search
  └── Tavily fallback if RAG returns < 300 chars

Phase 2 — Generate (1 LLM call)
  ├── Input: patient data + RAG context
  ├── Structured output: ReportContent (Pydantic schema)
  │     ├── arabic_summary        (RTL narrative)
  │     ├── english_summary
  │     ├── treatment_plan        (bullet list)
  │     ├── follow_up_notes
  │     └── home_care_instructions
  └── LLM: OpenRouter with structured output (function_calling)
      Retry: up to 3 attempts with exponential backoff

Phase 3 — Render PDF
  ├── Jinja2: render report.html template with ReportData
  ├── Playwright/Chromium: HTML → PDF (native Arabic RTL, full CSS3)
  └── Output: saved to data/reports/<uuid>.pdf
```

The report is bilingual by design — Arabic summary (RTL) + English — rendered via Chromium for full Unicode/RTL support without any PDF library workarounds.

---

## Data Model

**Files**: `db/models.py`, `db/crud.py`

Two SQLAlchemy tables backed by SQLite:

```
Patient
──────────────────────────────────
animal_id   PK  VARCHAR(6)      e.g. "A3X7K9" (6-char alphanumeric, randomly generated)
animal_name     VARCHAR(200)
animal_type     VARCHAR(100)    free text as doctor writes it
owner_name      VARCHAR(200)
doctor_name     VARCHAR(200)    treating doctor (free text, no FK)
created_at      DATETIME

Visit  (many per Patient)
──────────────────────────────────
visit_id    PK  VARCHAR(36)     UUID
animal_id   FK  → Patient
diagnosis       TEXT            formalized clinical English
treatment       TEXT            formalized, includes dosage/frequency
doctor_notes    TEXT            nullable
weight_kg       FLOAT           nullable (tracked per visit, not per patient)
visit_date      DATE
doctor_name     VARCHAR(200)    may differ from Patient.doctor_name
created_at      DATETIME
```

**Animal ID design**: 6-char uppercase alphanumeric (A–Z + 0–9), human-readable, short enough for a clinic to communicate verbally (e.g., "A3X7K9"). Generated with collision checking.

---

## API Reference

### User Agent

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Stream SSE response from فيتو (pet owner chatbot) |
| `GET` | `/health` | User agent health + Qdrant status |

**Chat request:**
```json
{ "message": "قطتي بتتقيأ من امبارح، إيه اللي ممكن يكون؟" }
```

**SSE event format:**
```
data: {"type": "token", "content": "أهلاً،"}
data: {"type": "token", "content": " يمكن..."}
data: {"type": "done", "thread_id": "..."}
```

---

### Vet Copilot

| Method | Path | Description |
|---|---|---|
| `POST` | `/copilot/chat` | Stream SSE response from Vet Copilot (doctor agent) |
| `GET` | `/copilot/patient/{animal_id}` | Retrieve full patient history by 6-char Animal ID |
| `POST` | `/copilot/generate-report` | Generate PDF medical report directly (bypass agent) |
| `GET` | `/copilot/reports/{filename}` | Download a generated PDF report |
| `GET` | `/copilot/health` | Vet Copilot health + Qdrant status |

**Direct report generation:**
```json
{
  "animal_name": "Luna",
  "animal_type": "cat",
  "owner_name": "Ahmed Hassan",
  "weight_kg": 4.2,
  "diagnosis": "Feline Upper Respiratory Infection",
  "treatment": "Amoxicillin 50mg twice daily for 7 days",
  "doctor_name": "Dr. Zeyad",
  "doctor_notes": "Monitor for conjunctivitis development"
}
```

Full Swagger UI: `http://localhost:8000/docs`

---

## Project Structure

```
vetvision-unified/
│
├── main.py                          # FastAPI app — lifespan, routers, CORS
├── config.py                        # Central config (Settings dataclass)
├── requirements.txt
├── Dockerfile                       # HF Spaces / Docker deployment
├── Procfile                         # Render/Heroku deploy
├── .env.example                     # Full environment variable reference
│
├── agents/
│   ├── user_agent/                  # فيتو — pet owner chatbot
│   │   ├── graph.py                 # LangGraph ReAct agent (OpenRouter + Gemini fallback)
│   │   ├── tools.py                 # vet_rag_search, tavily_search
│   │   └── prompts.py               # System prompt (bilingual, Egyptian personality)
│   │
│   └── copilot_agent/               # Vet Copilot — doctor agent
│       ├── graph/
│       │   ├── builder.py           # Graph compilation (router → tools → router → ...)
│       │   ├── nodes.py             # router_node (LLM call, system prompt injection)
│       │   ├── edges.py             # should_continue (tool call routing)
│       │   └── state.py             # CopilotState (TypedDict)
│       └── tools/
│           ├── __init__.py          # ALL_TOOLS list
│           ├── patient_records.py   # register_first_visit, log_returning_visit, get_patient_history
│           ├── vet_rag.py           # vet_rag_search (wraps shared RAG)
│           ├── web_search.py        # tavily_search
│           └── report/
│               ├── pipeline.py      # 3-phase: Research → Generate → Render
│               ├── tool.py          # generate_patient_report LangChain tool wrapper
│               └── schemas.py       # ReportContent, ReportData (Pydantic)
│
├── rag/
│   ├── retrieval.py                 # 6-step Advanced RAG (public: advanced_rag_retrieve)
│   ├── store.py                     # Qdrant vector store (cloud + local fallback)
│   ├── embeddings.py                # Jina embeddings (jina-embeddings-v3, 1024-dim)
│   └── chunking.py                  # Markdown chunking utilities
│
├── db/
│   ├── models.py                    # SQLAlchemy: Patient + Visit tables
│   └── crud.py                      # register_new_patient, add_visit_to_existing, etc.
│
├── routers/
│   ├── user_chat.py                 # /chat, /health endpoints
│   └── copilot.py                   # /copilot/* endpoints
│
├── scripts/
│   ├── build_index.py               # One-time: index all Markdown sources into Qdrant
│   └── add_pdf.py                   # Add individual PDFs to the knowledge base
│
└── data/                            # Runtime data (git-ignored except templates & fonts)
    ├── qdrant_db/                   # On-disk Qdrant (local/fallback mode)
    ├── patients.db                  # SQLite patient records
    ├── reports/                     # Generated PDF reports
    ├── templates/                   # report.html (Jinja2)
    └── fonts/                       # Arabic fonts (Amiri-Regular.ttf, Amiri-Bold.ttf)
```

---

## Setup & Local Development

### Prerequisites

- Python 3.10+
- A Qdrant instance (cloud or local) with the `vetvision` collection indexed

### 1. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium    # Required for PDF report generation
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in all required API keys in .env
```

### 3. Run locally

```bash
uvicorn main:app --reload --port 8000
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

On startup, the app will:
1. Connect to Qdrant and log collection point counts
2. Compile both LangGraph agents (loaded into memory as singletons)
3. Create `data/reports/`, `data/templates/`, `data/fonts/` if missing

---

## Deployment

### Hugging Face Spaces (Docker)

The `Dockerfile` is configured for HF Spaces CPU Basic:

- Installs all Chromium system dependencies as `root`
- Switches to non-root user (UID 1000 — HF Spaces requirement)
- Exposes port `7860`

Set all environment variables as HF Spaces **Secrets** (never in the Dockerfile).

### Render / Heroku

```bash
# Procfile
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Add `playwright install chromium` to your build command. Set all required env vars in your platform dashboard.

---

## Environment Variables

### Required

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Utility LLM for query enhancement, HyDE, report content generation |
| `JINA_API_KEY` | Embeddings (`jina-embeddings-v3`, 1024-dim) |
| `COHERE_API_KEY` | Reranking (`rerank-v3.5`) |
| `TAVILY_API_KEY` | Web search (both agents) |
| `OPENROUTER_API_KEY` | Primary agent LLM (both agents) |
| `GOOGLE_API_KEY` | Fallback LLM for User Agent (Gemini 2.0 Flash) |

### Qdrant (at least one mode required)

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_MODE` | `hybrid` | `cloud`, `local`, or `hybrid` (tries cloud, falls back to local) |
| `QDRANT_CLOUD_URL` | — | Cloud cluster URL |
| `QDRANT_CLOUD_API_KEY` | — | Cloud API key |
| `QDRANT_LOCAL_PATH` | `./data/qdrant_db` | On-disk Qdrant path |
| `COLLECTION_NAME` | `vetvision` | Qdrant collection name |

### Optional Overrides

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_MODEL` | `deepseek/deepseek-chat-v3-0324:free` | Primary LLM for both agents |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Fallback LLM (User Agent only) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Utility LLM model |
| `SQLITE_DB_PATH` | `./data/patients.db` | Patient records database |
| `REPORTS_DIR` | `./data/reports` | PDF output directory |
| `REPORT_TEMPLATES_DIR` | `./data/templates` | Jinja2 HTML template directory |
| `REPORT_FONTS_DIR` | `./data/fonts` | Arabic font directory |
| `LANGSMITH_API_KEY` | *(disabled)* | LangSmith tracing (optional) |
| `LANGSMITH_PROJECT` | `VetVision-Unified` | LangSmith project name |

---

## Indexing the Knowledge Base

One-time setup to populate the Qdrant vector store:

```bash
# Index all four Markdown sources (cat, dog, horse, other)
python scripts/build_index.py

# Add a single PDF (auto-converts to Markdown via Docling, then indexes)
python scripts/add_pdf.py <path/to/file.pdf> cat
```

The indexer chunks each source (`chunk_size=1250`, `chunk_overlap=250`), embeds with Jina, and upserts into Qdrant with metadata tags (`animal_type`, `source`, `Topic`, `SubTopic`, `chunk_index`). Uploads are batched with rate-limit sleep between batches.

---

## Author

**Zeyad Ahmed** — AI Engineer

| | |
|---|---|
| 📧 Email | [ziada00700@gmail.com](mailto:ziada00700@gmail.com) |
| 💼 LinkedIn | [linkedin.com/in/zeyad-ahmed-ab9595250](https://linkedin.com/in/zeyad-ahmed-ab9595250) |
| 📱 Phone | [+20 1200249877](tel:+201200249877) |

---

<div align="center">

Built with ❤️ for Egyptian veterinary care &nbsp;·&nbsp; v2.0.0

*© 2026 Zeyad Ahmed*

</div>
