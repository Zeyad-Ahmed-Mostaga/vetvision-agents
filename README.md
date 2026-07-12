<div align="center">

#   VetVision AI Agnets 

**Production-grade, multi-agent AI platform for veterinary care in Egypt.**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-purple)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

*Two AI agents. One backend. Built for Egyptian veterinary care.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Agents](#agents)
  - [VITO — User Agent](#vito--user-agent)
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
- [Author](#author)

---

## Overview

VetVision is a bilingual (Arabic/English) AI veterinary platform built for the Egyptian market. It unifies two previously separate projects into a single, modular FastAPI service (`v2.0.0`).

**Two users. Two agents. One backend.**

| Audience | Agent | Endpoint |
|---|---|---|
| Pet owners | **VITO** — warm, bilingual conversational assistant | `POST /chat` |
| Veterinary doctors | **Vet Copilot** — clinical decision support + EHR management | `POST /copilot/chat` |

**Core capabilities:**

| Feature | Details |
|---|---|
| 🔍 Advanced RAG | 6-step retrieval over 4 veterinary knowledge bases — Qdrant + Jina + Cohere |
| 🤖 LangGraph Agents | ReAct loop (VITO) · Router→Tools loop (Vet Copilot) · Sliding-window memory |
| 🗃️ Electronic Health Records | SQLite patient registry · 6-char Animal IDs · per-visit weight tracking · audit-safe writes |
| 📄 Bilingual PDF Reports | Arabic-RTL + English via Jinja2 → Playwright/Chromium |
| 📡 SSE Streaming | Token-by-token response delivery for both agents |
| 🔁 LLM Fallback | VITO auto-switches OpenRouter → Gemini on failure |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                          │
│                             (main.py)                                │
│                                                                      │
│   ┌────────────────────────┐   ┌──────────────────────────────────┐  │
│   │   /chat  · /health     │   │   /copilot/*                     │  │
│   │   routers/user_chat    │   │   routers/copilot                │  │
│   └───────────┬────────────┘   └─────────────┬────────────────────┘  │
└───────────────┼────────────────────────────── ┼─────────────────────┘
                │                               │
                ▼                               ▼
   ┌────────────────────────┐   ┌───────────────────────────────────┐
   │    VITO — User Agent   │   │        Vet Copilot                │
   │    LangGraph ReAct     │   │   LangGraph Router → Tools        │
   │   ─────────────────    │   │   ──────────────────────────────  │
   │   Tools:               │   │   Tools:                          │
   │   · vet_rag_search     │   │   · vet_rag_search                │
   │   · tavily_search      │   │   · tavily_search                 │
   │                        │   │   · register_first_visit  [WRITE] │
   │   Primary:  OpenRouter │   │   · log_returning_visit   [WRITE] │
   │   Fallback: Gemini     │   │   · get_patient_history   [READ]  │
   │                        │   │   · generate_patient_report       │
   └───────────┬────────────┘   └─────────────┬─────────────────────┘
               │                              │
               └──────────────┬───────────────┘
                              │
                              ▼
          ┌────────────────────────────────────┐
          │          Shared RAG Module         │
          │      rag/retrieval.py              │
          │   ─────────────────────────────    │
          │   1. Query Enhancement (OpenRouter) │
          │   2. HyDE Generation   (OpenRouter) │
          │   3. Metadata Filter   (Qdrant)     │
          │   4. Dual Retrieval    (Jina+Qdrant)│
          │   5. Deduplication                 │
          │   6. Cohere Reranking              │
          └────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **Shared RAG, separate agents** | Both agents call `advanced_rag_retrieve()` — zero duplication, independent behavior |
| **Singleton pattern** | LLMs, vector stores, and compiled graphs initialized once at startup and reused |
| **Central config** | All parameters in `config.py:Settings`, overridable via environment variables |
| **Audit-safe writes** | Patient record tools marked `[WRITE]` — LLM instructed never to write without explicit doctor confirmation |

---

## Agents

### VITO — User Agent

**Location:** `agents/user_agent/`

A friendly, bilingual AI assistant for pet owners. Responds in the user's language (Egyptian Arabic, English, or MSA). Built on a LangGraph **ReAct** loop with automatic LLM fallback.

#### Decision Engine

| Mode | Trigger | Action |
|---|---|---|
| **General Conversation** | Greeting / thanks | Respond warmly — no tools used |
| **Missing Info Gateway** | Medical Q without animal type · location Q without area · drug Q without name | Ask for the missing detail — never assume |
| **Ready to Search** | All required info present | `vet_rag_search` for medical/diet · `tavily_search` for clinics/drugs |
| **RAG Self-Correction** | RAG returns empty or irrelevant results | Auto-fallback to `tavily_search` |

#### Tools

| Tool | Primary Use | Notes |
|---|---|---|
| `vet_rag_search` | Medical symptoms, diet, toxins, behavior, general care | Query must be in English · `animal_type` confirmed before calling |
| `tavily_search` | Clinics, drug prices, real-world data · RAG fallback | Egypt-specific queries formulated in Arabic |

#### LLM Strategy

```
Primary  →  OpenRouter  (DeepSeek Chat V3 by default — configurable)
Fallback →  Google Gemini 2.0 Flash  (automatic, transparent to the graph)
```

Both LLMs are bound to the same tool list. On any OpenRouter exception, the node silently retries with Gemini — the user never sees the failure.

---

### Vet Copilot — Doctor Agent

**Location:** `agents/copilot_agent/`

A clinical-grade AI assistant for veterinary doctors. Handles patient intake, visit logging, medical knowledge lookup, and bilingual PDF report generation. Built on a LangGraph **router→tools** loop.

#### Capabilities

| # | Capability | Tool | Access |
|---|---|---|---|
| 1 | Medical Knowledge Lookup | `vet_rag_search` | Shared RAG pipeline |
| 2 | Web Search | `tavily_search` | Drug availability, latest protocols |
| 3 | Patient Registration | `register_first_visit` | **WRITE** — creates patient + first visit, returns 6-char Animal ID |
| 4 | Visit Logging | `log_returning_visit` | **WRITE** — appends visit to existing patient |
| 5 | History Retrieval | `get_patient_history` | **READ-ONLY** — safe for any lookup |
| 6 | PDF Report Generation | `generate_patient_report` | Triggers 3-phase report pipeline |

#### Write Safety Protocol

The copilot enforces strict rules at the system prompt level:

- Write tools are **only invoked** when the doctor uses explicit intent — e.g., *"Register this patient"*, *"Save this visit"*
- Read-only requests (history lookups, questions, summaries) **never** trigger a database write
- Rule of thumb baked into the agent: **when in doubt, ask — never write**

#### Medical Text Formalization

Before any write or report generation, the agent automatically formalizes input:
- Converts informal/Arabic text into structured, bullet-formatted clinical English
- Applies standard veterinary terminology (e.g., `"Feline Upper Respiratory Infection (Herpesvirus)"`)
- Includes dosage, frequency, and duration in treatment records

#### Patient Visit Flow

```
Doctor sends message
       │
       ▼
New conversation? ──YES──► Ask for doctor's name  [MANDATORY — nothing else happens first]
       │
       ▼  (name known from history)
First-time patient?
  ├── YES ──► Collect: animal name · type · owner · diagnosis · treatment · date · weight? · notes?
  │            ──► register_first_visit()  ──►  Return Animal ID  (e.g. "A3X7K9")
  └── NO  ──► Confirm existing 6-char Animal ID
               ──► log_returning_visit()   ──►  Confirm visit logged
```

---

## Advanced RAG Pipeline

**Location:** `rag/retrieval.py`

Both agents share the same 6-step retrieval pipeline. Every step includes a fault-tolerant fallback — a single service failure never breaks the chain.

```
Input: User question (any language)
       │
       ▼
┌─ Step 1 · Query Enhancement ────────────────────────────────────────┐
│  LLM translates and rewrites the query into clinical English        │
│  Fault: LLM fails → use raw original question                      │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─ Step 2 · HyDE — Hypothetical Document Embedding ───────────────────┐
│  LLM generates a short hypothetical veterinary passage              │
│  Embedded alongside the query for richer semantic coverage          │
│  Fault: LLM fails → skip HyDE leg entirely                         │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─ Step 3 · Metadata Filter ──────────────────────────────────────────┐
│  Qdrant filter: animal_type == { cat | dog | horse | other }       │
│  Eliminates cross-species retrieval contamination                  │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─ Step 4 · Dual Retrieval  (k = 8 per leg) ──────────────────────────┐
│  similarity_search(enhanced_query)  +  similarity_search(HyDE)     │
│  Embeddings: Jina jina-embeddings-v3  (1024-dim)                   │
│  Fault: retry ×3 with backoff → fallback to on-disk Qdrant         │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─ Step 5 · Deduplication ────────────────────────────────────────────┐
│  Dedup by first 200 chars of page_content                          │
│  Up to 16 raw candidates merged into a unique set                  │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─ Step 6 · Cohere Reranking ─────────────────────────────────────────┐
│  CohereRerank (rerank-v3.5) → top 3 documents                      │
│  Fault: Cohere fails → return top 3 from merged list (unranked)    │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
Output: Top 3 reranked document chunks → agent tool response
```

### Knowledge Base

| Key | Source | Species |
|---|---|---|
| `cat` | Cat veterinary guide | Cats & kittens |
| `dog` | Dog veterinary guide | Dogs & puppies |
| `horse` | Horse veterinary guide | Horses & foals |
| `other` | General animal owners guide | All other species |

**Chunking:** `chunk_size=1250` · `chunk_overlap=250` · `min_chunk_length=100`  
**Metadata per chunk:** `animal_type` · `source` · `Topic` · `SubTopic` · `chunk_index`

---

## PDF Report Generation

**Location:** `agents/copilot_agent/tools/report/`

Reports are generated through a 3-phase pipeline:

```
Phase 1 — Research  (no LLM calls)
  ├── Parallel RAG queries: diagnosis query + treatment query
  └── Tavily web search fallback if RAG returns < 300 characters

Phase 2 — Generate  (single LLM call with structured output)
  ├── Input:  patient data + RAG context
  ├── Schema: ReportContent (Pydantic)
  │     ├── arabic_summary          (RTL narrative)
  │     ├── english_summary
  │     ├── treatment_plan          (bullet list)
  │     ├── follow_up_notes
  │     └── home_care_instructions
  └── Retry: up to 3 attempts with exponential backoff

Phase 3 — Render PDF
  ├── Jinja2 renders the HTML report template
  ├── Playwright/Chromium converts HTML → PDF
  │     (native Arabic RTL · full CSS3 · Unicode fonts)
  └── Output saved to data/reports/<uuid>.pdf
```

> Reports are fully bilingual — Arabic (RTL) and English — rendered via Chromium for complete Unicode and RTL support.

---

## Data Model

**Location:** `db/models.py` · `db/crud.py`

Two SQLAlchemy tables backed by SQLite:

```
Patient
─────────────────────────────────────────────────────
animal_id    PK   VARCHAR(6)     6-char alphanumeric  e.g. "A3X7K9"
animal_name       VARCHAR(200)
animal_type       VARCHAR(100)   free text as the doctor writes it
owner_name        VARCHAR(200)
doctor_name       VARCHAR(200)   attending vet (free text, no FK)
created_at        DATETIME


Visit  (many-to-one → Patient)
─────────────────────────────────────────────────────
visit_id     PK   VARCHAR(36)    UUID
animal_id    FK   → Patient.animal_id
diagnosis         TEXT           formalized clinical English
treatment         TEXT           formalized, includes dosage/frequency
doctor_notes      TEXT           nullable — observations, flags
weight_kg         FLOAT          nullable — tracked per visit (not per patient)
visit_date        DATE
doctor_name       VARCHAR(200)   may differ from Patient.doctor_name
created_at        DATETIME
```

> **Animal ID:** 6-char uppercase alphanumeric (A–Z, 0–9) — short enough for verbal communication at a clinic. Generated randomly with collision checking.

---

## API Reference

### VITO — User Agent Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Stream SSE response from VITO |
| `GET` | `/health` | Health check + Qdrant status |

**Request:**
```json
POST /chat
{ "message": "My cat has been vomiting since yesterday, what could it be?" }
```

**SSE Response Stream:**
```
data: {"type": "token",  "content": "This could be caused by..."}
data: {"type": "token",  "content": " several factors..."}
data: {"type": "done",   "thread_id": "<id>"}
data: {"type": "error",  "content": "<message>"}   ← only on failure
```

---

### Vet Copilot Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/copilot/chat` | Stream SSE response from Vet Copilot |
| `GET` | `/copilot/patient/{animal_id}` | Retrieve full patient history by Animal ID |
| `POST` | `/copilot/generate-report` | Generate PDF medical report (bypass agent) |
| `GET` | `/copilot/reports/{filename}` | Download a generated PDF |
| `GET` | `/copilot/health` | Health check + Qdrant status |

**Direct report generation:**
```json
POST /copilot/generate-report
{
  "animal_name":  "Luna",
  "animal_type":  "cat",
  "owner_name":   "Ahmed Hassan",
  "weight_kg":    4.2,
  "diagnosis":    "Feline Upper Respiratory Infection",
  "treatment":    "Amoxicillin 50mg twice daily for 7 days",
  "doctor_name":  "Dr. Zeyad",
  "doctor_notes": "Monitor for conjunctivitis at next visit"
}
```

> Interactive API docs: `http://localhost:8000/docs`

---

## Project Structure

```
vetvision-ai/
│
├── main.py                          # FastAPI app entry point — lifespan, routers, CORS
├── config.py                        # Central Settings dataclass (single source of truth)
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker / HF Spaces deployment
├── Procfile                         # Render / Heroku deploy
├── .env.example                     # Full environment variable reference
│
├── agents/
│   ├── user_agent/                  # VITO — pet owner chatbot
│   │   ├── graph.py                 # LangGraph ReAct agent (OpenRouter + Gemini fallback)
│   │   ├── tools.py                 # vet_rag_search · tavily_search
│   │   └── prompts.py               # System prompt (bilingual · Egyptian personality)
│   │
│   └── copilot_agent/               # Vet Copilot — doctor agent
│       ├── graph/
│       │   ├── builder.py           # Graph compilation (router → tools → router → ...)
│       │   ├── nodes.py             # router_node — LLM call + system prompt injection
│       │   ├── edges.py             # should_continue — conditional routing logic
│       │   └── state.py             # CopilotState TypedDict
│       └── tools/
│           ├── __init__.py          # ALL_TOOLS export list
│           ├── patient_records.py   # register_first_visit · log_returning_visit · get_patient_history
│           ├── vet_rag.py           # vet_rag_search (wraps shared RAG)
│           ├── web_search.py        # tavily_search
│           └── report/
│               ├── pipeline.py      # 3-phase pipeline: Research → Generate → Render
│               ├── tool.py          # generate_patient_report LangChain tool wrapper
│               └── schemas.py       # ReportContent · ReportData Pydantic schemas
│
├── rag/
│   ├── retrieval.py                 # 6-step Advanced RAG — public API: advanced_rag_retrieve()
│   ├── store.py                     # Qdrant vector store (cloud primary · local fallback)
│   ├── embeddings.py                # Jina jina-embeddings-v3 (1024-dim)
│   └── chunking.py                  # Markdown chunking utilities
│
├── db/
│   ├── models.py                    # SQLAlchemy ORM: Patient + Visit tables
│   └── crud.py                      # register_new_patient · add_visit_to_existing · format_patient_history
│
├── routers/
│   ├── user_chat.py                 # /chat · /health
│   └── copilot.py                   # /copilot/*
│
├── scripts/
│   ├── build_index.py               # One-time: index all Markdown sources into Qdrant
│   └── add_pdf.py                   # Add a single PDF to the knowledge base
│
└── data/                            # Runtime data (git-ignored except templates & fonts)
    ├── qdrant_db/                   # On-disk Qdrant (local / fallback mode)
    ├── patients.db                  # SQLite patient records
    ├── reports/                     # Generated PDF reports
    ├── templates/                   # report.html (Jinja2 template)
    └── fonts/                       # Arabic fonts (Amiri-Regular.ttf · Amiri-Bold.ttf)
```

---

## Setup & Local Development

### Prerequisites

- Python 3.10+
- Qdrant instance (cloud or local) with the `vetvision` collection indexed

### 1. Install Dependencies

```bash
pip install -r requirements.txt
playwright install chromium     # Required for PDF report generation
```

### 2. Configure Environment

```bash
cp .env.example .env
# Open .env and fill in all required API keys
```

### 3. Run the Server

```bash
uvicorn main:app --reload --port 8000
```

| URL | Purpose |
|---|---|
| `http://localhost:8000/docs` | Swagger UI |
| `http://localhost:8000/redoc` | ReDoc |
| `http://localhost:8000/` | Root — service info JSON |

**On startup, the app automatically:**
1. Connects to Qdrant and logs collection point counts
2. Compiles both LangGraph agents as in-memory singletons
3. Creates `data/reports/`, `data/templates/`, `data/fonts/` if missing

---

## Deployment

### Docker / Hugging Face Spaces

The `Dockerfile` is production-ready for HF Spaces CPU Basic:

- Installs all Chromium system dependencies as `root`
- Switches to non-root user (UID 1000 — HF Spaces requirement)
- Exposes port `7860`

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

> Set all secrets as HF Spaces **Secrets** — never hardcode them in the Dockerfile.

### Render / Heroku

```
# Procfile
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Add `playwright install chromium` to your build command and configure all environment variables in your platform dashboard.

---

## Environment Variables

### Required

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Utility LLM — query enhancement, HyDE, report content generation |
| `JINA_API_KEY` | Embeddings — `jina-embeddings-v3` (1024-dim) |
| `COHERE_API_KEY` | Reranking — `rerank-v3.5` |
| `TAVILY_API_KEY` | Web search — both agents |
| `OPENROUTER_API_KEY` | Primary agent LLM — both agents |
| `GOOGLE_API_KEY` | Fallback LLM — VITO only (Gemini 2.0 Flash) |

### Qdrant — At Least One Mode Required

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_MODE` | `hybrid` | `cloud` · `local` · `hybrid` (cloud first, local fallback) |
| `QDRANT_CLOUD_URL` | — | Cloud cluster URL |
| `QDRANT_CLOUD_API_KEY` | — | Cloud API key |
| `QDRANT_LOCAL_PATH` | `./data/qdrant_db` | On-disk Qdrant path |
| `COLLECTION_NAME` | `vetvision` | Qdrant collection name |

### Optional Overrides

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_MODEL` | `deepseek/deepseek-chat-v3-0324:free` | Primary LLM for both agents |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Fallback LLM (VITO only) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Utility LLM model |
| `SQLITE_DB_PATH` | `./data/patients.db` | Patient records database path |
| `REPORTS_DIR` | `./data/reports` | PDF output directory |
| `REPORT_TEMPLATES_DIR` | `./data/templates` | Jinja2 HTML template directory |
| `REPORT_FONTS_DIR` | `./data/fonts` | Arabic font directory |
| `LANGSMITH_API_KEY` | *(disabled)* | LangSmith observability tracing |
| `LANGSMITH_PROJECT` | `VetVision-Unified` | LangSmith project name |

---

## Indexing the Knowledge Base

Run once to populate the Qdrant vector store before starting the server:

```bash
# Index all four Markdown sources (cat · dog · horse · other)
python scripts/build_index.py

# Add a single PDF to an existing collection
python scripts/add_pdf.py <path/to/file.pdf> cat
```

The indexer:
1. Loads the Markdown source for the given animal type
2. Chunks it — `chunk_size=1250` · `chunk_overlap=250`
3. Embeds each chunk with Jina (`jina-embeddings-v3`)
4. Upserts into Qdrant with metadata: `animal_type` · `source` · `Topic` · `SubTopic` · `chunk_index`
5. Batches uploads (`batch_size=200`) with rate-limit sleep between batches

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
