"""
agents/copilot_agent/tools/report/pipeline.py — 2-Phase Report Pipeline
=========================================================================
Phases:
  1. Research    — Multi-query RAG retrieval + Tavily fallback (no LLM)
  2. Generate    — Single LLM call: formalizes notes AND writes all sections
  3. Render PDF  — Jinja2 HTML → Playwright/Chromium → PDF file

Phase numbering kept as 1/2/3 for log clarity; architecturally it's 2 phases
of intelligence (Research + Generate) plus a render step.
"""

import logging
import time
import uuid
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq
from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import settings
from agents.copilot_agent.tools.report.schemas import (
    ReportContent, ReportData,
)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPORTS_DIR = Path(settings.reports_dir)
_TEMPLATES_DIR = Path(settings.report_templates_dir)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM Singleton ─────────────────────────────────────────────────────────────
_llm: Optional[ChatGroq] = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0.0,
            groq_api_key=settings.groq_api_key,
        )
    return _llm


def _llm_with_retry(schema, prompt_messages: list, invoke_kwargs: dict, phase_name: str):
    """Call LLM with structured output and retry on failure."""
    llm = _get_llm().with_structured_output(schema, method="function_calling")
    from langchain_core.prompts import ChatPromptTemplate
    chain = ChatPromptTemplate.from_messages(prompt_messages) | llm

    last_exc = None
    for attempt in range(1, settings.report_llm_retries + 1):
        try:
            result = chain.invoke(invoke_kwargs)
            logger.info("[%s] LLM call succeeded on attempt %d.", phase_name, attempt)
            return result
        except Exception as exc:
            last_exc = exc
            wait = 2 ** (attempt - 1)
            logger.warning("[%s] Attempt %d/%d failed: %s — retrying in %ds...",
                           phase_name, attempt, settings.report_llm_retries, exc, wait)
            if attempt < settings.report_llm_retries:
                time.sleep(wait)

    raise RuntimeError(f"[{phase_name}] All {settings.report_llm_retries} attempts failed: {last_exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Research: Multi-query RAG + Tavily fallback (no LLM calls)
# ══════════════════════════════════════════════════════════════════════════════

def _map_animal_type(animal_type_raw: str) -> str:
    """Normalize animal type to RAG collection category."""
    cats   = {"cat","kitten","cats","قطة","قط","قطط","هرة","هريرة"}
    dogs   = {"dog","puppy","dogs","كلب","جرو","كلاب"}
    horses = {"horse","foal","horses","حصان","مهر","خيل","فرس"}
    t = animal_type_raw.lower().strip()
    if t in cats:   return "cat"
    if t in dogs:   return "dog"
    if t in horses: return "horse"
    return "other"


def _rag_retrieve(diagnosis: str, animal_type_raw: str, rag_cat: str) -> list[str]:
    """Run multi-query RAG retrieval, deduplicate chunks."""
    from rag.retrieval import advanced_rag_retrieve
    queries = [
        f"{diagnosis} in {animal_type_raw}",
        f"{diagnosis} symptoms signs {animal_type_raw}",
        f"{diagnosis} treatment home care prevention {animal_type_raw}",
    ]
    seen: set[str] = set()
    chunks: list[str] = []
    for q in queries:
        try:
            docs = advanced_rag_retrieve(question=q, animal_type=rag_cat)
            for doc in docs:
                fp = doc.page_content.strip()[:200]
                if fp not in seen:
                    seen.add(fp)
                    chunks.append(doc.page_content)
        except Exception as exc:
            logger.warning("[Phase1-Research] RAG query failed (%s): %s", q, exc)
    logger.info("[Phase1-Research] RAG returned %d unique chunks.", len(chunks))
    return chunks


def _tavily_search(query: str) -> str:
    """Run a single Tavily web search, return raw text."""
    try:
        from agents.copilot_agent.tools.web_search import tavily_search
        result = tavily_search.invoke(query)
        return str(result) if result else ""
    except Exception as exc:
        logger.warning("[Phase1-Research] Tavily search failed (%s): %s", query, exc)
        return ""


def phase_research(diagnosis: str, treatment: str, animal_type_raw: str) -> str:
    """Gather veterinary knowledge: multi-query RAG + optional Tavily fallback.

    Returns a single string of assembled source material for the LLM.
    No LLM calls — just retrieval.
    """
    rag_cat = _map_animal_type(animal_type_raw)
    logger.info("[Phase1-Research] Start | diag=%s | rag_cat=%s", diagnosis, rag_cat)

    chunks = _rag_retrieve(diagnosis, animal_type_raw, rag_cat)
    rag_text = "\n\n---\n\n".join(chunks)
    rag_total_chars = sum(len(c) for c in chunks)

    web_parts: list[str] = []
    if rag_total_chars < settings.report_rag_min_chars:
        logger.info("[Phase1-Research] RAG insufficient (%d chars) — triggering Tavily fallback.", rag_total_chars)
        sym_raw = _tavily_search(f"{diagnosis} symptoms signs {animal_type_raw} veterinary")
        care_raw = _tavily_search(f"{diagnosis} home care treatment {animal_type_raw}")
        for label, raw in [("symptoms", sym_raw), ("care", care_raw)]:
            if raw:
                web_parts.append(f"[Web — {label}]\n{raw}")
    else:
        logger.info("[Phase1-Research] RAG sufficient (%d chars) — skipping Tavily.", rag_total_chars)

    web_text = "\n\n".join(web_parts)

    # Assemble final source text
    parts: list[str] = []
    if rag_text:
        parts.append(f"=== VETERINARY KNOWLEDGE BASE ===\n{rag_text}")
    if web_text:
        parts.append(f"=== WEB SEARCH RESULTS ===\n{web_text}")

    source_label = ("rag+web" if rag_text and web_text else
                    "rag" if rag_text else
                    "web" if web_text else "none")
    sources = "\n\n".join(parts) if parts else "No source material available."

    logger.info("[Phase1-Research] Done | source=%s | rag_chars=%d | web_chars=%d",
                source_label, len(rag_text), len(web_text))
    return sources


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Generate: Single LLM call — formalize + write all sections
# ══════════════════════════════════════════════════════════════════════════════

_GENERATE_SYSTEM = """\
You are an expert veterinary medical report writer producing a professional \
Arabic-primary bilingual clinical report for a pet owner.

You will receive:
1. RAW doctor notes (diagnosis + treatment + optional notes) — these may contain \
informal language, abbreviations, or spelling errors.
2. Source material from a veterinary knowledge base and/or web search.
3. Patient context (animal name, type, weight).

Your job has TWO parts:

PART A — FORMALIZE the raw doctor notes:
- Convert the raw diagnosis into a precise medical term (English) and its \
professional Arabic equivalent.
- Convert the raw treatment into a formal treatment plan sentence (English + Arabic).
- These must read as if written by a board-certified veterinarian.

PART B — WRITE the report sections in Arabic:
For each section, follow these rules:

**overview**: Write 3-5 sentences explaining what this condition is, what causes it, \
and why it matters for this specific animal type. Use warm, simple Arabic that a Animal \
owner with no medical background can understand. Medical terms may appear in English. \
Do NOT just restate the diagnosis — EXPLAIN the condition to animal owner.

**symptoms**: Write 3-6 bullet points (use • character) of specific, observable symptoms \
the owner should watch for at home. Each bullet must describe a concrete sign, not a \
vague category. Extract ONLY from source material.

**home_care**: Write 3-5 bullet points (use • character) of practical, actionable \
home care instructions. where sources \
provide them. Extract ONLY from source material. Never invent medical advice.

**prevention**: Write 2-4 bullet points (use • character) of prevention tips for this \
condition and animal type. If sources contain no prevention info, write exactly: \
'استشر طبيبك البيطري للحصول على نصائح الوقاية المناسبة لحالة حيوانك.'

**doctor_notes**: If doctor notes are provided, rewrite them into professional, clear \
Arabic suitable for a medical report — fix spelling, formalize language, remove \
abbreviations. If no notes provided (empty), return empty string.

CRITICAL RULES:
- Write ALL Arabic sections in natural, professional Arabic — not machine-translated.
- NEVER produce single-sentence sections. Each section must have real substance.
- NEVER use generic filler like 'يرجى مراجعة الطبيب' as a substitute for content.
- Extract information from sources. When sources lack detail for a section, write \
what you know from veterinary knowledge but keep it conservative and accurate.
- The animal owner will read this report. Be warm but professional.

SOURCE MATERIAL:
{sources}
"""

_FALLBACK_CONTENT = ReportContent(
    diagnosis_en="Refer to your veterinarian",
    diagnosis_ar="يرجى مراجعة الطبيب البيطري",
    treatment_en="Follow your veterinarian's instructions",
    treatment_ar="يرجى اتباع تعليمات الطبيب البيطري",
    overview="يرجى مراجعة طبيبك البيطري للحصول على مزيد من المعلومات حول حالة حيوانك.",
    symptoms="يرجى مراجعة طبيبك البيطري للاطلاع على الأعراض التي يجب مراقبتها.",
    home_care="يرجى اتباع تعليمات طبيبك البيطري للعناية المنزلية.",
    prevention="استشر طبيبك البيطري للحصول على نصائح الوقاية المناسبة لحالة حيوانك.",
    doctor_notes="",
)


def phase_generate(
    diagnosis_raw: str,
    treatment_raw: str,
    doctor_notes_raw: str,
    animal_name: str,
    animal_type: str,
    weight_kg: float,
    sources: str,
) -> ReportContent:
    """Single LLM call: formalize raw notes AND write all Arabic report sections."""
    logger.info("[Phase2-Generate] Generating report content for %s (%s)", animal_name, animal_type)

    prompt_messages = [
        ("system", _GENERATE_SYSTEM),
        ("human",
         "PATIENT CONTEXT:\n"
         "- Animal name: {animal_name}\n"
         "- Animal type: {animal_type}\n"
         "- Weight: {weight_kg} kg\n\n"
         "RAW DOCTOR NOTES (must be formalized — do NOT copy verbatim):\n"
         "- Diagnosis: {diagnosis_raw}\n"
         "- Treatment: {treatment_raw}\n"
         "- Additional notes: {doctor_notes_raw}"),
    ]

    try:
        result = _llm_with_retry(
            schema=ReportContent,
            prompt_messages=prompt_messages,
            invoke_kwargs={
                "sources": sources,
                "animal_name": animal_name,
                "animal_type": animal_type,
                "weight_kg": weight_kg,
                "diagnosis_raw": diagnosis_raw,
                "treatment_raw": treatment_raw,
                "doctor_notes_raw": doctor_notes_raw.strip() if doctor_notes_raw else "",
            },
            phase_name="Phase2-Generate",
        )
        logger.info("[Phase2-Generate] Content generated | diag_en=%s", result.diagnosis_en)
        return result
    except Exception as exc:
        logger.error("[Phase2-Generate] Fallback triggered. Error: %s", exc)
        return _FALLBACK_CONTENT.model_copy()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — HTML → Playwright → PDF (kept exactly as-is)
# ══════════════════════════════════════════════════════════════════════════════

_playwright_instance = None
_browser_instance = None
_browser_lock = None


def _get_browser_lock():
    import threading
    global _browser_lock
    if _browser_lock is None:
        _browser_lock = threading.Lock()
    return _browser_lock


def _get_browser():
    global _playwright_instance, _browser_instance
    lock = _get_browser_lock()
    with lock:
        if _browser_instance is None or not _browser_instance.is_connected():
            logger.info("[Phase3-Render] Launching Playwright Chromium...")
            from playwright.sync_api import sync_playwright
            _playwright_instance = sync_playwright().start()
            _browser_instance = _playwright_instance.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            logger.info("[Phase3-Render] Chromium ready.")
    return _browser_instance


def shutdown_browser() -> None:
    """Cleanly close Playwright browser on app shutdown."""
    global _playwright_instance, _browser_instance
    lock = _get_browser_lock()
    with lock:
        if _browser_instance:
            try: _browser_instance.close()
            except Exception: pass
            _browser_instance = None
        if _playwright_instance:
            try: _playwright_instance.stop()
            except Exception: pass
            _playwright_instance = None
    logger.info("[Phase3-Render] Chromium browser closed.")


def _render_html(report_data: ReportData) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(settings.report_templates_dir))),
        autoescape=select_autoescape(["html"]),
    )
    return env.get_template("report.html").render(report=report_data)


def _build_pdf_in_thread(html_content: str, filepath: str) -> None:
    browser = _get_browser()
    context = browser.new_context()
    page = context.new_page()
    try:
        page.set_content(html_content, wait_until="networkidle")
        page.pdf(path=filepath, format="A4", print_background=True,
                 margin={"top": "10mm", "bottom": "10mm", "left": "12mm", "right": "12mm"})
        logger.info("[Phase3-Render] PDF written: %s", filepath)
    finally:
        page.close()
        context.close()


_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="report-pdf")


def phase_render_pdf(report_data: ReportData, filepath: str) -> None:
    logger.info("[Phase3-Render] Rendering PDF | file=%s", filepath)
    html = _render_html(report_data)
    future = _thread_pool.submit(_build_pdf_in_thread, html, filepath)
    try:
        future.result(timeout=60)
    except concurrent.futures.TimeoutError:
        raise RuntimeError("[Phase3-Render] PDF generation timed out after 60 seconds.")
    except Exception as exc:
        raise RuntimeError(f"[Phase3-Render] PDF generation failed: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_report_pipeline(
    animal_name: str, animal_type: str, owner_name: str, weight_kg: float,
    diagnosis: str, treatment: str, doctor_name: str,
    doctor_notes: str = "", visit_date: Optional[str] = None,
) -> dict:
    """Run the 2-phase report generation pipeline. Synchronous — safe in LangGraph ToolNode.

    Returns a structured dictionary with:
        status (str): "ok"
        message (str): Human-readable Arabic success message
        data (dict): report_id, filename, download_url, file_size_kb,
                     execution_time_sec, patient_info, doctor_name
    """
    t_start = time.monotonic()
    report_id = str(uuid.uuid4())[:8]
    visit_date = visit_date or datetime.now().strftime("%Y-%m-%d")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    logger.info("[Report] START | id=%s | patient=%s | diag=%s", report_id, animal_name, diagnosis[:60])

    # Phase 1 — Research (no LLM calls)
    sources = phase_research(diagnosis, treatment, animal_type)

    # Phase 2 — Generate (single LLM call: formalize + write sections)
    content = phase_generate(
        diagnosis_raw=diagnosis,
        treatment_raw=treatment,
        doctor_notes_raw=doctor_notes,
        animal_name=animal_name,
        animal_type=animal_type,
        weight_kg=weight_kg,
        sources=sources,
    )

    report_data = ReportData(
        animal_name=animal_name, animal_type=animal_type, owner_name=owner_name, weight_kg=weight_kg,
        diagnosis_ar=content.diagnosis_ar, diagnosis_en=content.diagnosis_en,
        treatment_ar=content.treatment_ar, treatment_en=content.treatment_en,
        content=content, doctor_name=doctor_name,
        visit_date=visit_date, generated_at=generated_at, report_id=report_id,
    )

    safe_name = animal_name.replace(" ", "_").replace("/", "_")[:30]
    filename  = f"report_{report_id}_{safe_name}.pdf"
    filepath  = str(REPORTS_DIR / filename)

    # Phase 3 — Render PDF (Playwright — kept as-is)
    phase_render_pdf(report_data, filepath)

    import os
    if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
        raise RuntimeError(f"PDF not found on disk after generation: {filepath}")

    size_kb = os.path.getsize(filepath) // 1024
    elapsed = round(time.monotonic() - t_start, 1)
    logger.info("[Report] DONE | id=%s | file=%s | %dKB | %.1fs", report_id, filename, size_kb, elapsed)

    return {
        "status": "ok",
        "message": "✅ تم إنشاء التقرير بنجاح!",
        "data": {
            "report_id": report_id,
            "filename": filename,
            "download_url": f"/copilot/reports/{filename}",
            "file_size_kb": size_kb,
            "execution_time_sec": elapsed,
            "patient_info": {
                "animal_name": animal_name,
                "animal_type": animal_type,
                "owner_name": owner_name,
                "weight_kg": weight_kg,
            },
            "doctor_name": doctor_name,
        },
    }
