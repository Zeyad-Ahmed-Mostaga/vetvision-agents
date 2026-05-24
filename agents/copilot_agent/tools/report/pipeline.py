"""
agents/copilot_agent/tools/report/pipeline.py — 4-Phase Report Pipeline
=========================================================================
Phases:
  1. Formalize   — LLM rewrites raw doctor notes into formal medical language
  2. Research    — Multi-query RAG → sufficiency check → Tavily fallback
  3. Structure   — Single LLM call writes all Arabic report sections as JSON
  4. Render PDF  — Jinja2 HTML template → Playwright/Chromium → PDF file
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
    FormalizedNotes, ResearchResults, WebResultRelevance,
    ReportContent, ReportData,
)

logger = logging.getLogger(__name__)

# ── Paths (from config.settings — no fragile parent counting) ─────────────────
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
# PHASE 1 — Formalize raw doctor notes
# ══════════════════════════════════════════════════════════════════════════════

def phase_formalize(diagnosis_raw: str, treatment_raw: str, animal_type: str) -> FormalizedNotes:
    """Rewrite raw doctor notes into formal veterinary medical language."""
    logger.info("[Phase1] Formalizing notes for animal_type=%s", animal_type)
    prompt_messages = [
        ("system",
         "You are a professional veterinary medical editor. Rewrite the doctor's raw "
         "clinical notes into formal, concise veterinary medical language.\n"
         "Rules:\n"
         "- Diagnosis: One formal medical term or short phrase (max 8 words)\n"
         "- Treatment: One concise sentence describing the treatment plan\n"
         "- Write Arabic naturally as a professional Arabic veterinarian\n"
         "- is_medication: true ONLY if treatment involves a purchasable drug/product\n"
         "Respond ONLY with the structured data."),
        ("human", "Animal type: {animal_type}\nDiagnosis: {diagnosis}\nTreatment: {treatment}"),
    ]
    try:
        result = _llm_with_retry(
            schema=FormalizedNotes,
            prompt_messages=prompt_messages,
            invoke_kwargs={"animal_type": animal_type, "diagnosis": diagnosis_raw, "treatment": treatment_raw},
            phase_name="Phase1-Formalize",
        )
        logger.info("[Phase1] Done | diag_en=%s | is_med=%s", result.diagnosis_en, result.is_medication)
        return result
    except Exception as exc:
        logger.error("[Phase1] All retries exhausted — using raw inputs. Error: %s", exc)
        return FormalizedNotes(
            diagnosis_en=diagnosis_raw[:100], diagnosis_ar=diagnosis_raw[:100],
            treatment_en=treatment_raw[:200], treatment_ar=treatment_raw[:200],
            is_medication=False,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Multi-query RAG + validated Tavily fallback
# ══════════════════════════════════════════════════════════════════════════════

def _map_animal_type(animal_type_raw: str) -> str:
    cats   = {"cat","kitten","cats","قطة","قط","قطط","هرة","هريرة"}
    dogs   = {"dog","puppy","dogs","كلب","جرو","كلاب"}
    horses = {"horse","foal","horses","حصان","مهر","خيل","فرس"}
    t = animal_type_raw.lower().strip()
    if t in cats:   return "cat"
    if t in dogs:   return "dog"
    if t in horses: return "horse"
    return "other"


def _rag_retrieve(diagnosis_en: str, animal_type_raw: str, rag_cat: str) -> list[str]:
    from rag.retrieval import advanced_rag_retrieve
    queries = [
        f"{diagnosis_en} in {animal_type_raw}",
        f"{diagnosis_en} symptoms signs {animal_type_raw}",
        f"{diagnosis_en} treatment home care prevention {animal_type_raw}",
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
            logger.warning("[Phase2] RAG query failed (%s): %s", q, exc)
    logger.info("[Phase2] RAG returned %d unique chunks.", len(chunks))
    return chunks


def _validate_web_result(result_text: str, diagnosis_en: str, animal_type: str) -> bool:
    prompt_messages = [
        ("system",
         "You are a veterinary content validator. Determine if the given web search "
         "result contains specific, factual veterinary information relevant to the "
         "provided diagnosis and animal type. Answer ONLY with the structured data."),
        ("human", "Diagnosis: {diagnosis}\nAnimal type: {animal_type}\n\nWeb search result:\n{result_text}"),
    ]
    try:
        result = _llm_with_retry(
            schema=WebResultRelevance,
            prompt_messages=prompt_messages,
            invoke_kwargs={"diagnosis": diagnosis_en, "animal_type": animal_type, "result_text": result_text[:1500]},
            phase_name="Phase2-WebValidation",
        )
        logger.info("[Phase2] Web validation: is_relevant=%s", result.is_relevant)
        return result.is_relevant
    except Exception as exc:
        logger.warning("[Phase2] Web validation failed (%s) — excluding result.", exc)
        return False


def _tavily_search(query: str) -> str:
    try:
        from agents.copilot_agent.tools.web_search import tavily_search
        result = tavily_search.invoke(query)
        return str(result) if result else ""
    except Exception as exc:
        logger.warning("[Phase2] Tavily search failed (%s): %s", query, exc)
        return ""


def phase_research(formal: FormalizedNotes, animal_type_raw: str) -> ResearchResults:
    """Gather veterinary knowledge: multi-query RAG + optional validated Tavily fallback."""
    rag_cat = _map_animal_type(animal_type_raw)
    logger.info("[Phase2] Research start | diag=%s | rag_cat=%s", formal.diagnosis_en, rag_cat)

    chunks = _rag_retrieve(formal.diagnosis_en, animal_type_raw, rag_cat)
    rag_text = "\n\n---\n\n".join(chunks)
    rag_total_chars = sum(len(c) for c in chunks)

    web_parts: list[str] = []
    if rag_total_chars < settings.report_rag_min_chars:
        logger.info("[Phase2] RAG insufficient (%d chars) — triggering Tavily fallback.", rag_total_chars)
        sym_raw = _tavily_search(f"{formal.diagnosis_en} symptoms signs {animal_type_raw} veterinary")
        care_raw = _tavily_search(f"{formal.diagnosis_en} home care treatment {animal_type_raw}")
        for label, raw in [("symptoms", sym_raw), ("care", care_raw)]:
            if raw and _validate_web_result(raw, formal.diagnosis_en, animal_type_raw):
                web_parts.append(f"[Web — {label}]\n{raw}")
    else:
        logger.info("[Phase2] RAG sufficient (%d chars) — skipping Tavily.", rag_total_chars)

    web_text = "\n\n".join(web_parts)
    source = ("rag+web" if rag_text and web_text else
              "rag" if rag_text else
              "web" if web_text else "none")
    logger.info("[Phase2] Done | source=%s | rag=%d | web=%d", source, len(rag_text), len(web_text))
    return ResearchResults(rag_text=rag_text, web_text=web_text, source_label=source, rag_chunk_count=len(chunks))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Structure all content sections via a single LLM call
# ══════════════════════════════════════════════════════════════════════════════

_STRUCTURE_SYSTEM = """\
أنت طبيب بيطري عربي محترف تكتب تقريراً طبياً لصاحب الحيوان الأليف.
مهمتك: بناءً على مواد المصدر المقدمة، اكتب محتوى التقرير لكل قسم.
القواعد: استخدم المصادر فقط — لا اختراع. اكتب بعربية مفهومة لصاحب الحيوان.
مواد المصدر:
{sources}
"""

_FALLBACK_CONTENT = ReportContent(
    overview="يرجى مراجعة طبيبك البيطري للحصول على مزيد من المعلومات.",
    symptoms="يرجى مراجعة طبيبك البيطري للاطلاع على الأعراض.",
    home_care="يرجى اتباع تعليمات طبيبك البيطري.",
    prevention="استشر طبيبك البيطري للحصول على نصائح الوقاية.",
    medication_info="يرجى مراجعة طبيبك البيطري للحصول على معلومات الدواء.",
    doctor_notes="",
)


def phase_structure(formal: FormalizedNotes, research: ResearchResults, doctor_notes_input: str) -> ReportContent:
    """Single LLM call writing all Arabic report sections as validated JSON."""
    logger.info("[Phase3] Structuring content | source=%s", research.source_label)
    parts: list[str] = []
    if research.rag_text:
        parts.append(f"=== قاعدة المعرفة البيطرية ===\n{research.rag_text}")
    if research.web_text:
        parts.append(f"=== نتائج البحث ===\n{research.web_text}")
    sources = "\n\n".join(parts) if parts else "لا توجد مصادر."

    prompt_messages = [
        ("system", _STRUCTURE_SYSTEM),
        ("human",
         "التشخيص EN: {diagnosis_en}\nالتشخيص AR: {diagnosis_ar}\n"
         "العلاج: {treatment_ar}\nدواء: {is_med}\nملاحظات: {doctor_notes}"),
    ]
    try:
        result = _llm_with_retry(
            schema=ReportContent,
            prompt_messages=prompt_messages,
            invoke_kwargs={
                "sources": sources,
                "diagnosis_en": formal.diagnosis_en, "diagnosis_ar": formal.diagnosis_ar,
                "treatment_ar": formal.treatment_ar,
                "is_med": "نعم" if formal.is_medication else "لا",
                "doctor_notes": doctor_notes_input.strip() if doctor_notes_input else "",
            },
            phase_name="Phase3-Structure",
        )
        logger.info("[Phase3] Content structured successfully.")
        return result
    except Exception as exc:
        logger.error("[Phase3] Fallback triggered. Error: %s", exc)
        fallback = _FALLBACK_CONTENT.model_copy()
        fallback.doctor_notes = doctor_notes_input.strip() if doctor_notes_input else ""
        return fallback


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — HTML → Playwright → PDF
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
            logger.info("[Phase4] Launching Playwright Chromium...")
            from playwright.sync_api import sync_playwright
            _playwright_instance = sync_playwright().start()
            _browser_instance = _playwright_instance.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            logger.info("[Phase4] Chromium ready.")
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
    logger.info("[Phase4] Chromium browser closed.")


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
        logger.info("[Phase4] PDF written: %s", filepath)
    finally:
        page.close()
        context.close()


_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="report-pdf")


def phase_render_pdf(report_data: ReportData, filepath: str) -> None:
    logger.info("[Phase4] Rendering PDF | file=%s", filepath)
    html = _render_html(report_data)
    future = _thread_pool.submit(_build_pdf_in_thread, html, filepath)
    try:
        future.result(timeout=60)
    except concurrent.futures.TimeoutError:
        raise RuntimeError("[Phase4] PDF generation timed out after 60 seconds.")
    except Exception as exc:
        raise RuntimeError(f"[Phase4] PDF generation failed: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_report_pipeline(
    animal_name: str, animal_type: str, owner_name: str, weight_kg: float,
    diagnosis: str, treatment: str, doctor_name: str,
    doctor_notes: str = "", visit_date: Optional[str] = None,
) -> str:
    """Run the full 4-phase report generation pipeline. Synchronous — safe in LangGraph ToolNode."""
    t_start = time.monotonic()
    report_id = str(uuid.uuid4())[:8]
    visit_date = visit_date or datetime.now().strftime("%Y-%m-%d")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    logger.info("[Report] START | id=%s | patient=%s | diag=%s", report_id, animal_name, diagnosis[:60])

    formal   = phase_formalize(diagnosis, treatment, animal_type)
    research = phase_research(formal, animal_type)
    content  = phase_structure(formal, research, doctor_notes)

    report_data = ReportData(
        animal_name=animal_name, animal_type=animal_type, owner_name=owner_name, weight_kg=weight_kg,
        diagnosis_ar=formal.diagnosis_ar, diagnosis_en=formal.diagnosis_en,
        treatment_ar=formal.treatment_ar, treatment_en=formal.treatment_en,
        content=content, doctor_name=doctor_name,
        doctor_notes_input=doctor_notes.strip() if doctor_notes else "",
        visit_date=visit_date, generated_at=generated_at, report_id=report_id,
    )

    safe_name = animal_name.replace(" ", "_").replace("/", "_")[:30]
    filename  = f"report_{report_id}_{safe_name}.pdf"
    filepath  = str(REPORTS_DIR / filename)

    phase_render_pdf(report_data, filepath)

    import os
    if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
        raise RuntimeError(f"PDF not found on disk after generation: {filepath}")

    size_kb = os.path.getsize(filepath) // 1024
    elapsed = time.monotonic() - t_start
    logger.info("[Report] DONE | id=%s | file=%s | %dKB | %.1fs", report_id, filename, size_kb, elapsed)

    return (
        f"✅ تم إنشاء التقرير بنجاح!\n"
        f"Filename: {filename}\n"
        f"Download URL: /copilot/reports/{filename}\n"
        f"File size: {size_kb} KB\n"
        f"Patient: {animal_name} ({animal_type}) | Doctor: Dr. {doctor_name}\n"
        f"Report ID: {report_id} | Time: {elapsed:.1f}s"
    )
