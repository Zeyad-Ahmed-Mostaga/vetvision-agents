"""
tools/report/schemas.py — Pydantic Data Models for the Report Pipeline
========================================================================
Data contracts for the 2-phase report pipeline.
Phase 1: Research (RAG + optional Tavily) — no schema needed, returns raw text.
Phase 2: Generate  — single LLM call that formalizes AND writes all sections.

All LLM output is validated via .with_structured_output() — zero string parsing.
"""

from pydantic import BaseModel, Field


# ── Phase 2 output — single LLM call produces everything ─────────────────────

class ReportContent(BaseModel):
    """All AI-generated report content from a single LLM call.
    The LLM formalizes the raw doctor notes AND writes all Arabic sections."""

    # ── Formalized medical info (replaces old Phase 1) ────────────────────────
    diagnosis_en: str = Field(
        description=(
            "Formal English medical diagnosis. One precise veterinary term or "
            "short phrase, max 8 words. Example: 'Feline Panleukopenia Virus Infection'"
        )
    )
    diagnosis_ar: str = Field(
        description=(
            "Formal Arabic medical diagnosis — written exactly as a professional "
            "Arabic-speaking veterinarian would write it in a clinical report. "
            "Must be medically precise, not a casual translation."
        )
    )
    treatment_en: str = Field(
        description=(
            "Formal English treatment plan — one concise sentence describing "
            "what was prescribed or performed."
        )
    )
    treatment_ar: str = Field(
        description=(
            "Formal Arabic treatment plan — one concise sentence in natural "
            "veterinary Arabic. Must sound like a real vet wrote it."
        )
    )

    # ── Report body sections (all in Arabic) ──────────────────────────────────
    overview: str = Field(
        description=(
            "3-5 sentences in Arabic explaining what this condition/diagnosis is, "
            "what causes it, and why it matters for this animal type. "
            "Write for a Animal owner with no medical background — use simple, warm language. "
            "Medical terms (drug names, disease names) may appear in English where natural. "
            "Do NOT just restate the diagnosis. Explain the condition."
        )
    )
    symptoms: str = Field(
        description=(
            "A bullet-point list (using • character) in Arabic of 3-6 specific symptoms "
            "the owner should watch for at home AFTER this visit. "
            "Each bullet must be a concrete, observable sign — not vague. "
            "Extract ONLY from the provided source material. "
            "Example of good bullet: '• فقدان الشهية ورفض الطعام لأكثر من ٢٤ ساعة' "
            "Example of bad bullet: '• أعراض عامة'"
        )
    )
    home_care: str = Field(
        description=(
            "3-5 bullet points (using • character) in Arabic with practical, actionable "
            "home care instructions the owner should follow. "
            "Each point must be specific enough to act on — include quantities, frequencies, "
            "or durations where the sources provide them. "
            "Extract ONLY from sources. Do not invent medical instructions. "
            "Example of good point: '• تقديم كميات صغيرة من الماء كل ساعة لتجنب الجفاف' "
            "Example of bad point: '• الاهتمام بالحيوان'"
        )
    )
    prevention: str = Field(
        description=(
            "2-4 bullet points (using • character) in Arabic with prevention tips "
            "relevant to this specific condition and animal type. "
            "If no prevention information exists in the sources, write exactly: "
            "'استشر طبيبك البيطري للحصول على نصائح الوقاية المناسبة لحالة حيوانك.'"
        )
    )
    doctor_notes: str = Field(
        description=(
            "If the doctor provided additional notes: rewrite them into professional, "
            "clear Arabic suitable for a medical report. Fix any spelling errors, "
            "informal language, or abbreviations. The output must read as if written "
            "by a professional veterinarian. "
            "If no doctor notes were provided (empty input), return an empty string."
        )
    )


# ── Final assembled report data object ────────────────────────────────────────

class ReportData(BaseModel):
    """Complete data object passed to the Jinja2 HTML template for rendering."""
    # Patient info
    animal_name: str
    animal_type: str
    owner_name: str
    weight_kg: float

    # Formalized medical info (from ReportContent)
    diagnosis_ar: str
    diagnosis_en: str
    treatment_ar: str
    treatment_en: str

    # AI-generated content sections
    content: ReportContent

    # Doctor info
    doctor_name: str

    # Meta
    visit_date: str
    generated_at: str
    report_id: str
