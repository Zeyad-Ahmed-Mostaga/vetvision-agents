"""
tools/report/schemas.py — Pydantic Data Models for the Report Pipeline
========================================================================
All data contracts between pipeline phases.  Every LLM output is validated
against one of these models via .with_structured_output() — zero string parsing.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Phase 1 output ────────────────────────────────────────────────────────────

class FormalizedNotes(BaseModel):
    """LLM-formalized version of the doctor's raw notes."""
    diagnosis_en: str = Field(
        description="Formal English medical diagnosis term or short phrase (≤8 words)"
    )
    diagnosis_ar: str = Field(
        description="Formal Arabic medical diagnosis — written as a professional Arabic vet would write it"
    )
    treatment_en: str = Field(
        description="Formal English treatment plan — one concise sentence"
    )
    treatment_ar: str = Field(
        description="Formal Arabic treatment plan — one concise sentence in natural veterinary Arabic"
    )
    is_medication: bool = Field(
        description=(
            "True ONLY if the treatment involves a drug, medication, or product "
            "purchased from a pharmacy or veterinary store. False for procedures, "
            "rest, dietary changes, or non-purchasable interventions."
        )
    )


# ── Phase 2 intermediate ──────────────────────────────────────────────────────

class WebResultRelevance(BaseModel):
    """Used to score a web search result's relevance before including it."""
    is_relevant: bool = Field(
        description=(
            "True if the web search result contains specific, factual veterinary "
            "information about the given diagnosis and animal type. "
            "False if it is generic, promotional, or unrelated."
        )
    )
    reason: str = Field(description="One sentence explaining the relevance decision.")


class ResearchResults(BaseModel):
    """Aggregated research data from RAG + optional web search."""
    rag_text: str       # Concatenated RAG chunks (may be empty)
    web_text: str       # Validated web search results (may be empty)
    source_label: str   # "rag" | "web" | "rag+web" | "none"
    rag_chunk_count: int


# ── Phase 3 output ────────────────────────────────────────────────────────────

class ReportContent(BaseModel):
    """All AI-generated report sections, primarily in Arabic."""
    overview: str = Field(
        description=(
            "2-3 sentences in Arabic explaining what this condition is and what causes it. "
            "Use friendly, non-medical language a pet owner can understand. "
            "Medical terms (e.g., drug names) may be in English where natural."
        )
    )
    symptoms: str = Field(
        description=(
            "Bullet-point list (using • or -) in Arabic of specific symptoms "
            "the owner should watch for at home. Extract ONLY from the provided sources."
        )
    )
    home_care: str = Field(
        description=(
            "Practical home care instructions in Arabic (2-4 bullet points). "
            "Extract ONLY from sources. Do not invent instructions."
        )
    )
    prevention: str = Field(
        description=(
            "2-3 prevention tips in Arabic relevant to this condition and animal type. "
            "If no prevention info is in sources, write: "
            "استشر طبيبك البيطري للحصول على نصائح الوقاية المناسبة."
        )
    )
    medication_info: str = Field(
        description=(
            "If is_medication is True: medication name and typical dosage — "
            "written in Arabic (drug name may be in English/Latin). "
            "Do NOT include any prices or cost information. "
            "If is_medication is False: write 'لا يتطلب العلاج دواءً. ' followed by the treatment plan in Arabic."
        )
    )
    doctor_notes: str = Field(
        description=(
            "Additional notes for the pet owner from the doctor. "
            "If doctor provided notes, write them in Arabic. "
            "If no notes were provided, write an empty string."
        )
    )


# ── Final assembled report data object ───────────────────────────────────────

class ReportData(BaseModel):
    """Complete data object passed to the Jinja2 HTML template for rendering."""
    # Patient info
    animal_name: str
    animal_type: str
    owner_name: str
    weight_kg: float

    # Formalized medical info
    diagnosis_ar: str
    diagnosis_en: str
    treatment_ar: str
    treatment_en: str

    # AI-generated content
    content: ReportContent

    # Doctor info — name only, no doctor_id
    doctor_name: str
    doctor_notes_input: str  # Raw doctor notes from tool input (may be empty)

    # Meta
    visit_date: str
    generated_at: str
    report_id: str
