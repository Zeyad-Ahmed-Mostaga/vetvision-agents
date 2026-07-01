"""
agents/copilot_agent/tools/report/tool.py — LangChain Tool Wrapper
====================================================================
Wraps the report pipeline as a @tool for the LangGraph agent.

Async safety:
  The tool function is synchronous (required by LangGraph's ToolNode).
  Playwright runs in a ThreadPoolExecutor worker thread with no event loop.
"""

import logging
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.copilot_agent.tools.report.pipeline import generate_report_pipeline, REPORTS_DIR

logger = logging.getLogger(__name__)


# ── Input Schema ──────────────────────────────────────────────────────────────

class GenerateReportInput(BaseModel):
    """Input schema for the generate_patient_report tool."""
    animal_name:  str            = Field(..., description="Name of the animal patient")
    animal_type:  str            = Field(..., description="Type of animal (e.g. cat, dog, parrot, rabbit)")
    owner_name:   str            = Field(..., description="Name of the pet owner")
    weight_kg:    float          = Field(..., description="Weight of the animal in kilograms at this visit")
    diagnosis:    str            = Field(
        ...,
        description=(
            "Formalized medical diagnosis. Before passing to this tool, YOU MUST "
            "convert the doctor's raw/informal diagnosis into a proper veterinary "
            "medical term or phrase (e.g. 'Feline Panleukopenia Virus Infection' "
            "instead of 'بانلوكوبينيا' or 'cat has parvo'). "
            "The pipeline will further refine this, but provide your best "
            "professional formalization as a first pass."
        ),
    )
    treatment:    str            = Field(
        ...,
        description=(
            "Formalized treatment plan. Before passing to this tool, YOU MUST "
            "convert the doctor's raw/informal treatment notes into a clear, "
            "professional medical treatment statement (e.g. 'IV fluid therapy "
            "with antiemetic and antibiotic course for 5 days' instead of "
            "'هنديله محاليل ومضاد حيوي'). "
            "The pipeline will further refine this, but provide your best "
            "professional formalization as a first pass."
        ),
    )
    doctor_name:  str            = Field(..., description="Full name of the attending veterinarian (as stated in conversation)")
    doctor_notes: Optional[str]  = Field(
        default="",
        description=(
            "Optional additional notes from the doctor to the animal owner. "
            "If the doctor provided notes, YOU MUST rewrite them into clear, "
            "professional medical language before passing to this tool — fix "
            "informal phrasing, abbreviations, and spelling errors. "
            "The pipeline will further refine this, but provide your best "
            "professional formalization as a first pass. "
            "Leave empty or omit if no additional notes."
        ),
    )


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool(args_schema=GenerateReportInput)
def generate_patient_report(
    animal_name: str,
    animal_type: str,
    owner_name: str,
    weight_kg: float,
    diagnosis: str,
    treatment: str,
    doctor_name: str,
    doctor_notes: Optional[str] = "",
) -> str:
    """
    Generate a professional Arabic-primary PDF medical report for a veterinary patient.

    The report includes:
    - Patient information (animal, owner, weight at this visit, visit date)
    - Formal diagnosis and treatment plan in Arabic and English
    - Condition overview, symptoms to watch, home care, prevention tips
    - Doctor's notes section (optional — rewritten into professional Arabic)
    - VetVision branded header and footer

    Content is grounded in the VetVision knowledge base (RAG) with web search
    as fallback. Returns a success message with filename and download URL.
    """
    logger.info(
        "[Tool:generate_patient_report] Called | animal=%s (%s) | diag=%.60s",
        animal_name, animal_type, diagnosis,
    )
    try:
        result = generate_report_pipeline(
            animal_name=animal_name,
            animal_type=animal_type,
            owner_name=owner_name,
            weight_kg=weight_kg,
            diagnosis=diagnosis,
            treatment=treatment,
            doctor_name=doctor_name,
            doctor_notes=doctor_notes or "",
        )
        # Format the structured dict into the descriptive string the LLM expects
        d = result["data"]
        p = d["patient_info"]
        return (
            f"{result['message']}\n"
            f"Filename: {d['filename']}\n"
            f"Download URL: {d['download_url']}\n"
            f"File size: {d['file_size_kb']} KB\n"
            f"Patient: {p['animal_name']} ({p['animal_type']}) | Doctor: Dr. {d['doctor_name']}\n"
            f"Report ID: {d['report_id']} | Time: {d['execution_time_sec']}s"
        )
    except Exception as exc:
        logger.error("[Tool:generate_patient_report] Pipeline failed: %s", exc, exc_info=True)
        return (
            f"❌ فشل إنشاء التقرير: {exc}\n"
            "يرجى المحاولة مرة أخرى أو التواصل مع الدعم الفني."
        )
