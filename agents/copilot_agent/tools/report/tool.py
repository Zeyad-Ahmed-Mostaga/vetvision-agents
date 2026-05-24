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
    diagnosis:    str            = Field(..., description="Diagnosis (raw doctor notes accepted — will be formalized)")
    treatment:    str            = Field(..., description="Treatment prescribed (raw notes accepted)")
    doctor_name:  str            = Field(..., description="Full name of the attending veterinarian (as stated in conversation)")
    doctor_notes: Optional[str]  = Field(
        default="",
        description=(
            "Optional additional notes from the doctor to the pet owner. "
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
    - Medication information
    - Doctor's notes section (optional)
    - VetVision branded header and footer

    Content is grounded in the VetVision knowledge base (RAG) with validated
    web search as fallback. Returns a success message with filename and download URL.
    """
    logger.info(
        "[Tool:generate_patient_report] Called | animal=%s (%s) | diag=%.60s",
        animal_name, animal_type, diagnosis,
    )
    try:
        return generate_report_pipeline(
            animal_name=animal_name,
            animal_type=animal_type,
            owner_name=owner_name,
            weight_kg=weight_kg,
            diagnosis=diagnosis,
            treatment=treatment,
            doctor_name=doctor_name,
            doctor_notes=doctor_notes or "",
        )
    except Exception as exc:
        logger.error("[Tool:generate_patient_report] Pipeline failed: %s", exc, exc_info=True)
        return (
            f"❌ فشل إنشاء التقرير: {exc}\n"
            "يرجى المحاولة مرة أخرى أو التواصل مع الدعم الفني."
        )
