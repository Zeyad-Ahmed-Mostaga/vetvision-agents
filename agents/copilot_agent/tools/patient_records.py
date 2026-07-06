"""
agents/copilot_agent/tools/patient_records.py — Patient Record Management Tools
=================================================================================
Three tools:
  1. register_first_visit  — WRITE: creates a NEW patient + first visit, returns 6-char ID
  2. log_returning_visit   — WRITE: appends a new visit to an existing patient by Animal ID
  3. get_patient_history   — READ-ONLY: retrieves full history by 6-char Animal ID

CRITICAL WRITE SAFETY:
  Tools 1 and 2 perform database writes. They MUST ONLY be called when the doctor
  has EXPLICITLY requested to register/save a patient visit.
  NEVER call them for questions, summaries, history lookups, or general conversation.
"""

import logging
import re
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from db.crud import register_new_patient, add_visit_to_existing, format_patient_history

logger = logging.getLogger(__name__)

# ── Pydantic Schemas ──────────────────────────────────────────────────────────


class RegisterFirstVisitInput(BaseModel):
    """Input schema for register_first_visit — new patient only."""
    animal_name:  str             = Field(..., description="Name of the animal patient")
    animal_type:  str             = Field(..., description="Type of animal (free text, e.g. 'cat', 'dog', 'parrot')")
    owner_name:   str             = Field(..., description="Full name of the Animal owner")
    weight_kg:    Optional[float] = Field(None, description="Animal's weight in kilograms at this visit (optional — pass null if unknown)")
    diagnosis:    str             = Field(..., description="Clinical diagnosis for this visit")
    treatment:    str             = Field(..., description="Treatment prescribed for this visit")
    doctor_notes: Optional[str]   = Field(None, description="Doctor's clinical observations, suspected conditions, or notes (optional — pass null if none)")
    visit_date:   str             = Field(..., description="Date of visit in ISO format YYYY-MM-DD")
    doctor_name:  str             = Field(..., description="Full name of the attending veterinarian (as stated in conversation)")


class LogReturningVisitInput(BaseModel):
    """Input schema for log_returning_visit — existing patient only."""
    animal_id:    str             = Field(..., description="Existing 6-character alphanumeric Animal ID (e.g. 'A3X7K9')")
    weight_kg:    Optional[float] = Field(None, description="Animal's weight in kilograms at this visit (optional — pass null if unknown)")
    diagnosis:    str             = Field(..., description="Clinical diagnosis for this visit")
    treatment:    str             = Field(..., description="Treatment prescribed for this visit")
    doctor_notes: Optional[str]   = Field(None, description="Doctor's clinical observations, suspected conditions, or notes (optional — pass null if none)")
    visit_date:   str             = Field(..., description="Date of visit in ISO format YYYY-MM-DD")
    doctor_name:  str             = Field(..., description="Full name of the attending veterinarian (as stated in conversation)")

    @field_validator("animal_id")
    @classmethod
    def validate_animal_id(cls, v: str) -> str:
        normalized = v.strip().upper()
        if not re.match(r'^[A-Z0-9]{6}$', normalized):
            raise ValueError(
                f"Invalid Animal ID '{v}'. Must be exactly 6 alphanumeric characters (letters and digits only)."
            )
        return normalized


class GetPatientHistoryInput(BaseModel):
    """Input schema for get_patient_history."""
    animal_id: str = Field(
        ...,
        description=(
            "The patient's 6-character alphanumeric Animal ID (e.g. 'A3X7K9'). "
            "Case-insensitive — will be normalized to uppercase."
        )
    )


# ── Tool 1: Register First Visit ──────────────────────────────────────────────

@tool(args_schema=RegisterFirstVisitInput)
def register_first_visit(
    animal_name: str,
    animal_type: str,
    owner_name: str,
    weight_kg: Optional[float],
    diagnosis: str,
    treatment: str,
    doctor_notes: Optional[str],
    visit_date: str,
    doctor_name: str,
) -> str:
    """
    Register a BRAND-NEW patient and log their FIRST visit to the database.

    ⚠️ WRITE OPERATION — STRICT SAFETY RULE:
    Only call this tool when the doctor has EXPLICITLY confirmed:
      1. This is the patient's FIRST-EVER visit, AND
      2. They want to SAVE / REGISTER this visit.
    NEVER call for questions, summaries, history lookups, or any other purpose.

    This tool generates a unique 6-character Animal ID and returns it to the doctor.
    The doctor should keep this ID for future visits.

    Returns:
        Confirmation message with the generated 6-character Animal ID.
    """
    logger.info(
        "[register_first_visit] animal=%s type=%s owner=%s doctor=%s",
        animal_name, animal_type, owner_name, doctor_name,
    )
    try:
        animal_id = register_new_patient(
            animal_name=animal_name,
            animal_type=animal_type,
            owner_name=owner_name,
            weight_kg=weight_kg,
            diagnosis=diagnosis,
            treatment=treatment,
            doctor_notes=doctor_notes,
            visit_date=visit_date,
            doctor_name=doctor_name,
        )
        weight_display = f"{weight_kg} kg" if weight_kg is not None else "not recorded"
        notes_display = doctor_notes if doctor_notes else "None"
        return (
            f"✅ New patient registered successfully!\n"
            f"\n"
            f"🆔 Animal ID: **{animal_id}**\n"
            f"   Please give this ID to the owner — they'll need it for future visits.\n"
            f"\n"
            f"Patient:    {animal_name} ({animal_type})\n"
            f"Owner:      {owner_name}\n"
            f"Weight:     {weight_display}\n"
            f"Diagnosis:  {diagnosis}\n"
            f"Treatment:  {treatment}\n"
            f"Notes:      {notes_display}\n"
            f"Visit Date: {visit_date}\n"
            f"Doctor:     Dr. {doctor_name}"
        )
    except Exception as exc:
        logger.error("[register_first_visit] Failed: %s", exc, exc_info=True)
        return f"❌ Failed to register patient: {exc}"


# ── Tool 2: Log Returning Visit ───────────────────────────────────────────────

@tool(args_schema=LogReturningVisitInput)
def log_returning_visit(
    animal_id: str,
    weight_kg: Optional[float],
    diagnosis: str,
    treatment: str,
    doctor_notes: Optional[str],
    visit_date: str,
    doctor_name: str,
) -> str:
    """
    Log a new visit for an EXISTING (returning) patient using their Animal ID.

    ⚠️ WRITE OPERATION — STRICT SAFETY RULE:
    Only call this tool when the doctor has EXPLICITLY confirmed:
      1. This is a RETURNING patient (they have visited before), AND
      2. They have provided the existing 6-character Animal ID, AND
      3. They want to SAVE / LOG this visit.
    NEVER call for questions, summaries, history lookups, or any other purpose.

    Args:
        animal_id: Existing 6-char Animal ID (e.g. 'A3X7K9'). Case-insensitive.

    Returns:
        Confirmation message, or an error if the Animal ID is not found.
    """
    animal_id = animal_id.strip().upper()
    logger.info(
        "[log_returning_visit] animal_id=%s doctor=%s",
        animal_id, doctor_name,
    )
    try:
        result = add_visit_to_existing(
            animal_id=animal_id,
            weight_kg=weight_kg,
            diagnosis=diagnosis,
            treatment=treatment,
            doctor_notes=doctor_notes,
            visit_date=visit_date,
            doctor_name=doctor_name,
        )
        weight_display = f"{weight_kg} kg" if weight_kg is not None else "not recorded"
        notes_display = doctor_notes if doctor_notes else "None"
        return (
            f"✅ Visit logged successfully for returning patient!\n"
            f"\n"
            f"🆔 Animal ID:  {result['animal_id']}\n"
            f"Patient:      {result['animal_name']}\n"
            f"Weight:       {weight_display}\n"
            f"Diagnosis:    {diagnosis}\n"
            f"Treatment:    {treatment}\n"
            f"Notes:        {notes_display}\n"
            f"Visit Date:   {visit_date}\n"
            f"Doctor:       Dr. {doctor_name}"
        )
    except ValueError as exc:
        logger.warning("[log_returning_visit] Validation error: %s", exc)
        return f"❌ {exc}"
    except Exception as exc:
        logger.error("[log_returning_visit] Failed: %s", exc, exc_info=True)
        return f"❌ Failed to log visit: {exc}"


# ── Tool 3: Get Patient History (READ-ONLY) ───────────────────────────────────

@tool(args_schema=GetPatientHistoryInput)
def get_patient_history(animal_id: str) -> str:
    """
    Retrieve the full medical history of a patient by their 6-character Animal ID.

    ✅ READ-ONLY — This tool does NOT write anything to the database.
    Safe to call for summaries, history lookups, and general information retrieval.

    Returns a formatted summary including:
    - Patient details (name, type, owner, doctor)
    - All past visits with: visit date, weight at that visit, diagnosis, treatment

    Args:
        animal_id: 6-character alphanumeric Animal ID (e.g. 'A3X7K9').
    """
    animal_id = animal_id.strip().upper()
    logger.info("[get_patient_history] animal_id=%s", animal_id)

    try:
        result = format_patient_history(animal_id)
        return result
    except Exception as exc:
        logger.error("[get_patient_history] Failed: %s", exc, exc_info=True)
        return f"❌ Failed to retrieve patient history: {exc}"
