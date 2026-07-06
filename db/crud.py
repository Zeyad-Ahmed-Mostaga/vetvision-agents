"""
db/crud.py — Database Operations
==================================
CRUD functions for patient records and visit history.

Two distinct write paths enforce the first-time vs. returning patient flow:
  - register_new_patient()      → creates a new Patient + first Visit, returns 6-char ID
  - add_visit_to_existing()     → appends a Visit to an existing Patient by animal_id

All functions create their own session and handle cleanup — no session leaking.
"""

import logging
import re
from datetime import date, datetime
from typing import Optional

from db.models import SessionLocal, Patient, Visit, _new_animal_id

logger = logging.getLogger(__name__)

# ── ID Helpers ────────────────────────────────────────────────────────────────

_ANIMAL_ID_RE = re.compile(r'^[A-Z0-9]{6}$')


def _is_valid_animal_id(animal_id: str) -> bool:
    """Return True if animal_id is exactly 6 uppercase alphanumeric chars."""
    return bool(_ANIMAL_ID_RE.match(animal_id.strip().upper()))


def _generate_unique_id(session) -> str:
    """
    Generate a 6-char alphanumeric Animal ID that does not already exist in the DB.
    Tries up to 100 times before raising (collision probability is astronomically low).
    """
    for _ in range(100):
        new_id = _new_animal_id()
        exists = session.query(Patient).filter(Patient.animal_id == new_id).first()
        if not exists:
            return new_id
    raise RuntimeError("Failed to generate a unique Animal ID after 100 attempts.")


def _parse_date(visit_date: str) -> date:
    """Parse ISO date string; fall back to today if invalid."""
    try:
        return date.fromisoformat(visit_date)
    except (ValueError, TypeError):
        logger.warning("[DB] Invalid visit_date '%s', using today.", visit_date)
        return date.today()


# ── Write: First-Time Patient ─────────────────────────────────────────────────

def register_new_patient(
    animal_name: str,
    animal_type: str,
    owner_name: str,
    weight_kg: float,
    diagnosis: str,
    treatment: str,
    visit_date: str,
    doctor_name: str,
    doctor_notes: Optional[str] = None,
) -> str:
    """
    Register a brand-new patient and log their first visit.

    Creates a new Patient record with a freshly generated 6-char Animal ID,
    then immediately attaches the first Visit record.

    Returns:
        The generated Animal ID (6 uppercase alphanumeric chars).

    Raises:
        Exception on any DB error (caller should handle).
    """
    session = SessionLocal()
    try:
        animal_id = _generate_unique_id(session)

        patient = Patient(
            animal_id=animal_id,
            animal_name=animal_name,
            animal_type=animal_type,
            owner_name=owner_name,
            doctor_name=doctor_name,
        )
        session.add(patient)
        session.flush()  # persist patient so FK constraint is satisfied

        visit = Visit(
            animal_id=animal_id,
            diagnosis=diagnosis,
            treatment=treatment,
            doctor_notes=doctor_notes,
            weight_kg=weight_kg,
            visit_date=_parse_date(visit_date),
            doctor_name=doctor_name,
        )
        session.add(visit)
        session.commit()

        logger.info(
            "[DB] New patient registered | ID=%s | animal=%s | doctor=%s",
            animal_id, animal_name, doctor_name,
        )
        return animal_id

    except Exception as exc:
        session.rollback()
        logger.error("[DB] register_new_patient failed: %s", exc, exc_info=True)
        raise
    finally:
        session.close()


# ── Write: Returning Patient ──────────────────────────────────────────────────

def add_visit_to_existing(
    animal_id: str,
    weight_kg: float,
    diagnosis: str,
    treatment: str,
    visit_date: str,
    doctor_name: str,
    doctor_notes: Optional[str] = None,
) -> dict:
    """
    Append a new visit to an existing patient record identified by their Animal ID.

    Args:
        animal_id:   Existing 6-char Animal ID (case-insensitive — normalized to uppercase).
        weight_kg:   Animal's weight at this visit.
        diagnosis:   Clinical diagnosis for this visit.
        treatment:   Treatment prescribed.
        visit_date:  ISO date string (YYYY-MM-DD); defaults to today if invalid.
        doctor_name: Name of the attending doctor.

    Returns:
        Dict with patient info and new visit_id on success.

    Raises:
        ValueError if animal_id is not found in the database.
        Exception on any other DB error.
    """
    animal_id = animal_id.strip().upper()

    if not _is_valid_animal_id(animal_id):
        raise ValueError(
            f"Invalid Animal ID '{animal_id}'. Must be exactly 6 alphanumeric characters."
        )

    session = SessionLocal()
    try:
        patient = session.query(Patient).filter(Patient.animal_id == animal_id).first()

        if not patient:
            raise ValueError(
                f"No patient found with Animal ID '{animal_id}'. "
                "Please verify the ID or register as a new patient."
            )

        visit = Visit(
            animal_id=animal_id,
            diagnosis=diagnosis,
            treatment=treatment,
            doctor_notes=doctor_notes,
            weight_kg=weight_kg,
            visit_date=_parse_date(visit_date),
            doctor_name=doctor_name,
        )
        session.add(visit)
        session.commit()

        logger.info(
            "[DB] Visit added | animal_id=%s | visit_id=%s | doctor=%s",
            animal_id, visit.visit_id[:8], doctor_name,
        )
        return {
            "animal_id":   animal_id,
            "animal_name": patient.animal_name,
            "visit_id":    visit.visit_id,
        }

    except ValueError:
        session.rollback()
        raise
    except Exception as exc:
        session.rollback()
        logger.error("[DB] add_visit_to_existing failed: %s", exc, exc_info=True)
        raise
    finally:
        session.close()


# ── Read: Patient History ─────────────────────────────────────────────────────

def get_patient_history(animal_id: str) -> Optional[dict]:
    """
    Retrieve complete medical history for a patient by Animal ID.

    Returns:
        Dict with patient info and per-visit records (including weight per visit),
        or None if not found.
    """
    animal_id = animal_id.strip().upper()

    session = SessionLocal()
    try:
        patient = session.query(Patient).filter(Patient.animal_id == animal_id).first()

        if not patient:
            return None

        visits_data = []
        for v in patient.visits:
            visits_data.append({
                "visit_id":      v.visit_id,
                "diagnosis":     v.diagnosis,
                "treatment":     v.treatment,
                "doctor_notes":  v.doctor_notes,
                "weight_kg":     v.weight_kg,
                "visit_date":    v.visit_date.isoformat() if v.visit_date else "N/A",
                "doctor_name":   v.doctor_name,
            })

        return {
            "animal_id":    patient.animal_id,
            "animal_name":  patient.animal_name,
            "animal_type":  patient.animal_type,
            "owner_name":   patient.owner_name,
            "doctor_name":  patient.doctor_name,
            "created_at":   patient.created_at.isoformat() if patient.created_at else "N/A",
            "total_visits": len(visits_data),
            "visits":       visits_data,
        }

    except Exception as exc:
        logger.error("[DB] get_patient_history failed: %s", exc, exc_info=True)
        raise
    finally:
        session.close()


def format_patient_history(animal_id: str) -> str:
    """
    Format patient history as a human-readable string for LLM consumption.
    """
    data = get_patient_history(animal_id)

    if data is None:
        return f"No patient found with Animal ID: {animal_id}"

    lines = [
        "═══ Patient Record ═══",
        f"  Name:        {data['animal_name']}",
        f"  Type:        {data['animal_type']}",
        f"  Owner:       {data['owner_name']}",
        f"  Animal ID:   {data['animal_id']}",
        f"  Doctor:      {data['doctor_name']}",
        f"  Registered:  {data['created_at']}",
        "",
        f"═══ Visit History ({data['total_visits']} visit(s)) ═══",
    ]

    if not data["visits"]:
        lines.append("  No visits recorded.")
    else:
        for i, v in enumerate(data["visits"], 1):
            weight_str = f"{v['weight_kg']} kg" if v["weight_kg"] is not None else "N/A"
            notes_str = v['doctor_notes'] if v.get('doctor_notes') else "None"
            lines.extend([
                "",
                f"  ── Visit {i} ({v['visit_date']}) ──",
                f"  Weight:     {weight_str}",
                f"  Diagnosis:  {v['diagnosis']}",
                f"  Treatment:  {v['treatment']}",
                f"  Notes:      {notes_str}",
                f"  Doctor:     {v['doctor_name']}",
            ])

    return "\n".join(lines)
