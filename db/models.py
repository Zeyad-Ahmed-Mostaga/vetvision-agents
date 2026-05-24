"""
db/models.py — SQLAlchemy Models
=================================
Patient + Visit tables for veterinary medical records.

Design decisions:
  - Animal ID is a 6-character alphanumeric string (uppercase letters + digits),
    e.g. "A3X7K9". Generated randomly with collision checking.
  - weight_kg lives on Visit (not Patient) so weight changes are tracked over time.
  - doctor_name is stored as free text (no doctor_id anywhere in this system).
  - Two tables: Patient (static identity) + Visit (per-visit clinical data).
"""

import random
import string
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Text, Date, DateTime,
    ForeignKey, create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from config import settings

Base = declarative_base()

# ── ID Generators ─────────────────────────────────────────────────────────────

_ID_CHARS = string.ascii_uppercase + string.digits  # A-Z + 0-9


def _new_animal_id() -> str:
    """Generate a random 6-character alphanumeric Animal ID (e.g. 'A3X7K9')."""
    return "".join(random.choices(_ID_CHARS, k=6))


def _new_visit_id() -> str:
    """Generate a UUID-style visit ID."""
    import uuid
    return str(uuid.uuid4())


# ── Models ────────────────────────────────────────────────────────────────────

class Patient(Base):
    __tablename__ = "patients"

    animal_id   = Column(String(6),   primary_key=True, default=_new_animal_id)
    animal_name = Column(String(200), nullable=False)
    animal_type = Column(String(100), nullable=False)   # free text, exactly as doctor writes
    owner_name  = Column(String(200), nullable=False)
    doctor_name = Column(String(200), nullable=False)   # treating doctor's name
    created_at  = Column(DateTime,   default=datetime.utcnow)

    visits = relationship("Visit", back_populates="patient", order_by="Visit.visit_date")

    def __repr__(self):
        return f"<Patient {self.animal_id} | {self.animal_name} ({self.animal_type}) owner={self.owner_name}>"


class Visit(Base):
    __tablename__ = "visits"

    visit_id    = Column(String(36),  primary_key=True, default=_new_visit_id)
    animal_id   = Column(String(6),   ForeignKey("patients.animal_id"), nullable=False)
    diagnosis   = Column(Text,        nullable=False)
    treatment   = Column(Text,        nullable=False)
    weight_kg   = Column(Float,       nullable=True)    # weight tracked per visit
    visit_date  = Column(Date,        nullable=False)
    doctor_name = Column(String(200), nullable=False)   # doctor who performed this visit
    created_at  = Column(DateTime,   default=datetime.utcnow)

    patient = relationship("Patient", back_populates="visits")

    def __repr__(self):
        return f"<Visit {self.visit_id[:8]} date={self.visit_date} animal_id={self.animal_id}>"


# ── Engine & Session Factory ─────────────────────────────────────────────────
# check_same_thread=False is safe with SQLAlchemy's connection pooling
_engine = create_engine(
    f"sqlite:///{settings.sqlite_db_path}",
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)

# Create tables on import (idempotent)
Base.metadata.create_all(bind=_engine)
