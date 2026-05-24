"""
agents/copilot_agent/tools/__init__.py — Tool Registry
========================================================
Exports ALL_TOOLS list used by the graph builder.

Patient record tools:
  - register_first_visit  (WRITE — new patient only)
  - log_returning_visit   (WRITE — returning patient, requires existing Animal ID)
  - get_patient_history   (READ-ONLY)

Knowledge & search tools:
  - vet_rag_search        (searches VetVision knowledge base)
  - tavily_search         (web search, Egypt-aware)

Report tool:
  - generate_patient_report  (Playwright-based bilingual PDF)
"""

from agents.copilot_agent.tools.vet_rag import vet_rag_search
from agents.copilot_agent.tools.web_search import tavily_search
from agents.copilot_agent.tools.patient_records import (
    register_first_visit,
    log_returning_visit,
    get_patient_history,
)
from agents.copilot_agent.tools.report import generate_patient_report

ALL_TOOLS = [
    vet_rag_search,
    tavily_search,
    register_first_visit,
    log_returning_visit,
    get_patient_history,
    generate_patient_report,
]
