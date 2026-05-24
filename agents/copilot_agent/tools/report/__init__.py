"""
agents/copilot_agent/tools/report/__init__.py — Report Generation Package
===========================================================================
Exports the generate_patient_report tool for the tool registry.
"""

from agents.copilot_agent.tools.report.tool import generate_patient_report, REPORTS_DIR

__all__ = ["generate_patient_report", "REPORTS_DIR"]
