"""
agents/copilot_agent/tools/web_search.py — Tavily Web Search Tool
===================================================================
Uses LangChain's built-in TavilySearch for web search and fallback.
"""

from langchain_tavily import TavilySearch
from config import settings

tavily_search = TavilySearch(
    max_results=4,
    topic="general",
    include_answer=True,
    tavily_api_key=settings.tavily_api_key,
    description=(
        "Web search tool. Use this for: "
        "1. Fallback when 'vet_rag_search' fails or returns irrelevant results. "
        "2. Finding medication/treatment prices in Egypt (e.g., 'سعر أموكسيسيلين في مصر'). "
        "3. Looking up specific commercial medications, dosages, or current real-world data. "
        "4. Any veterinary question not covered by the knowledge base. "
        "RULES: "
        "- For Egypt-specific searches (prices, clinics), formulate the query in Arabic. "
        "- For medical/scientific queries, use English. "
        "- Rephrase the input into a concise, keyword-rich search query."
    ),
)
