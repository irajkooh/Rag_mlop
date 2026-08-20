"""
SupervisorAgent — thin wrapper delegating to RAGWorkflow (LangGraph).
"""
from typing import List, Optional

from .rag_workflow import RAGWorkflow


class SupervisorAgent:
    """Thin facade that preserves the existing backend.py call signature."""

    def __init__(
        self,
        router_agent,
        sql_gen_agent,
        table_agent,
        doc_image_agent,
        grading_agent,
        hallucination_agent,
        vector_search_tool,
    ):
        self._workflow = RAGWorkflow(
            router_agent,
            sql_gen_agent,
            table_agent,
            doc_image_agent,
            grading_agent,
            hallucination_agent,
            vector_search_tool,
        )

    def handle(
        self,
        question: str,
        memory,
        n_results: int = 8,
        temperature: float = 0.0,
        source_filter: Optional[List[str]] = None,
    ) -> dict:
        return self._workflow.invoke(question, memory, n_results, temperature, source_filter)
