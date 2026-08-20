"""
WorkflowState — shared state TypedDict passed between all LangGraph nodes.
"""
from typing import Any, List, Optional
from typing_extensions import TypedDict


class WorkflowState(TypedDict, total=False):
    # Input
    query: str
    memory: Any                      # ConversationMemory instance
    n_results: int
    temperature: float
    source_filter: Optional[List[str]]

    # Routing
    route: str                       # "table" | "doc" | "both" | "chitchat"

    # Table path
    sql_query: str
    table_answer: str                # SQL answer held for hybrid merge

    # Answer assembly
    answer: str
    sources: List[str]
    chunks_used: int
    answer_method: str               # "table_query" | "rag" | "chitchat"
    context: str                     # raw retrieved context for hallucination check

    # Evaluation
    grade: str                       # "PASS" | "FAIL"
    hallucinated: bool
