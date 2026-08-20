"""
RAGWorkflow — LangGraph StateGraph that orchestrates all RAG agents.

Graph flow:
    START
      → chitchat_detector
        ├─ "chitchat"   → END
        └─ "continue"   → router
                           ├─ "table"  → table_agent
                           │               ├─ (answer set) → grader
                           │               └─ (no answer)  → doc_image_agent
                           └─ "doc"    → doc_image_agent
                                            → grader
                                              → hallucination_checker
                                                → END
"""
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional

from langgraph.graph import StateGraph, START, END

from .workflow_state import WorkflowState

logger = logging.getLogger(__name__)

_PRONOUN_RE     = re.compile(r'\b(he|she|his|her|him|they|their)\b', re.IGNORECASE)
_PROPER_NAME_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')


def _expand_pronoun_query(question: str, memory) -> str:
    """If question contains a person-referencing pronoun, prepend the most recently discussed person name."""
    if not _PRONOUN_RE.search(question) or not memory:
        return question
    for msg in reversed(memory.get_history_for_prompt()):
        if msg["role"] == "assistant":
            names = _PROPER_NAME_RE.findall(msg["content"])
            if names:
                return f"{names[0]}: {question}"
    return question

# ── Chitchat / meta data ──────────────────────────────────────────────────────

_GREETING_RESPONSES = {
    "hi": "Hi! Ask me anything about your uploaded documents.",
    "hello": "Hello! Ask me anything about your uploaded documents.",
    "hey": "Hey! Ask me anything about your uploaded documents.",
    "hiya": "Hi there! Ask me anything about your uploaded documents.",
    "howdy": "Howdy! Ask me anything about your uploaded documents.",
    "greetings": "Greetings! Ask me anything about your uploaded documents.",
    "sup": "Hey! Ask me anything about your uploaded documents.",
    "yo": "Hey! Ask me anything about your uploaded documents.",
    "good morning": "Good morning! Ask me anything about your uploaded documents.",
    "good afternoon": "Good afternoon! Ask me anything about your uploaded documents.",
    "good evening": "Good evening! Ask me anything about your uploaded documents.",
    "good day": "Good day! Ask me anything about your uploaded documents.",
    "how are you": "I'm doing well, thank you! Ask me anything about your uploaded documents.",
    "how are you doing": "I'm doing well, thank you! Ask me anything about your uploaded documents.",
    "how do you do": "I'm doing well, thank you! How can I help with your documents?",
    "what's up": "Not much! Ready to answer questions about your documents.",
    "whats up": "Not much! Ready to answer questions about your documents.",
    "what is up": "Not much! Ready to answer questions about your documents.",
    "thanks": "You're welcome! Let me know if you have more questions.",
    "thank you": "You're welcome! Let me know if you have more questions.",
    "thx": "You're welcome! Let me know if you have more questions.",
    "ty": "You're welcome! Let me know if you have more questions.",
    "bye": "Goodbye! Feel free to come back anytime.",
    "goodbye": "Goodbye! Feel free to come back anytime.",
    "see you": "See you! Feel free to come back anytime.",
    "cya": "See you later! Feel free to come back anytime.",
    "ok": "Let me know if you have any questions about your documents.",
    "okay": "Let me know if you have any questions about your documents.",
    "cool": "Glad to help! Let me know if you have more questions.",
    "great": "Glad to help! Let me know if you have more questions.",
    "nice": "Thanks! Let me know if you have more questions.",
}

_META_PATTERNS = [
    "how can you help", "how you can help", "what can you do",
    "what do you do", "who are you", "what are you",
    "help me", "how does this work", "how do you work",
]

_META_ANSWER = (
    "I'm your document assistant. Here's how I can help:\n\n"
    "1. **Upload documents** (PDF, Word, Excel, CSV, TXT, images) or **add URLs** in the Documents tab\n"
    "2. **Ask questions** about your uploaded documents and I'll answer based on their content\n"
    "3. I can handle text, tables, charts, and scanned images\n"
    "4. Use the **Read** button to hear answers aloud\n\n"
    "Upload some documents and start asking questions!"
)

_DOCS_LIST_PATTERNS = [
    "how many doc", "how many file", "list doc", "list file",
    "what doc", "what file", "which doc", "which file",
    "show doc", "show file", "what are the doc", "what are the file",
    "what is indexed", "what is uploaded", "what have you indexed",
    "what have you uploaded", "what documents do you have",
    "what files do you have", "tell me the doc", "tell me the file",
    "name the doc", "name the file",
    "how many docx", "how many xlsx", "how many csv", "how many pdf",
    "how many image", "how many txt", "how many png", "how many jpg",
    "list docx", "list xlsx", "list csv", "list pdf", "list image", "list txt",
    "show docx", "show xlsx", "show csv", "show pdf", "show image", "show txt",
    "what docx", "what xlsx", "what csv", "what pdf",
    "which docx", "which xlsx", "which csv", "which pdf",
    "any docx", "any xlsx", "any csv", "any pdf", "any image",
    "excel file", "word file", "spreadsheet", "word document",
]


class RAGWorkflow:
    """LangGraph-based orchestrator for the multimodal RAG agentic workflow."""

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
        self._router = router_agent
        self._sql_gen = sql_gen_agent
        self._table = table_agent
        self._doc_image = doc_image_agent
        self._grader = grading_agent
        self._hallucination = hallucination_agent
        self._vs_tool = vector_search_tool
        self._graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(WorkflowState)

        g.add_node("chitchat_detector", self._chitchat_node)
        g.add_node("router", self._router_node)
        g.add_node("table_agent", self._table_node)
        g.add_node("doc_image_agent", self._doc_image_node)
        g.add_node("grader", self._grader_node)
        g.add_node("hallucination_checker", self._hallucination_node)

        g.add_edge(START, "chitchat_detector")
        g.add_conditional_edges(
            "chitchat_detector",
            self._after_chitchat,
            {"chitchat": END, "continue": "router"},
        )
        g.add_conditional_edges(
            "router",
            self._after_router,
            {"table": "table_agent", "doc": "doc_image_agent", "both": "table_agent"},
        )
        g.add_conditional_edges(
            "table_agent",
            self._after_table,
            {"grader": "grader", "doc_image_agent": "doc_image_agent"},
        )
        g.add_edge("doc_image_agent", "grader")
        g.add_edge("grader", "hallucination_checker")
        g.add_edge("hallucination_checker", END)

        return g.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _chitchat_node(self, state: WorkflowState) -> dict:
        query = state["query"]
        normalized = query.strip().lower().rstrip("!?.,")

        if normalized in _GREETING_RESPONSES:
            return {
                "route": "chitchat",
                "answer": _GREETING_RESPONSES[normalized],
                "sources": [],
                "sql_query": "",
                "answer_method": "chitchat",
                "chunks_used": 0,
                "grade": "PASS",
                "hallucinated": False,
            }

        for pattern in _META_PATTERNS:
            if pattern in normalized:
                return {
                    "route": "chitchat",
                    "answer": _META_ANSWER,
                    "sources": [],
                    "sql_query": "",
                    "answer_method": "chitchat",
                    "chunks_used": 0,
                    "grade": "PASS",
                    "hallucinated": False,
                }

        if any(p in normalized for p in _DOCS_LIST_PATTERNS):
            answer = self._build_docs_list() or "No documents are indexed yet. Please upload some documents first."
            return {
                "route": "chitchat",
                "answer": answer,
                "sources": [],
                "sql_query": "",
                "answer_method": "chitchat",
                "chunks_used": 0,
                "grade": "PASS",
                "hallucinated": False,
            }

        if self._vs_tool.total_chunks() == 0:
            return {
                "route": "chitchat",
                "answer": "No documents are indexed yet. Please upload some documents first.",
                "sources": [],
                "sql_query": "",
                "answer_method": "chitchat",
                "chunks_used": 0,
                "grade": "PASS",
                "hallucinated": False,
            }

        return {"route": "continue"}

    def _router_node(self, state: WorkflowState) -> dict:
        expanded = _expand_pronoun_query(state["query"], state.get("memory"))
        route = self._router.route(expanded)
        return {"route": route, "query": expanded}

    def _table_node(self, state: WorkflowState) -> dict:
        result = self._table.run(state["query"], self._sql_gen, state.get("source_filter"))
        is_hybrid = state.get("route") == "both"

        if result is None:
            return {}  # answer stays unset → conditional edge falls through to doc_image_agent

        table_answer, table_sql, table_sources = result
        sources = state.get("source_filter") or table_sources

        if is_hybrid:
            # Hold the SQL result; final answer comes from RAG after merging both sources.
            return {
                "table_answer": table_answer,
                "sql_query": table_sql,
                "sources": sources,
            }

        memory = state.get("memory")
        if memory is not None:
            memory.add("user", state["query"])
            memory.add("assistant", f"{table_answer}\n\n[SQL used: {table_sql}]")
        return {
            "answer": table_answer,
            "sql_query": table_sql,
            "sources": sources,
            "chunks_used": 0,
            "answer_method": "table_query",
            "context": "",
        }

    def _doc_image_node(self, state: WorkflowState) -> dict:
        table_answer = state.get("table_answer", "")
        answer, doc_sources, chunks_used, context = self._doc_image.run(
            state["query"],
            state.get("memory"),
            n_results=state.get("n_results", 8),
            temperature=state.get("temperature", 0.0),
            source_filter=state.get("source_filter"),
            table_answer=table_answer,
        )
        is_hybrid = bool(table_answer)
        sources = list({*(state.get("sources") or []), *doc_sources}) if is_hybrid else doc_sources
        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": chunks_used,
            "answer_method": "hybrid" if is_hybrid else "rag",
            "sql_query": state.get("sql_query", ""),
            "context": context,
        }

    def _grader_node(self, state: WorkflowState) -> dict:
        try:
            grade = self._grader.grade(state["query"], state.get("answer", ""))
        except Exception:
            grade = "PASS"
        return {"grade": grade}

    def _hallucination_node(self, state: WorkflowState) -> dict:
        try:
            context = state.get("context", "")
            hallucinated = self._hallucination.check(context, state.get("answer", "")) if context else False
        except Exception:
            hallucinated = False
        return {"hallucinated": hallucinated}

    # ── Conditional edge functions ────────────────────────────────────────────

    @staticmethod
    def _after_chitchat(state: WorkflowState) -> str:
        return "chitchat" if state.get("route") in ("chitchat", None) else "continue"

    @staticmethod
    def _after_router(state: WorkflowState) -> str:
        route = state.get("route")
        if route in ("table", "both"):
            return "both" if route == "both" else "table"
        return "doc"

    @staticmethod
    def _after_table(state: WorkflowState) -> str:
        if state.get("route") == "both":
            return "doc_image_agent"  # always run RAG too for hybrid queries
        return "grader" if state.get("answer") else "doc_image_agent"

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(
        self,
        question: str,
        memory,
        n_results: int = 8,
        temperature: float = 0.0,
        source_filter: Optional[List[str]] = None,
    ) -> dict:
        initial: WorkflowState = {
            "query": question,
            "memory": memory,
            "n_results": n_results,
            "temperature": temperature,
            "source_filter": source_filter,
        }
        final = self._graph.invoke(initial)
        return {
            "answer": final.get("answer", ""),
            "sources": final.get("sources", []),
            "sql_query": final.get("sql_query", ""),
            "answer_method": final.get("answer_method", "rag"),
            "chunks_used": final.get("chunks_used", 0),
            "grade": final.get("grade", "PASS"),
            "hallucinated": final.get("hallucinated", False),
        }

    def get_mermaid(self) -> str:
        return self._graph.get_graph().draw_mermaid()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_docs_list(self) -> Optional[str]:
        sources = self._vs_tool.list_sources()
        if not sources:
            return None
        lines = "\n".join(f"- {s}" for s in sources)
        ext_counts = Counter(Path(s).suffix.lower() for s in sources)
        breakdown = ", ".join(f"{cnt} {ext}" for ext, cnt in sorted(ext_counts.items()))
        return (
            f"There are **{len(sources)}** indexed document(s) ({breakdown}):\n\n"
            f"{lines}"
        )
