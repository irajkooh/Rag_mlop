"""
DocImageAgent — retrieves answers from documents and images using RAG.
"""
import logging
import re
from collections import defaultdict
from typing import List, Optional, Tuple

from Prompts.rag_prompts import RELEVANCE_THRESHOLD

# Matches the person-name prefix that pronoun expansion prepends: "Iraj Koohi: ..."
_PERSON_PREFIX_RE = re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*:')

logger = logging.getLogger(__name__)

_SUMMARIZE_ALL_RE = re.compile(
    r"\b(?:summarize|summarise)\b.{0,50}\b(?:each|all|every)\b.{0,30}\b(?:doc|document|file)\b"
    r"|\beach\b.{0,30}\b(?:doc|document|file)\b.{0,50}\b(?:summarize|summarise|summary|overview)\b",
    re.IGNORECASE,
)


class DocImageAgent:
    """Retrieves and answers questions from indexed documents/images via RAG."""

    def __init__(self, rag_engine, vector_search_tool):
        self._rag = rag_engine
        self._vs_tool = vector_search_tool

    def run(
        self,
        question: str,
        memory,
        n_results: int = 8,
        temperature: float = 0.0,
        source_filter: Optional[List[str]] = None,
        table_answer: str = "",
    ) -> Tuple[str, List[str], int, str]:
        """Answer question from docs/images. Returns (answer, sources, chunks_used, context).

        table_answer: SQL result to inject as extra context for hybrid queries.
        context: the raw text context sent to the LLM (used by hallucination checker).
        """
        if not source_filter and not table_answer and _SUMMARIZE_ALL_RE.search(question):
            return self._summarize_all_docs(question, memory)

        if not source_filter:
            results = self._vs_tool.search_per_source(question, n_per_source=4)
        else:
            results = self._vs_tool.search(question, n_results=n_results, source_filter=source_filter)

        relevant = [r for r in results if r.get("distance", 2.0) <= RELEVANCE_THRESHOLD]
        chunks_used = len(relevant)

        best_per_src: dict = {}
        if relevant:
            best_dist = min(r.get("distance", 2.0) for r in relevant)
            # Build per-source best distance so we only attribute sources whose
            # closest chunk is near the overall best — prevents unrelated sources
            # from appearing just because one of their chunks scraped past threshold.
            for r in relevant:
                src = r["metadata"].get("source", "")
                d = r.get("distance", 2.0)
                if src not in best_per_src or d < best_per_src[src]:
                    best_per_src[src] = d
            source_chunks = [
                r for r in relevant
                if best_per_src.get(r["metadata"].get("source", ""), 2.0) <= best_dist + 0.10
            ]
        else:
            source_chunks = results if not source_filter else []

        # Send only the best-source chunks to the LLM — source_chunks are already
        # filtered to sources whose closest chunk is within best_dist + 0.10, so
        # unrelated sources (IBC, standards) won't drown out the actual answer.
        llm_results = source_chunks if source_chunks else (relevant if relevant else results)

        # Person-name filter: when the query was pronoun-expanded (starts with "Name: ..."),
        # keep only chunks that explicitly mention that person. This prevents img_1.png
        # (which contains multiple people) from returning another person's data as Iraj's.
        m = _PERSON_PREFIX_RE.match(question)
        if m:
            first_name = m.group(1).split()[0].lower()
            person_chunks = [r for r in llm_results if first_name in r["text"].lower()]
            if person_chunks:
                llm_results = person_chunks

        context = self._rag._build_context(llm_results)

        parts = []
        for token in self._rag.query(
            question,
            memory,
            n_results=n_results,
            temperature=temperature,
            stream=False,
            source_filter=source_filter,
            pre_fetched_results=llm_results,
            extra_context=table_answer,
        ):
            parts.append(token)
        answer = "".join(parts)

        # Source attribution.
        # Primary: map "Doc N" numbers in the answer back to their source file.
        # _build_context numbers chunks sequentially, skipping question-only and
        # non-content chunks — replicate that numbering here.
        doc_to_src: dict = {}
        _idx = 1
        for r in llm_results:
            _txt = r["text"]
            if self._rag._is_question_only(_txt) or self._rag._is_non_content_chunk(_txt):
                continue
            doc_to_src[_idx] = r["metadata"].get("source", "")
            _idx += 1

        doc_refs = re.findall(r'\bDoc\s+(\d+)\b', answer, re.IGNORECASE)
        cited_by_num = list({doc_to_src[int(n)] for n in doc_refs if int(n) in doc_to_src})

        # Fallback: filename substring in answer text (when LLM cites by name).
        candidate_sources = list({r["metadata"].get("source", "") for r in source_chunks})
        cited_by_name = [s for s in candidate_sources if s in answer]

        if not source_filter:
            cited = cited_by_num or cited_by_name
            if cited:
                sources = cited
            elif best_per_src:
                # No explicit citation — attribute only the single closest source
                # to avoid unrelated files (IBC, word.docx) being shown when the
                # LLM gives a brief answer with no Doc-N reference.
                sources = [min(best_per_src, key=best_per_src.get)]
            else:
                sources = candidate_sources
        else:
            sources = candidate_sources

        return answer, sources, chunks_used, context

    def _summarize_all_docs(self, question: str, memory) -> Tuple[str, List[str], int, str]:
        """Fetch chunks grouped by source and ask the LLM to summarize each document by name."""
        from Tools.llm_tool import LLMTool
        from utils.rag_engine import SYSTEM_PROMPT

        sources = list(self._vs_tool.list_sources())
        if not sources:
            return "No documents are indexed yet.", [], 0, ""

        all_results = self._vs_tool.search_per_source(question, n_per_source=4)

        by_source: dict = defaultdict(list)
        for r in all_results:
            src = r["metadata"].get("source", "unknown")
            text = self._rag._fix_ocr(r["text"])
            if not self._rag._is_question_only(text) and not self._rag._is_non_content_chunk(text):
                by_source[src].append(text)

        context_parts = []
        for src in sources:
            chunks = by_source.get(src, [])
            body = "\n\n".join(chunks) if chunks else "(no text content available)"
            context_parts.append(f"[{src}]\n{body}")
        context = "\n\n---\n\n".join(context_parts)

        source_list = "\n".join(f"- {s}" for s in sources)
        user_message = (
            f"[CONTEXT from documents — grouped by document name]\n{context}\n\n"
            f"[QUESTION]\n{question}\n\n"
            f"Summarize EACH of the following {len(sources)} documents using ONLY the excerpts "
            f"provided above. Use the exact document name as a bold header for each section. "
            f"If a document has no text content available, say so briefly.\n\n"
            f"Documents to summarize:\n{source_list}"
        )

        # Exclude conversation history — prior questions bleed topic context into
        # summaries (e.g. a previous "occupant load" Q causes all docs to be
        # summarized as if they're about occupant loads).
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        llm = LLMTool(self._rag)
        answer = llm.call(messages, max_tokens=4096)

        memory.add("user", question)
        memory.add("assistant", answer)

        return answer, sources, len(all_results), context
