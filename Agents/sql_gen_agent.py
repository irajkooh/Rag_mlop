"""
SQLGenAgent — converts natural language questions to SQLite SQL and executes them.
"""
import logging
import re as _re
from pathlib import Path
from typing import Optional, Tuple, List

from Prompts.sql_prompts import build_sql_prompt, SQL_SYSTEM
from Tools.sql_execution_tool import strip_sql_fences, execute_sql

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}

_STOPWORDS = {
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "shall", "may",
    "might", "must", "can", "the", "a", "an", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "by", "from", "as", "if",
    "this", "that", "these", "those", "it", "its", "what", "which", "who",
    "how", "when", "where", "why", "all", "any", "some", "such", "no",
    "than", "too", "very", "just", "also", "so", "me", "my", "give",
}


class SQLGenAgent:
    """Selects relevant tables, generates SQL via LLM, executes, and returns results."""

    def __init__(self, llm_tool):
        self._llm = llm_tool

    _MAX_TABLES_IN_PROMPT = 5

    def select_relevant_tables(self, question: str, schema_info: list) -> list:
        """Return the most relevant table(s) based on keyword overlap, capped at _MAX_TABLES_IN_PROMPT."""
        if len(schema_info) <= 1:
            return schema_info
        q_words = set(_re.sub(r"[^a-z0-9]", " ", question.lower()).split()) - _STOPWORDS
        scores = []
        for s in schema_info:
            col_words = set(_re.sub(r"[^a-z0-9]", " ", " ".join(s["numeric_cols"] + s["text_cols"]).lower()).split())
            sample_words = set(_re.sub(r"[^a-z0-9]", " ", s["sample"].lower()).split())
            score = len(q_words & col_words) + 0.3 * len(q_words & sample_words)
            # Image tables are actual data tables; boost their score over OCR-heavy PDFs
            if Path(s["source"]).suffix.lower() in _IMAGE_EXTS:
                score *= 1.5
            scores.append(score)
        max_score = max(scores)
        ranked = sorted(zip(scores, schema_info), key=lambda x: -x[0])
        if max_score == 0:
            # No keyword overlap — return top candidates sorted by image-first priority
            return [s for _, s in ranked][: self._MAX_TABLES_IN_PROMPT]
        if len(ranked) >= 2 and ranked[0][0] >= 2 * ranked[1][0] + 0.01:
            return [ranked[0][1]]
        candidates = [s for sc, s in ranked if sc >= max_score * 0.5]
        return candidates[: self._MAX_TABLES_IN_PROMPT]

    def generate_and_execute(
        self,
        question: str,
        schema_info: list,
        conn,
    ) -> Optional[Tuple[str, List, List[str]]]:
        """Generate SQL, execute it, retry once on error.
        Returns (sql, rows, col_names) or None."""
        relevant = self.select_relevant_tables(question, schema_info)
        if not relevant:
            return None

        sql_prompt = build_sql_prompt(question, relevant)
        messages = [
            {"role": "system", "content": SQL_SYSTEM},
            {"role": "user", "content": sql_prompt},
        ]

        for attempt in range(2):
            try:
                llm_out = self._llm.call(messages, max_tokens=512)
            except Exception as e:
                logger.warning(f"LLM SQL generation failed: {e}")
                return None
            if not llm_out:
                return None
            sql = strip_sql_fences(llm_out.strip())
            result = execute_sql(sql, conn)
            if result is not None:
                return result
            logger.warning(f"SQL exec failed (attempt {attempt + 1}): SQL: {sql}")
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": sql},
                    {"role": "user", "content": "That SQL failed. Return only the corrected SQL query."},
                ]

        return None
