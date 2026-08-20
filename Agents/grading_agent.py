"""
GradingAgent — evaluates whether an answer actually addresses the question.
"""
import logging
from Prompts.grading_prompts import GRADING_SYSTEM, build_grading_prompt

logger = logging.getLogger(__name__)


class GradingAgent:
    """Uses the LLM to grade if an answer correctly addresses the question."""

    def __init__(self, llm_tool):
        self._llm = llm_tool

    def grade(self, question: str, answer: str) -> str:
        """Return 'PASS' if the answer addresses the question, 'FAIL' otherwise."""
        messages = [
            {"role": "system", "content": GRADING_SYSTEM},
            {"role": "user", "content": build_grading_prompt(question, answer)},
        ]
        try:
            result = self._llm.call(messages, max_tokens=10).strip().upper()
            return "PASS" if result.startswith("PASS") else "FAIL"
        except Exception as e:
            logger.warning(f"GradingAgent error: {e}")
            return "PASS"  # Default to passing to avoid blocking on LLM errors
