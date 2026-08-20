"""
HallucinationAgent — checks whether an answer is grounded in the retrieved context.
"""
import logging
from Prompts.grading_prompts import HALLUCINATION_SYSTEM, build_hallucination_prompt

logger = logging.getLogger(__name__)


class HallucinationAgent:
    """Uses the LLM to detect if an answer contains claims not present in the context."""

    def __init__(self, llm_tool):
        self._llm = llm_tool

    def check(self, context: str, answer: str) -> bool:
        """Return True if the answer is hallucinated (not grounded in context)."""
        if not context or not answer:
            return False
        messages = [
            {"role": "system", "content": HALLUCINATION_SYSTEM},
            {"role": "user", "content": build_hallucination_prompt(context, answer)},
        ]
        try:
            result = self._llm.call(messages, max_tokens=10).strip().upper()
            return result.startswith("HALLUCINATED")
        except Exception as e:
            logger.warning(f"HallucinationAgent error: {e}")
            return False  # Default to not hallucinated to avoid blocking on LLM errors
