GRADING_SYSTEM = (
    "You are a strict answer quality evaluator. "
    "Respond with exactly one word: PASS or FAIL. "
    "PASS means the answer directly and correctly addresses the question. "
    "FAIL means the answer is off-topic, evasive, or says it doesn't know when the context should allow an answer."
)

HALLUCINATION_SYSTEM = (
    "You are a hallucination detector. "
    "Respond with exactly one word: GROUNDED or HALLUCINATED. "
    "GROUNDED means every factual claim in the answer can be traced to the provided context. "
    "HALLUCINATED means the answer contains facts, numbers, or claims not present in the context."
)


def build_grading_prompt(question: str, answer: str) -> str:
    return f"Question: {question}\n\nAnswer: {answer}\n\nDoes this answer directly address the question? Reply PASS or FAIL only."


def build_hallucination_prompt(context: str, answer: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Is every claim in the answer supported by the context above? Reply GROUNDED or HALLUCINATED only."
    )
