TABLE_ANALYST_SYSTEM = (
    "You are a helpful data analyst. Answer in one or two clear sentences based on the SQL results. "
    "Use the question to frame the answer naturally (e.g. 'You spent $X on groceries in April.'). "
    "Report numbers exactly as returned by SQL — do not round, negate, or recalculate them."
)

LLM_TABLE_EXTRACT_PROMPT = (
    "The following is text extracted from a document. "
    "If it contains a table, output ONLY a valid CSV representation — "
    "no explanation, no markdown fences, just raw CSV rows. "
    "Use clean column names in the header row. Strip leading row/line numbers. "
    "If no table is present, output exactly: NO_TABLE\n\n"
)


def build_table_answer_prompt(question: str, sql: str, result_str: str, detail_str: str = "") -> str:
    return (
        f"Question: {question}\n\n"
        f"SQL: {sql}\n\n"
        f"Results:\n{result_str}"
        f"{detail_str}\n\n"
        "Report the exact values from the results. Do not round, negate, or recalculate any numbers."
    )
