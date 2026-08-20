"""
RouterAgent — decides whether a question should be answered via SQL/tables, RAG, or both.
"""
import re

_TABLE_INTENT_RE = re.compile(
    r"\b(sum|total|average|avg|mean|maximum|minimum|count|how much|how many|"
    r"calculate|tallest|largest|smallest|highest|lowest|most|least|"
    r"(?:max|min)(?!\s+\d)|"
    r"per (month|year|day|week|item|person|category)|"
    r"sell|sells|sold|selling|"
    r"(sales|revenue|profit|cost|price|amount|balance|credit|debit|spendings?|paid|owe)\b.*\b(of|for|by|in|per)\b|"
    r"\b(of|for|by|in)\b.*\b(sales|revenue|profit|cost|price|amount|balance|credit|debit|spendings?)|"
    r"\bwhich\b.{0,40}\b(is|are|was|were|has|have|had)\b|"
    r"\b(list|show|give me|find|get|fetch|return|display)\b.{0,40}\b(name|rep|person|contact|manager|owner|agent|employee|staff|client|customer|vendor)\b)\b",
    re.IGNORECASE,
)

# Specifically financial/business domain — used to distinguish hybrid vs doc-only routing.
_FINANCIAL_RE = re.compile(
    r"\b(sales|revenue|profit|cost|price|amount|balance|credit|debit|spendings?|paid|owe|"
    r"invoice|purchase|transaction|income|expense|salary|wage|fee|"
    r"sell|sells|sold|selling)\b",
    re.IGNORECASE,
)

# Signals that the answer likely lives in a document (building codes, regulations, procedures).
# Note: \boccupan has no trailing \b so it matches occupant/occupancy/occupants.
_DOC_SIGNAL_RE = re.compile(
    r"(?:"
    r"\b(?:requirement|regulation|code|standard|procedure|guideline|specification|"
    r"compliance|ordinance|statute|policy|provision|"
    r"ADA|IBC|ASCE|OSHA|NFPA|"
    r"egress|sprinkler|handrail|clearance|fire.?rating|"
    r"accessible|accessibility|"
    r"stairway|stair|tread|riser|rise|slope|ramp|"
    r"ceiling|habitable|corridor|setback|"
    r"exit|smoke|alarm|suppression|"
    r"construct|install|design)\b"
    r"|\boccupan"  # matches occupant, occupancy, occupants
    r")",
    re.IGNORECASE,
)


class RouterAgent:
    """Routes a question to 'table' (SQL), 'doc' (RAG), or 'both'."""

    def is_table_question(self, question: str) -> bool:
        """Return True if the question asks for quantitative, analytical, or lookup data from tables."""
        return bool(_TABLE_INTENT_RE.search(question))

    def route(self, question: str) -> str:
        """Return 'table', 'doc', or 'both'.

        Doc signals win over generic quantitative terms (maximum, minimum, how many) because
        those words appear in both building-code and financial questions. Only route 'both'
        when there is a doc signal AND a specifically financial term (sales, balance, etc.).
        """
        has_doc = bool(_DOC_SIGNAL_RE.search(question))
        has_table = bool(_TABLE_INTENT_RE.search(question))
        if has_doc and has_table:
            return "both" if bool(_FINANCIAL_RE.search(question)) else "doc"
        return "table" if has_table else "doc"
