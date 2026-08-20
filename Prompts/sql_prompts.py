SQL_SYSTEM = "Output only a valid SQLite SELECT statement. No markdown. No explanation."


def build_sql_prompt(question: str, schema_info: list) -> str:
    """Build the NL-to-SQL prompt given schema info dicts and the user question."""
    schema_text = "\n\n".join(
        f"Table `{s['table_name']}` (source: {s['source']}, {s['nrows']} rows)\n"
        f"  Numeric columns (REAL — safe for SUM/AVG/MIN/MAX): {', '.join(s['numeric_cols']) or 'none'}\n"
        f"  Text columns (strings only — NEVER use in SUM/AVG/MIN/MAX): {', '.join(s['text_cols'])}\n"
        f"  Sample rows:\n{s['sample']}"
        for s in schema_info
    )
    table_col_blocks = "\n".join(
        f"  `{s['table_name']}` valid columns: "
        + ", ".join(f"`{c}`" for c in s["numeric_cols"] + s["text_cols"])
        for s in schema_info
    )
    text_col_warnings = []
    for s in schema_info:
        bad_cols = [c for c in s["text_cols"] if c.lower() in ("balance", "total", "amount", "price")]
        if bad_cols:
            text_col_warnings.append(
                f"WARNING: {', '.join(bad_cols)} in `{s['table_name']}` are TEXT (OCR output with symbols like '$', '€', ',') — "
                f"they CANNOT be summed. Use {', '.join(s['numeric_cols']) or 'numeric columns'} instead."
            )
        num_lower = [c.lower() for c in s["numeric_cols"]]
        if "balance" in num_lower and "credit" in num_lower:
            text_col_warnings.append(
                f"WARNING: In `{s['table_name']}`, 'Balance' is a running account total after each transaction — "
                f"it is NOT the transaction amount. For spending/payment queries, ALWAYS use SUM(Credit) WHERE Credit < 0. "
                f"NEVER use SUM(Balance), AVG(Balance), or MIN(Balance) for spending totals."
            )
        elif "balance" in num_lower and "credit" not in num_lower:
            # Only Balance available — it's likely a running total, warn accordingly
            text_col_warnings.append(
                f"WARNING: In `{s['table_name']}`, 'Balance' is a running account total, not a per-transaction amount. "
                f"Use it only when the question explicitly asks for account balance."
            )
    warnings_block = ("\n".join(text_col_warnings) + "\n\n") if text_col_warnings else ""

    return (
        "SQLite database tables:\n\n"
        + schema_text
        + "\n\nCOLUMN CONSTRAINTS (only these column names are valid — no others exist):\n"
        + table_col_blocks
        + "\n\n"
        + warnings_block
        + "RULES:\n"
        "  1. NEVER use text columns in SUM/AVG/MIN/MAX — they contain strings, not numbers.\n"
        "  1b. NEVER recompute a column that already exists. If a 'Sales' column is present, use SUM(Sales) — do NOT write SUM('Unit Price' * Quantity) or similar.\n"
        "  2. Date comparisons — choose format based on the sample:\n"
        "     - If sample dates are ISO (2022-01-02): WHERE DATE(Date)='2022-01-02'\n"
        "     - If sample dates are M/D/YYYY (1/2/2022): WHERE Date='1/2/2022'  (no DATE() — it breaks non-ISO)\n"
        "     - If unsure: WHERE Date LIKE '%1/2/2022%'\n"
        "     NEVER use DATE() on a column whose sample shows M/D/YYYY values.\n"
        "     Month names → numbers: jan=1, feb=2, mar=3, apr=4, may=5, jun=6, jul=7, aug=8, sep=9, oct=10, nov=11, dec=12\n"
        "     Month filter (ISO):     WHERE strftime('%m', Date)='04'  (April)\n"
        "     Month filter (M/D/Y):   WHERE Date LIKE '4/%'  (April, when sample shows M/D/YYYY)\n"
        "  3. For spending/expense queries on a bank table: filter Credit < 0 (negative = debit/spending)\n"
        "  4. ONLY filter on columns explicitly mentioned in the question. Do NOT infer extra filters from sample data.\n"
        "     WRONG (question says 'all items'): WHERE LOWER(Item)='apple' AND LOWER(Sales_Rep)='william'\n"
        "     CORRECT:                           WHERE LOWER(Sales_Rep)='william'\n"
        "  5. Always use LOWER() for text column comparisons.\n"
        "     For exact names: WHERE LOWER(Sales_Rep)='william'\n"
        "     For categories/keywords: WHERE LOWER(Description) LIKE '%grocery%'\n"
        "     NEVER use bare equality for text without LOWER().\n"
        "  6. For spending/payment queries (how much paid, total spent, expenses), wrap the result in ABS() and alias it:\n"
        "     SELECT ABS(SUM(Credit)) AS Amount_Spent FROM tbl WHERE ...\n"
        "     This returns a positive number since bank debits are stored as negative values.\n"
        "  7. If both 'Credit' and 'Balance' columns exist: Balance is the running account balance AFTER each transaction.\n"
        "     NEVER aggregate Balance for spending queries. Always use Credit (negative = spending, positive = income).\n"
        "  8. Always include ALL filters from the question in the WHERE clause — never in CASE WHEN.\n"
        "     WRONG: SELECT ABS(SUM(CASE WHEN LOWER(Description) LIKE '%grocery%' THEN Credit END)) FROM tbl WHERE strftime('%m', Date)='03' AND Credit < 0\n"
        "     CORRECT: SELECT ABS(SUM(Credit)) FROM tbl WHERE strftime('%m', Date)='03' AND LOWER(Description) LIKE '%grocery%' AND Credit < 0\n"
        "     If the question mentions a category (e.g. 'grocery'), include LOWER(Description) LIKE '%grocery%' in WHERE.\n"
        "     If the question mentions a month, include the date filter in WHERE. Include BOTH if both are mentioned.\n"
        "\n"
        "Query patterns:\n"
        "  Specific day (ISO):     WHERE DATE(Date)='2022-01-02'\n"
        "  Specific day (M/D/Y):   WHERE Date='1/2/2022'  (when sample shows that format)\n"
        "  Month filter (ISO):     WHERE strftime('%m', Date)='04'  (April)\n"
        "  Month filter (M/D/Y):   WHERE Date LIKE '4/%'  (April, when sample shows M/D/YYYY)\n"
        "  Exact text filter:      WHERE LOWER(Sales_Rep)='william'\n"
        "  Category/keyword:       WHERE LOWER(Description) LIKE '%grocery%'\n"
        "  All items for person:   SELECT SUM(Sales) FROM tbl WHERE LOWER(Sales_Rep)='william'\n"
        "  Numeric aggregation:    SELECT SUM(Sales) FROM tbl WHERE ...\n"
        "  Spending total (ISO):    SELECT SUM(Credit) FROM tbl WHERE strftime('%m', Date)='04' AND Credit < 0\n"
        "  Spending total (M/D/Y): SELECT SUM(Credit) FROM tbl WHERE Date LIKE '4/%' AND Credit < 0\n"
        "  Spending by category (ISO):  SELECT ABS(SUM(Credit)) AS Amount_Spent FROM tbl WHERE strftime('%m', Date)='03' AND LOWER(Description) LIKE '%grocery%' AND Credit < 0\n"
        "  Spending by category (M/D/Y): SELECT ABS(SUM(Credit)) AS Amount_Spent FROM tbl WHERE Date LIKE '4/%' AND LOWER(Description) LIKE '%grocery%' AND Credit < 0\n"
        "  Spending list:          SELECT Date, Description, Credit FROM tbl WHERE Credit < 0 ORDER BY Credit\n"
        "\n"
        f"Question: {question}\n\n"
        "Write one SQLite SELECT statement. OUTPUT ONLY THE SQL — no markdown, no explanation.\n"
    )
