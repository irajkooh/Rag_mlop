"""
SQL execution helpers: fence-stripping and safe query execution.
"""
import re as _re
from typing import Optional, Tuple, List


def strip_sql_fences(raw: str) -> str:
    """Remove markdown code fences from an LLM-generated SQL string."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
        if "```" in raw:
            raw = raw[: raw.rfind("```")]
    return raw.strip()


def execute_sql(
    sql: str,
    conn,
) -> Optional[Tuple[str, List, List[str]]]:
    """Execute sql on conn. Returns (sql, rows, col_names) or None on failure."""
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        return sql, rows, col_names
    except Exception:
        return None


def build_detail_sql(aggregate_sql: str, limit: int = 15) -> Optional[str]:
    """From a single-value aggregate SQL, build a SELECT * query for underlying rows."""
    m = _re.search(r'\bFROM\b', aggregate_sql, _re.IGNORECASE)
    if m:
        from_clause = aggregate_sql[m.start():].rstrip(';').strip()
        # Strip any trailing ORDER BY so we can append LIMIT cleanly
        from_clause = _re.sub(r'\s+ORDER\s+BY\b.*$', '', from_clause, flags=_re.IGNORECASE).strip()
        return f"SELECT * {from_clause} LIMIT {limit}"
    return None
