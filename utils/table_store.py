import json
import logging
import re
import sqlite3
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
TABLE_DIR = Path(__file__).parent.parent / "data" / "tables"

_num_re = re.compile(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$')
_merged_newline_re = re.compile(r'^(.*?)\n(-?\d[\d,]*(?:\.\d+)?)\s*$', re.DOTALL)


def _safe_name(source: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", source)[:40]


def _try_clean_numeric(series: pd.Series) -> "pd.Series | None":
    """Extract trailing number from OCR-garbled text (e.g. 'Mobile -833.71'). Returns float series if >60% parse."""
    # Skip date-like columns — datetime strings end in digits and would be mangled
    try:
        parsed_dates = pd.to_datetime(series, errors="coerce", format="mixed")
    except Exception:
        parsed_dates = pd.to_datetime(series, errors="coerce")
    if parsed_dates.notna().mean() > 0.5:
        return None

    def extract_last_num(s):
        s_str = str(s).strip()
        is_negative = s_str.startswith('-') and not s_str[1:2].isdigit()
        m = _num_re.search(s_str)
        if m:
            val = float(m.group(1).replace(',', ''))
            if is_negative and val > 0:
                val = -val
            return val
        return None

    cleaned = series.map(extract_last_num)
    if cleaned.notna().mean() > 0.6:
        return cleaned
    return None


def _split_merged_desc_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Split OCR-merged 'Description\\nBalance' text columns into separate columns.

    When ≥15% of non-empty values in a text column match 'text\\nnumber' (e.g.
    'Grocery\\n-693.90'), the column is renamed to 'Description' and a new numeric
    'Balance' column is added with the extracted numbers.  Must run BEFORE
    _try_clean_numeric so that description text is not discarded.
    """
    df = df.copy()
    for col in list(df.columns):
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        str_series = df[col].fillna('').astype(str)

        # Skip date-like columns
        try:
            pd_dates = pd.to_datetime(df[col], errors='coerce', format='mixed')
        except Exception:
            pd_dates = pd.to_datetime(df[col], errors='coerce')
        if pd_dates.notna().mean() > 0.5:
            continue

        # Detect: ≥15% of non-empty values contain 'text\nnumber'
        non_empty = str_series.str.strip().ne('') & ~str_series.isin(['nan', 'None'])
        if non_empty.sum() == 0:
            continue
        has_merged = str_series.apply(lambda s: bool(_merged_newline_re.match(s.strip())))
        if has_merged.sum() / non_empty.sum() < 0.15:
            continue

        # Split each value into (description text, balance number)
        desc_vals, bal_vals = [], []
        for val in df[col]:
            s = str(val).strip() if pd.notna(val) else ''
            if not s or s in ('nan', 'None'):
                desc_vals.append(None)
                bal_vals.append(float('nan'))
                continue
            m = _merged_newline_re.match(s)
            if m:
                desc_vals.append(m.group(1).strip() or None)
                bal_vals.append(float(m.group(2).replace(',', '')))
            else:
                # Try trailing number without newline (e.g. 'Amazon Omron -1,026.58')
                nm = _num_re.search(s)
                if nm:
                    text_part = s[:nm.start()].strip()
                    if text_part and not text_part.replace('-', '').replace(' ', '').isdigit():
                        desc_vals.append(text_part)
                        bal_vals.append(float(nm.group(1).replace(',', '')))
                    else:
                        desc_vals.append(s)
                        bal_vals.append(float('nan'))
                else:
                    desc_vals.append(s)
                    bal_vals.append(float('nan'))

        # Rename the source column to 'Description' (avoid collisions)
        desc_name = 'Description'
        if desc_name in df.columns and desc_name != col:
            n = 2
            while f'Description_{n}' in df.columns and f'Description_{n}' != col:
                n += 1
            desc_name = f'Description_{n}'
        if col != desc_name:
            df = df.rename(columns={col: desc_name})
        df[desc_name] = desc_vals

        # Name the new numeric column based on table context:
        # If Credit/Debit already exists, the extracted number is a running Balance.
        # Otherwise it's a transaction Amount (e.g. sales tables, invoices).
        existing_lower = {str(c).lower() for c in df.columns}
        has_credit = "credit" in existing_lower or "debit" in existing_lower
        default_num_name = 'Balance' if has_credit else 'Amount'
        bal_name = default_num_name
        if bal_name in df.columns:
            n = 2
            while f'{default_num_name}_{n}' in df.columns:
                n += 1
            bal_name = f'{default_num_name}_{n}'
        df[bal_name] = bal_vals

    return df


class TableStore:
    def __init__(self, table_dir=TABLE_DIR):
        self.dir = Path(table_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.dir / "index.json"
        self._index: dict[str, int] = {}
        self._load_index()

    def _load_index(self):
        if self._index_path.exists():
            try:
                raw = json.loads(self._index_path.read_text())
                # Migrate from old parquet-path format (values were lists) to int count format
                self._index = {k: v for k, v in raw.items() if isinstance(v, int)}
            except Exception:
                self._index = {}

    def _save_index(self):
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def _db_path(self, source: str) -> Path:
        return self.dir / f"{_safe_name(source)}.db"

    def save(self, source: str, dataframes: list):
        db_path = self._db_path(source)
        db_path.unlink(missing_ok=True)
        n = 0
        if dataframes:
            conn = sqlite3.connect(str(db_path))
            for i, df in enumerate(dataframes):
                try:
                    lower_cols = [str(c).strip().lower() for c in df.columns]
                    if len(lower_cols) != len(set(lower_cols)):
                        logger.debug(f"Skipping table {i} in '{source}': duplicate column names (mis-detected table)")
                        continue
                    df.to_sql(f"t{i}", conn, if_exists="replace", index=False)
                    n += 1
                except Exception as e:
                    logger.warning(f"SQLite write failed for '{source}' table {i}: {e}")
                    try:
                        conn.rollback()
                    except Exception:
                        pass
            conn.close()
        if n > 0:
            self._index[source] = n
        else:
            self._index.pop(source, None)
        self._save_index()

    def load(self, source: str) -> list[pd.DataFrame]:
        n = self._index.get(source, -1)
        if n <= 0:
            return []
        db_path = self._db_path(source)
        if not db_path.exists():
            return []
        conn = sqlite3.connect(str(db_path))
        tables = []
        for i in range(n):
            try:
                tables.append(pd.read_sql(f"SELECT * FROM t{i}", conn))
            except Exception:
                pass
        conn.close()
        return tables

    def has_tables(self, source: str) -> bool:
        return self._index.get(source, 0) > 0

    def merged_count(self, source: str) -> int:
        """User-visible table count after processing and merging fragments."""
        try:
            _, schema_info = self.load_into_memory([source])
            return len(schema_info)
        except Exception:
            return self._index.get(source, 0)

    def was_attempted(self, source: str) -> bool:
        return source in self._index

    def remove(self, source: str):
        self._db_path(source).unlink(missing_ok=True)
        self._index.pop(source, None)
        self._save_index()

    def clear_all(self):
        for source in list(self._index.keys()):
            self._db_path(source).unlink(missing_ok=True)
        self._index = {}
        self._save_index()

    def load_into_memory(self, sources: list[str], max_per_source: int = 20) -> tuple[sqlite3.Connection, list[dict]]:
        """Load tables for `sources` into an in-memory SQLite DB.

        Returns (conn, schema_info) where schema_info is a list of dicts with
        keys: table_name, source, columns, sample_str.
        Only tables with ≥2 rows and ≥2 cols and at least one numeric-ish column
        are included (garbage-table filter). At most max_per_source useful tables
        are loaded per source to bound memory usage.
        """
        conn = sqlite3.connect(":memory:")
        schema_info = []
        for src in sources:
            dfs = self.load(src)
            useful_count = 0
            # Pipeline each df, then merge same-schema fragments into one table
            processed: list[pd.DataFrame] = []
            for df in dfs:
                if useful_count >= max_per_source:
                    break
                if not _is_useful(df):
                    continue
                df = _maybe_promote_header(df)
                df = _infer_column_names(df)
                df = _split_merged_desc_balance(df)
                for col in list(df.columns):
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        cleaned = _try_clean_numeric(df[col])
                        if cleaned is not None:
                            df[col] = cleaned
                df.columns = [
                    re.sub(r'[^a-zA-Z0-9_]', '_', str(c)).strip('_') or f'col_{i}'
                    for i, c in enumerate(df.columns)
                ]
                processed.append(df)
                useful_count += 1

            # Merge fragments from the same source that share ≥50% of columns (overlap by max cols).
            # Fragments with little column overlap are kept as separate tables.
            groups: list[list[pd.DataFrame]] = []
            for df in processed:
                placed = False
                for group in groups:
                    rep = group[0]
                    a, b = set(rep.columns), set(df.columns)
                    overlap = len(a & b) / max(len(a), len(b), 1)
                    if overlap >= 0.5:
                        group.append(df)
                        placed = True
                        break
                if not placed:
                    groups.append([df])

            src_idx = 0
            for group_dfs in groups:
                if len(group_dfs) > 1:
                    merged = pd.concat(group_dfs, ignore_index=True, join='outer')
                    merged = merged.dropna(axis=1, how='all')
                else:
                    merged = group_dfs[0]
                merged = _fill_missing_credit_from_balance(merged)
                tname = f"{_safe_name(src)}_t{src_idx}"
                src_idx += 1
                try:
                    merged.to_sql(tname, conn, if_exists="replace", index=False)
                except Exception as e:
                    logger.warning(f"in-memory load failed for '{src}': {e}")
                    continue
                numeric_cols = [c for c in merged.columns if pd.api.types.is_numeric_dtype(merged[c])]
                text_cols = [c for c in merged.columns if not pd.api.types.is_numeric_dtype(merged[c])]
                sample = merged.head(min(len(merged), 8)).to_string(index=False)
                schema_info.append({
                    "table_name": tname,
                    "source": src,
                    "numeric_cols": numeric_cols,
                    "text_cols": text_cols,
                    "sample": sample,
                    "nrows": len(merged),
                })
        return conn, schema_info


def _infer_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Assign meaningful names to tables whose columns are all sequential integers.

    Detects common bank-statement patterns: row number, date, credit/debit, description, balance.
    Only runs when every column name is a digit string (e.g. '0','1','2','3','4').
    """
    cols = [str(c) for c in df.columns]
    if not all(c.isdigit() for c in cols):
        return df

    new_cols = []
    drop_indices = set()
    used = set()

    def _unique(name):
        if name not in used:
            used.add(name)
            return name
        n = 2
        while f"{name}_{n}" in used:
            n += 1
        used.add(f"{name}_{n}")
        return f"{name}_{n}"

    for idx, col in enumerate(df.columns):
        series = df[col]
        # Date detection
        try:
            parsed = pd.to_datetime(series, errors='coerce', format='mixed')
        except Exception:
            parsed = pd.to_datetime(series, errors='coerce')
        if parsed.notna().mean() > 0.5:
            new_cols.append(_unique('Date'))
            continue
        # Numeric column — check native dtype first, then try coercing object columns
        if pd.api.types.is_numeric_dtype(series):
            num_series = series
        else:
            num_series = pd.to_numeric(
                series.astype(str).str.replace(r'[$,\s%]', '', regex=True),
                errors='coerce',
            )
        non_null = num_series.dropna()
        if len(non_null) / max(len(series), 1) >= 0.5:
            if len(non_null) == 0:
                new_cols.append(_unique('col'))
                continue
            # Row-number heuristic: all positive integers > 50, nearly all integers → OCR artifact, drop
            if (non_null > 50).all() and (non_null % 1 == 0).mean() > 0.9:
                drop_indices.add(idx)
                new_cols.append(None)
                continue
            # Mostly negative → credit/debit amounts
            elif (non_null < 0).mean() > 0.3:
                if 'Credit' not in used:
                    new_cols.append(_unique('Credit'))
                else:
                    new_cols.append(_unique('Balance'))
            else:
                new_cols.append(_unique('Amount'))
            continue
        # Text column → Description
        new_cols.append(_unique('Description'))

    keep = [i for i in range(len(new_cols)) if i not in drop_indices]
    named = [new_cols[i] for i in keep]
    if len(named) != len(set(named)):
        return df  # Abort if we produced duplicates somehow
    df = df.copy()
    df = df.iloc[:, keep]
    df.columns = named
    return df


def _maybe_promote_header(df: pd.DataFrame) -> pd.DataFrame:
    """Promote first data row to column headers when extraction missed the header row.

    Triggered when all column names are sequential integers (0, 1, 2, ...) and the
    first row contains at least 2 non-numeric string values that look like headers.
    """
    if len(df) < 2:
        return df
    # Check all column names are sequential integers starting at 0 (int or stringified form)
    cols = list(df.columns)
    n = len(cols)
    if cols != list(range(n)) and cols != [str(i) for i in range(n)]:
        return df
    # Check first row has ≥2 non-numeric, non-empty string values
    first_row = df.iloc[0]
    text_vals = [
        str(v).strip() for v in first_row
        if pd.notna(v)
        and str(v).strip() not in ("", "nan")
        and not str(v).strip().replace(".", "").replace("-", "").replace(",", "").isdigit()
    ]
    if len(text_vals) < 2:
        return df
    # Build new column names from first row; fall back to col_N for blanks/numbers/None
    new_cols = []
    for i, v in enumerate(first_row):
        s = str(v).strip()
        # Take only the first line when OCR merged multiple header lines
        s = s.split('\n')[0].strip()
        if not s or s.lower() in ("nan", "none") or s.replace(".", "").replace("-", "").replace(",", "").isdigit():
            s = f"col_{i}"
        new_cols.append(s)
    # Abort promotion if it would create duplicate column names
    if len(new_cols) != len(set(new_cols)):
        return df
    df = df.copy()
    df.columns = new_cols
    df = df.iloc[1:].reset_index(drop=True)
    # Drop col_N fallback columns that contain sequential positive integers (OCR row-number artifacts)
    cols_to_drop = []
    for c in df.columns:
        if not (isinstance(c, str) and c.startswith('col_')):
            continue
        num = pd.to_numeric(df[c].astype(str).str.strip(), errors='coerce')
        non_null = num.dropna()
        if (len(non_null) / max(len(df[c]), 1) > 0.8
                and len(non_null) > 0
                and (non_null > 0).all()
                and (non_null % 1 == 0).mean() > 0.9):
            cols_to_drop.append(c)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def _fill_missing_credit_from_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Recover NaN Credit values using Credit[i] = Balance[i] - Balance[i-1]."""
    import math
    cols_lower = {str(c).lower(): c for c in df.columns}
    if 'credit' not in cols_lower or 'balance' not in cols_lower:
        return df
    credit_col = cols_lower['credit']
    balance_col = cols_lower['balance']
    if not (pd.api.types.is_numeric_dtype(df[credit_col]) and pd.api.types.is_numeric_dtype(df[balance_col])):
        return df
    df = df.copy()
    credit = df[credit_col].astype(float).tolist()
    balance = df[balance_col].astype(float).tolist()
    # Pass 1: forward-fill Balance where NaN using Credit + previous Balance
    for i in range(1, len(df)):
        if math.isnan(balance[i]) and not math.isnan(balance[i - 1]) and not math.isnan(credit[i]):
            balance[i] = balance[i - 1] + credit[i]
    # Pass 1b: backward-fill Balance where NaN using next Balance - next Credit
    for i in range(len(df) - 2, -1, -1):
        if math.isnan(balance[i]) and not math.isnan(balance[i + 1]) and not math.isnan(credit[i + 1]):
            balance[i] = balance[i + 1] - credit[i + 1]
    # Pass 2: recover Credit where NaN from consecutive Balance values
    for i in range(1, len(df)):
        if math.isnan(credit[i]) and not math.isnan(balance[i]) and not math.isnan(balance[i - 1]):
            credit[i] = balance[i] - balance[i - 1]
    df[credit_col] = credit
    df[balance_col] = balance
    return df


def _is_useful(df: pd.DataFrame) -> bool:
    if len(df) < 2 or len(df.columns) < 2:
        return False
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return True
        parsed = pd.to_numeric(
            df[col].astype(str).str.replace(r"[$,\s%]", "", regex=True),
            errors="coerce",
        )
        if parsed.notna().mean() > 0.35:
            return True
    # Text-only: valid if every column has ≥50% non-empty values and ≥2 distinct values
    for col in df.columns:
        vals = df[col].astype(str).str.strip()
        non_empty = vals[(vals.str.len() > 0) & ~vals.isin(["nan", "None", ""])]
        if len(non_empty) / max(len(df), 1) < 0.5:
            return False
        if non_empty.nunique() < 2:
            return False
    return True
