"""
SQL Validation Layer — checks generated queries before execution.
Returns {"valid": True, "sql": sql} or {"valid": False, "error": str, "sql": fixed_sql_or_original}
Auto-fixes where possible; rejects gracefully otherwise.
"""
import re
import duckdb
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col_names(df: pd.DataFrame) -> list[str]:
    return [c.lower() for c in df.columns]


def _quoted_cols(sql: str) -> list[str]:
    """Extract all double-quoted identifiers from SQL."""
    return re.findall(r'"([^"]+)"', sql)


def _unquoted_cols(sql: str) -> list[str]:
    """Extract bare word identifiers that might be column names (rough heuristic)."""
    # Remove string literals and quoted identifiers first
    cleaned = re.sub(r"'[^']*'", '', sql)
    cleaned = re.sub(r'"[^"]*"', '', cleaned)
    keywords = {
        'select','from','where','group','by','order','having','limit','and','or','not',
        'as','on','join','left','right','inner','outer','full','with','case','when',
        'then','else','end','is','null','in','between','like','distinct','asc','desc',
        'sum','count','avg','min','max','round','coalesce','nullif','strptime','strftime',
        'cast','trim','lower','upper','substr','extract','date','true','false','over',
        'partition','row_number','rank','lag','lead','sales_data','cte','ranked',
    }
    tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', cleaned)
    return [t for t in tokens if t.lower() not in keywords]


# ── Check 1: Table name ───────────────────────────────────────────────────────

def _check_table_name(sql: str, df: pd.DataFrame) -> dict:
    tables = re.findall(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
    tables += re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE)
    for t in tables:
        if t.lower() not in ('sales_data', 'ranked', 'cte'):
            return {
                "valid": False,
                "sql": sql,
                "error": f"❌ Unknown table **'{t}'**. The only valid table is `sales_data`."
            }
    return {"valid": True, "sql": sql}


# ── Check 2: Column existence ─────────────────────────────────────────────────

def _check_columns_exist(sql: str, df: pd.DataFrame) -> dict:
    cols_lower = _col_names(df)
    quoted = _quoted_cols(sql)
    missing = [c for c in quoted if c.lower() not in cols_lower]
    if missing:
        suggestions = {}
        for m in missing:
            close = [c for c in df.columns if m.lower() in c.lower() or c.lower() in m.lower()]
            if close:
                suggestions[m] = close[0]
        if suggestions:
            hint = ", ".join(f"'{k}' → try '{v}'" for k, v in suggestions.items())
        else:
            hint = f"Available columns: {list(df.columns)}"
        return {
            "valid": False,
            "sql": sql,
            "error": f"❌ Column(s) not found: **{missing}**. {hint}"
        }
    return {"valid": True, "sql": sql}


# ── Check 3: strptime on TIMESTAMP columns ────────────────────────────────────

def _check_date_types(sql: str, df: pd.DataFrame) -> dict:
    timestamp_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    for col in timestamp_cols:
        pattern = rf'strptime\s*\(\s*["\']?{re.escape(col)}["\']?'
        if re.search(pattern, sql, re.IGNORECASE):
            fixed = re.sub(
                rf'strftime\s*\(\s*strptime\s*\(\s*["\']?{re.escape(col)}["\']?\s*,\s*\'[^\']+\'\s*\)',
                f'strftime({col}',
                sql, flags=re.IGNORECASE
            )
            # Simpler fix: remove the inner strptime wrapper
            fixed = re.sub(
                rf'strptime\s*\(\s*["\']?{re.escape(col)}["\']?\s*,\s*\'[^\']+\'\s*\)',
                col,
                sql, flags=re.IGNORECASE
            )
            return {
                "valid": True,   # auto-fixed, continue
                "sql": fixed,
                "warning": f"⚠️ Auto-fixed: '{col}' is a TIMESTAMP column — removed strptime() wrapper."
            }
    return {"valid": True, "sql": sql}


# ── Check 4: Discount scale ───────────────────────────────────────────────────

def _check_discount_scale(sql: str, df: pd.DataFrame) -> dict:
    has_discount = any('discount' in c.lower() for c in df.columns)
    if not has_discount:
        return {"valid": True, "sql": sql}

    matches = re.finditer(r'["\']?[Dd]iscount["\']?\s*([><=!]+)\s*([0-9]+(?:\.[0-9]+)?)', sql)
    fixed_sql = sql
    for m in matches:
        val = float(m.group(2))
        if val > 1:
            corrected = val / 100
            fixed_sql = fixed_sql.replace(m.group(0), m.group(0).replace(m.group(2), str(corrected)))
            return {
                "valid": True,   # auto-fixed
                "sql": fixed_sql,
                "warning": f"⚠️ Auto-fixed: Discount {val} → {corrected} (Discount is stored as 0–1 decimal)."
            }
    return {"valid": True, "sql": sql}


# ── Check 5: GROUP BY completeness ───────────────────────────────────────────

def _check_group_by(sql: str, df: pd.DataFrame) -> dict:
    """Check that non-aggregated SELECT columns appear in GROUP BY."""
    if 'GROUP BY' not in sql.upper():
        return {"valid": True, "sql": sql}

    # Extract SELECT clause (between SELECT and FROM)
    select_match = re.search(r'SELECT\s+(.+?)\s+FROM\s+', sql, re.IGNORECASE | re.DOTALL)
    groupby_match = re.search(r'GROUP\s+BY\s+(.+?)(?:ORDER\s+BY|HAVING|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)

    if not select_match or not groupby_match:
        return {"valid": True, "sql": sql}

    select_clause = select_match.group(1)
    groupby_clause = groupby_match.group(1).strip()

    # Find non-aggregated quoted column names in SELECT
    agg_pattern = r'(?:SUM|COUNT|AVG|MIN|MAX|ROUND|COALESCE|NULLIF)\s*\([^)]*\)'
    select_no_agg = re.sub(agg_pattern, '', select_clause, flags=re.IGNORECASE)
    select_cols = re.findall(r'"([^"]+)"', select_no_agg)

    # Check which are missing from GROUP BY
    groupby_lower = groupby_clause.lower()
    missing = [c for c in select_cols if c.lower() not in groupby_lower]

    if missing:
        # Auto-fix: append missing columns to GROUP BY
        new_groupby = groupby_clause.rstrip() + ', ' + ', '.join(f'"{c}"' for c in missing)
        fixed = re.sub(
            r'(GROUP\s+BY\s+)(.+?)(\s+(?:ORDER\s+BY|HAVING|LIMIT)|$)',
            lambda m: m.group(1) + new_groupby + m.group(3),
            sql, flags=re.IGNORECASE | re.DOTALL
        )
        return {
            "valid": True,   # auto-fixed
            "sql": fixed,
            "warning": f"⚠️ Auto-fixed: Added missing GROUP BY columns: {missing}"
        }
    return {"valid": True, "sql": sql}


# ── Check 6: Divide-by-zero ───────────────────────────────────────────────────

def _check_divide_by_zero(sql: str, df: pd.DataFrame) -> dict:
    # Look for division by SUM/COUNT without NULLIF protection
    if re.search(r'/\s*SUM\s*\(', sql, re.IGNORECASE) and 'NULLIF' not in sql.upper():
        # Auto-fix: wrap SUM denominator with NULLIF
        fixed = re.sub(
            r'/\s*(SUM\s*\([^)]+\))',
            r'/ NULLIF(\1, 0)',
            sql, flags=re.IGNORECASE
        )
        return {
            "valid": True,   # auto-fixed
            "sql": fixed,
            "warning": "⚠️ Auto-fixed: Wrapped SUM() denominator with NULLIF(..., 0) to prevent divide-by-zero."
        }
    return {"valid": True, "sql": sql}


# ── Check 7: DuckDB EXPLAIN (syntax + binder errors) ─────────────────────────

def _check_explain(sql: str, df: pd.DataFrame) -> dict:
    try:
        sales_data = df  # noqa: F841 — required for duckdb.query to resolve table
        duckdb.query(f"EXPLAIN {sql}")
        return {"valid": True, "sql": sql}
    except Exception as e:
        err = str(e)

        # Column not found
        if "Referenced column" in err or "does not exist" in err or "Binder Error" in err:
            col_match = re.search(r'"([^"]+)"', err)
            col_name = col_match.group(1) if col_match else ""
            close = [c for c in df.columns if col_name.lower() in c.lower() or c.lower() in col_name.lower()]
            hint = f" Did you mean: **{close[0]}**?" if close else f" Available: {list(df.columns)}"
            return {
                "valid": False,
                "sql": sql,
                "error": f"❌ Column **'{col_name}'** not found.{hint}"
            }

        # Syntax error
        if "Parser Error" in err:
            return {
                "valid": False,
                "sql": sql,
                "error": f"❌ SQL syntax error: {err.split('Parser Error:')[-1].strip()}"
            }

        # GROUP BY error
        if "must appear in the GROUP BY" in err:
            col_match = re.search(r'"([^"]+)"', err)
            col = col_match.group(1) if col_match else "a column"
            return {
                "valid": False,
                "sql": sql,
                "error": f"❌ **'{col}'** must be in GROUP BY. Add `GROUP BY \"{col}\"` to the query."
            }

        # strptime on TIMESTAMP
        if "strptime(TIMESTAMP" in err:
            return {
                "valid": False,
                "sql": sql,
                "error": "❌ Used strptime() on a TIMESTAMP column. Use strftime(column, '%Y-%m-%d') directly."
            }

        return {
            "valid": False,
            "sql": sql,
            "error": f"❌ Query validation failed: {err}"
        }


# ── Check 8: Chart dimensions (post-execution) ────────────────────────────────

def validate_chart_dimensions(result_df: pd.DataFrame) -> dict:
    """Call this after query execution to validate the result is chartable."""
    numeric = result_df.select_dtypes(include="number").columns.tolist()
    text = result_df.select_dtypes(include="object").columns.tolist()

    if len(result_df) == 0:
        return {"valid": False, "warning": "⚠️ Query returned 0 rows — nothing to chart."}
    if not numeric:
        return {"valid": False, "warning": "⚠️ No numeric columns in result — cannot render chart."}
    if not text and len(result_df) < 2:
        return {"valid": False, "warning": "⚠️ Single value result — chart not applicable."}
    return {"valid": True}


# ── Main entry point ──────────────────────────────────────────────────────────

CHECKS = [
    _check_table_name,
    _check_columns_exist,
    _check_date_types,
    _check_discount_scale,
    _check_group_by,
    _check_divide_by_zero,
    _check_explain,
]


def validate_sql(sql: str, df: pd.DataFrame) -> dict:
    """
    Run all validation checks on a generated SQL query.

    Returns:
        {
          "valid": True,
          "sql": str,           # possibly auto-fixed SQL
          "warnings": [str],    # list of auto-fix notes
        }
        or
        {
          "valid": False,
          "sql": str,           # original SQL
          "error": str,         # human-friendly error message
          "warnings": [str],
        }
    """
    warnings = []
    current_sql = sql

    for check in CHECKS:
        result = check(current_sql, df)
        if result.get("warning"):
            warnings.append(result["warning"])
        if result.get("sql"):
            current_sql = result["sql"]   # carry forward auto-fixes
        if not result["valid"]:
            return {
                "valid": False,
                "sql": current_sql,
                "error": result.get("error", "Unknown validation error."),
                "warnings": warnings,
            }

    return {"valid": True, "sql": current_sql, "warnings": warnings}
