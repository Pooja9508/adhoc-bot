import os
import re
import duckdb
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from validator import validate_sql

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATA_PATH = "data/sales_data.csv"
try:
    df_global = pd.read_csv(DATA_PATH, encoding='latin-1')
except FileNotFoundError:
    df_global = pd.DataFrame()  # empty fallback — uploaded datasets still work
except Exception:
    df_global = pd.DataFrame()

STATE_ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

STATE_ALIASES = {
    "cali": "California", "calif": "California", "cal": "California",
    "tex": "Texas", "fla": "Florida", "penn": "Pennsylvania",
    "mass": "Massachusetts", "conn": "Connecticut", "minn": "Minnesota",
}


def resolve_state(token: str) -> str | None:
    """Return the full state name for an abbreviation or alias, or None if not found.
    2-letter codes only match when ALL CAPS (CA, TX) to avoid replacing common words like 'me', 'in', 'or'.
    Longer aliases (cali, calif) match case-insensitively.
    """
    stripped = token.strip(".,!?")
    # 2-letter abbreviation: MUST be all-caps and all-alpha to avoid catching words like 'me', 'or', 'in'
    if len(stripped) == 2 and stripped.isalpha() and stripped == stripped.upper():
        if stripped in STATE_ABBR:
            return STATE_ABBR[stripped]
    # Longer aliases: case-insensitive
    lower = stripped.lower()
    if lower in STATE_ALIASES:
        return STATE_ALIASES[lower]
    return None


def normalize_request(user_request: str) -> str:
    """Replace state abbreviations/aliases with full names in the request."""
    words = re.split(r"(\s+)", user_request)
    result = []
    for word in words:
        resolved = resolve_state(word)
        result.append(resolved if resolved else word)
    return "".join(result)


def _build_time_rules(df: pd.DataFrame) -> str:
    """Return TIME RULES if dataset has a time-of-day column (e.g. transaction_time)."""
    import datetime
    time_cols = []
    for c in df.columns:
        if any(kw in c.lower() for kw in ["time", "hour"]) and df[c].dtype == object:
            try:
                non_null = df[c].dropna()
                if len(non_null) == 0:
                    continue
                sample = non_null.iloc[0]
                if isinstance(sample, datetime.time):
                    time_cols.append(c)
            except Exception:
                continue

    if not time_cols:
        return ""

    col = time_cols[0]
    return f"""
TIME-OF-DAY RULES — '{col}' contains time-of-day values (stored as TIME objects):
- Extract hour : HOUR({col}) as hour_of_day
- Group by hour: GROUP BY HOUR({col}) ORDER BY hour_of_day
- Example (sales by hour):
  SELECT HOUR({col}) as hour_of_day,
         SUM(unit_price * transaction_qty) as total_revenue,
         COUNT(*) as transaction_count
  FROM sales_data
  GROUP BY hour_of_day
  ORDER BY total_revenue DESC
- Do NOT use strptime() or strftime() on time columns — use HOUR(), MINUTE() directly
- Format hours as integers (0–23); the UI will display them as AM/PM
"""


def _build_date_rules(df: pd.DataFrame) -> str:
    """Return the correct DATE RULES block based on actual column types in df."""
    # Find date-like columns
    timestamp_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    varchar_date_cols = [
        c for c in df.columns
        if c not in timestamp_cols
        and any(kw in c.lower() for kw in ["date", "time", "month", "year", "period"])
        and df[c].dtype == object
    ]

    if timestamp_cols:
        # Columns are already TIMESTAMP — use strftime directly, no strptime needed
        example_col = timestamp_cols[0]
        return f"""
DATE RULES — these columns are already TIMESTAMP/DATE type (NOT VARCHAR strings): {timestamp_cols}
- NEVER use strptime() on TIMESTAMP columns — it only works on VARCHAR and will throw a Binder Error
- Use strftime() directly on the column:
  - Daily grouping  : strftime({example_col}, '%Y-%m-%d') as sale_date
  - Monthly grouping: strftime({example_col}, '%Y-%m') as sale_month
  - Yearly grouping : strftime({example_col}, '%Y') as sale_year
  - Year filter     : strftime({example_col}, '%Y') = '2023'
  - Month filter    : strftime({example_col}, '%Y-%m') = '2023-01'
  - Day filter      : strftime({example_col}, '%Y-%m-%d') = '2023-01-15'
- Correct  : strftime(transaction_date, '%Y-%m') = '2023-01'
- Incorrect: strptime(transaction_date, '%Y-%m-%d') — will cause Binder Error
"""
    elif varchar_date_cols:
        # Superstore-style VARCHAR dates in M/D/YYYY format
        example_col = f'"{varchar_date_cols[0]}"'
        return f"""
DATE RULES — date columns ({varchar_date_cols}) are stored as VARCHAR in M/D/YYYY format (e.g. '11/8/2016'):
- ALWAYS parse with strptime before any date function: strptime({example_col}, '%m/%d/%Y')
- Monthly grouping : strftime(strptime({example_col}, '%m/%d/%Y'), '%Y-%m') as order_month
- Yearly  grouping : strftime(strptime({example_col}, '%m/%d/%Y'), '%Y') as order_year
- Year filter      : strftime(strptime({example_col}, '%m/%d/%Y'), '%Y') = '2017'
- Month filter     : strftime(strptime({example_col}, '%m/%d/%Y'), '%Y-%m') = '2017-03'
- NEVER use CAST({example_col} AS DATE) — the format is M/D/YYYY not YYYY-MM-DD
- NEVER use date_part(), YEAR(), or MONTH() directly on {example_col}
"""
    else:
        return ""


def generate_sql(user_request: str, df: pd.DataFrame) -> str:
    normalized = normalize_request(user_request)
    columns = df.columns.tolist()
    sample = df.head(2).to_string(index=False)

    # Collect unique states from data for the LLM to reference
    state_col = next((c for c in df.columns if c.lower() == "state"), None)
    unique_states = sorted(df[state_col].dropna().unique().tolist()) if state_col else []

    date_rules = _build_date_rules(df)
    time_rules = _build_time_rules(df)

    prompt = f"""
You are a SQL expert. Write ONE single DuckDB SQL query to answer this request.

Exact column names (use these exactly):
{columns}

Sample data:
{sample}

Available states in data (exact spelling):
{unique_states}

Table name: sales_data (already loaded as pandas dataframe)

Rules:
- Return ONLY one single SQL query
- No markdown, no backticks, no explanation, no semicolons
- Use double quotes around column names that have spaces

{date_rules}
{time_rules}

QUERY TYPE — choose based on the request:
1. AGGREGATE (total, sum, count, how many, average, max, min, units sold, revenue):
   → Use aggregate functions: SUM(), COUNT(), AVG(), MAX(), MIN()
   → CRITICAL: Every non-aggregated column in SELECT must appear in GROUP BY
   → Example: SELECT store_location, SUM(unit_price * transaction_qty) as total_revenue FROM sales_data GROUP BY store_location ORDER BY total_revenue DESC
   → WRONG: SELECT store_location, SUM(...) FROM sales_data  ← missing GROUP BY store_location
   → If comparing across a dimension (region, category, store, segment), always GROUP BY that dimension

2. DETAIL/LIST (show me, give me, list, find, orders from, records for):
   → Use SELECT * to return all columns
   → Add ORDER BY relevant to request (dates DESC, sales DESC, etc.)
   → Example: SELECT * FROM sales_data WHERE ...

COLUMN MAPPING RULES — map business terms to actual columns intelligently:
Available columns: {columns}

- "sales" / "revenue" / "income":
  → If columns contain both a price col (unit_price, price, selling_price) AND a qty col (transaction_qty, quantity, units_sold):
     use price_col * qty_col as total_revenue  e.g. SUM(unit_price * transaction_qty)
  → Otherwise look for: Sales, Revenue, Amount, discounted_price, actual_price
  → Last resort: use any numeric column that represents value

- "quantity" / "units" / "transactions" / "orders" / "count":
  → Look for: transaction_qty, Quantity, Units, Qty, units_sold
  → For order count use COUNT(DISTINCT order_id or transaction_id)

- "profit" → Profit, Margin, profit_margin
- "discount" → Discount, discount_percentage, discount_pct
- "hour" / "time of day" → HOUR(transaction_time) — see TIME-OF-DAY RULES above
- "average transaction value" / "avg order value" → AVG(unit_price * transaction_qty)

- If a price column is a STRING with currency symbols (₹, $, €, £) or commas, clean it first:
  CAST(REPLACE(REPLACE(REPLACE(column_name, '₹', ''), '$', ''), ',', '') AS DOUBLE) as clean_price

FILTER RULES (apply to both types):
- For string filters (State, City, Region, Segment, Category):
  - Use TRIM(LOWER("Column")) = 'lowercase_value'
  - Always add AND "Column" IS NOT NULL
- Match state names against the available states list using exact lowercase spelling
- DISCOUNT SCALE: "Discount" is stored as a decimal between 0 and 1 (e.g. 0.2 = 20%, 0.8 = 80%)
  - "greater than 20% discount" → "Discount" > 0.20
  - "less than 30% discount"    → "Discount" < 0.30
  - "discount of 50%"           → "Discount" = 0.50
  - NEVER use "Discount" > 20 — that would return zero rows
- Use LIMIT only when request says "top N" or "first N" WITHOUT a grouping dimension
- For "top N per group" / "top N by segment/category/region" queries:
  Use a window function with ROW_NUMBER() OVER (PARTITION BY ...) pattern:
  WITH ranked AS (
      SELECT "Customer Name", "Segment", SUM("Sales") as total_sales,
             ROW_NUMBER() OVER (PARTITION BY "Segment" ORDER BY SUM("Sales") DESC) as rnk
      FROM sales_data GROUP BY "Customer Name", "Segment"
  )
  SELECT rnk as rank, "Customer Name", "Segment", total_sales
  FROM ranked WHERE rnk <= N ORDER BY "Segment", rnk
  Apply this pattern whenever the request says "per", "by each", "for each", "within each", or "by [dimension]" alongside a top/best/highest keyword

Request: {normalized}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    sql = response.choices[0].message.content.strip()
    return sql.split(";")[0].strip()

def refine_query(existing_sql: str, new_condition: str, df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    state_col = next((c for c in df.columns if c.lower() == "state"), None)
    available_states = sorted(df[state_col].dropna().unique().tolist()) if state_col else []
    normalized = normalize_request(new_condition)
    prompt = f"""
You are a SQL expert. Modify an existing DuckDB SQL query by adding a new condition.

New condition to apply: "{normalized}"

Existing SQL:
{existing_sql}

Exact column names: {columns}
Available states: {available_states}
Table name: sales_data

Rules:
- Return ONLY the modified SQL query
- No markdown, no backticks, no explanation, no semicolons
- Add the new condition using AND into the existing WHERE clause
- If no WHERE clause exists, add one before GROUP BY / ORDER BY
- Use TRIM(LOWER("Column")) = 'lowercase_value' for string/category filters
- Always add AND "Column" IS NOT NULL alongside any new string filter
- Keep ALL existing SELECT columns, WHERE conditions, GROUP BY, ORDER BY unchanged
- Update LIMIT only if the new condition explicitly changes the count
- CRITICAL: Every column name that contains a space MUST be wrapped in double quotes e.g. "Order ID", "Order Date", "Customer Name", "Product Name", "Sub-Category", "Ship Mode"
- NEVER write column names with spaces without double quotes — this causes a syntax error
- DISCOUNT SCALE: "Discount" is stored as a decimal 0–1 (0.2 = 20%, 0.8 = 80%)
  - "greater than 20%" → "Discount" > 0.20   |   "less than 30%" → "Discount" < 0.30
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    sql = response.choices[0].message.content.strip()
    return sql.split(";")[0].strip()


def make_detail_sql(agg_sql: str) -> str:
    """Convert an aggregate SQL into a SELECT * detail query with the same filters."""
    s = agg_sql.strip()
    s = re.sub(r'(?is)SELECT\s+.+?FROM\s+', 'SELECT * FROM ', s)
    s = re.sub(r'(?i)\s+GROUP\s+BY\s+[^\n;]+', '', s)
    s = re.sub(r'(?i)\s+ORDER\s+BY\s+[^\n;]+', '', s)
    return s.strip() + '\nORDER BY "Order Date" DESC'


def export_to_excel(df: pd.DataFrame, filename: str) -> str:
    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/{filename}.xlsx"
    df.to_excel(path, index=False)
    return path

def handle_data_pull(user_request: str, df_input: pd.DataFrame = None, existing_sql: str = None) -> dict:
    df = df_input if df_input is not None else df_global
    try:
        sql = existing_sql if existing_sql else generate_sql(user_request, df)

        # ── Validate before execution ──────────────────────────────────────
        validation = validate_sql(sql, df)
        if not validation["valid"]:
            return {
                "status": "error",
                "sql": sql,
                "error": validation["error"],
                "warnings": validation.get("warnings", []),
            }
        sql = validation["sql"]  # use auto-fixed SQL if applicable

        sales_data = df
        result_df = duckdb.query(sql).df()

        # Handle 0-row results gracefully
        if len(result_df) == 0:
            return {
                "status": "error",
                "sql": sql,
                "error": "⚠️ The query returned no results. Try adjusting your filters or broadening your question.",
            }

        # Detect aggregate result: exactly 1 row, only numeric cols, no text cols
        is_agg = (
            len(result_df) == 1
            and len(result_df.select_dtypes(include="number").columns) >= 1
            and len(result_df.select_dtypes(include="object").columns) == 0
        )

        detail_df = None
        detail_rows = len(result_df)
        if is_agg:
            try:
                detail_sql = make_detail_sql(sql)
                detail_df = duckdb.query(detail_sql).df()
                detail_rows = len(detail_df)
            except Exception:
                detail_df = None

        filepath = export_to_excel(result_df, "data_pull_result")
        return {
            "status": "success",
            "sql": sql,
            "dataframe": result_df,
            "detail_dataframe": detail_df,
            "filepath": filepath,
            "rows": detail_rows,
            "is_aggregate": is_agg,
        }
    except Exception as e:
        err_str = str(e)
        if "rate_limit_exceeded" in err_str or "429" in err_str or "Rate limit" in err_str:
            wait_match = re.search(r"try again in ([0-9hms\.]+)", err_str)
            wait_time = wait_match.group(1) if wait_match else "a few minutes"
            friendly = (
                f"⏳ **Groq API rate limit reached.** You've used today's free-tier token quota.\n\n"
                f"Please wait **{wait_time}** and try again.\n\n"
                f"💡 *Tip: Upgrade to Groq Dev Tier at console.groq.com for higher limits.*"
            )
            return {"status": "error", "sql": locals().get("sql", ""), "error": friendly}
        return {
            "status": "error",
            "sql": locals().get("sql", ""),
            "error": err_str
        }