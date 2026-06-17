import os
import re
import json
import duckdb
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from validator import validate_sql

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATA_PATH = "data/sales_data.csv"
df_global = pd.read_csv(DATA_PATH, encoding='latin-1')

POSITIVE_KEYWORDS = [
    "performing well", "performing best", "best performing", "top performing",
    "leading", "outperform", "strongest", "best region", "best category",
    "doing well", "doing best", "which is best", "which performs best",
    "performing good", "highest performing", "most profitable", "top region",
    "performing the best", "who is leading", "which region leads",
    "is best", "performs best", "region is best", "category is best",
    "segment is best", "doing the best", "performing well", "excelling",
    "which is performing", "which region performs", "best overall",
    "top overall", "highest profit", "most revenue", "biggest revenue",
]

NEGATIVE_KEYWORDS = [
    "underperform", "worst", "lowest", "weakest", "lagging", "behind",
    "problem", "issue", "struggling", "poor", "bad", "least profitable",
]


def is_positive_question(user_request: str) -> bool:
    """Return True if the question asks about top/best performers, False for underperformers."""
    req_lower = user_request.lower()
    has_positive = any(kw in req_lower for kw in POSITIVE_KEYWORDS)
    has_negative = any(kw in req_lower for kw in NEGATIVE_KEYWORDS)
    # Explicit negative overrides positive (e.g. "which region is NOT performing well")
    if has_negative:
        return False
    return has_positive


def _build_date_rules_analyst(df: pd.DataFrame) -> str:
    """Return the correct DATE RULES block based on actual column types in df."""
    timestamp_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    varchar_date_cols = [
        c for c in df.columns
        if c not in timestamp_cols
        and any(kw in c.lower() for kw in ["date", "time", "month", "year", "period"])
        and df[c].dtype == object
    ]

    if timestamp_cols:
        example_col = timestamp_cols[0]
        return f"""
DATE RULES — these columns are already TIMESTAMP/DATE type (NOT VARCHAR strings): {timestamp_cols}
- NEVER use strptime() on TIMESTAMP columns — it only works on VARCHAR and will throw a Binder Error
- Use strftime() directly on the column:
  - Monthly grouping: strftime({example_col}, '%Y-%m')
  - Yearly grouping : strftime({example_col}, '%Y')
  - Year filter     : strftime({example_col}, '%Y') = '2023'
  - Month filter    : strftime({example_col}, '%Y-%m') = '2023-01'
- Correct  : strftime(transaction_date, '%Y-%m')
- Incorrect: strptime(transaction_date, '%Y-%m-%d') — will cause Binder Error
"""
    elif varchar_date_cols:
        example_col = f'"{varchar_date_cols[0]}"'
        return f"""
DATE RULES — date columns ({varchar_date_cols}) are stored as VARCHAR in M/D/YYYY format (e.g. '11/8/2016'):
- ALWAYS parse with strptime: strptime({example_col}, '%m/%d/%Y')
- Monthly grouping: strftime(strptime({example_col}, '%m/%d/%Y'), '%Y-%m')
- Year filter     : strftime(strptime({example_col}, '%m/%d/%Y'), '%Y') = '2017'
- NEVER use CAST({example_col} AS DATE) — will fail due to M/D/YYYY format
- NEVER use date_part(), YEAR(), or MONTH() directly on {example_col}
"""
    else:
        return ""


def generate_analysis_sql(user_request: str, df: pd.DataFrame, positive: bool = False) -> str:
    columns = df.columns.tolist()
    sample = df.head(2).to_string(index=False)
    order_rule = (
        "Order by profit_margin_pct DESC (best performer first) — question is about top/best performers"
        if positive else
        "Order by profit_margin_pct ASC (worst performer first) — question is about underperformance"
    )
    date_rules = _build_date_rules_analyst(df)
    prompt = f"""
You are a SQL expert. Write ONE single DuckDB SQL query to answer this business question.

Exact column names (use these exactly):
{columns}

Sample data:
{sample}

Table name: sales_data (already loaded as pandas dataframe)

CRITICAL RULES — follow exactly:
- Return ONLY one single SQL query
- No markdown, no backticks, no explanation, no semicolons
- Use double quotes around column names that have spaces
- ALWAYS use GROUP BY and aggregate functions — NEVER return raw rows
- NEVER select "Customer Name", "Product Name", "Order ID", or any row-level field
- CRITICAL: Every non-aggregated column in SELECT must appear in GROUP BY
  WRONG: SELECT store_location, SUM(...) FROM sales_data
  RIGHT: SELECT store_location, SUM(...) FROM sales_data GROUP BY store_location

{date_rules}

STANDARD KPI SET (use for all comparison questions):
  SUM("Sales") as total_sales,
  SUM("Profit") as total_profit,
  ROUND(SUM("Profit") * 100.0 / NULLIF(SUM("Sales"), 0), 1) as profit_margin_pct,
  ROUND(AVG(COALESCE("Discount", 0)) * 100, 1) as avg_discount_pct,
  COUNT(DISTINCT "Order ID") as order_count

COLUMN MAPPING — when exact column name is not present:
- "sales" / "revenue" → Sales, Revenue, Amount, selling_price, discounted_price, actual_price, or price * rating_count
- "profit" → Profit, Margin, profit_margin
- "discount" → Discount, discount_percentage
- "quantity" → Quantity, Qty, rating_count, units_sold
- If price columns are STRINGS with symbols (₹, $, £, €) or commas, clean first:
  CAST(REPLACE(REPLACE(REPLACE(col, '₹',''), '$',''), ',','') AS DOUBLE)
- Use the best available proxy — never fail just because an exact column name is missing

DIMENSION RULES:
- Region questions    → GROUP BY "Region" WHERE "Region" IS NOT NULL
- Category questions  → GROUP BY "Category" WHERE "Category" IS NOT NULL
- Sub-category        → GROUP BY "Sub-Category" WHERE "Sub-Category" IS NOT NULL
- Segment questions   → GROUP BY "Segment" WHERE "Segment" IS NOT NULL
- Time/trend          → GROUP BY strftime("Order Date", '%Y-%m') ORDER BY 1 ASC
- Product questions   → GROUP BY "Product Name" ORDER BY total_profit ASC LIMIT 10

SAFETY RULES:
- Use NULLIF(SUM("Sales"), 0) in every division to prevent divide-by-zero errors
- Use COALESCE(column, 0) on nullable numeric columns
- Always filter IS NOT NULL on the GROUP BY column
- Return ALL groups — no filtering on dimension values, no arbitrary LIMIT
- {order_rule}

Question: {user_request}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    sql = response.choices[0].message.content.strip()
    return sql.split(";")[0].strip()


def _precompute_analysis_facts(df: pd.DataFrame, positive: bool) -> str:
    """Pre-compute all KPIs and rankings from the DataFrame in Python.
    The LLM receives only these facts — it never calculates anything itself."""
    lines = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols    = df.select_dtypes(include="object").columns.tolist()
    dim = text_cols[0] if text_cols else None

    if len(df) == 0:
        return "No data returned by the query."

    for num_col in numeric_cols:
        col_series = df[num_col].dropna()
        if len(col_series) == 0:
            lines.append(f"\n[{num_col}]: all values are null — skipped")
            continue
        col_total = col_series.sum()
        col_avg   = col_series.mean()
        col_max   = col_series.max()
        col_min   = col_series.min()
        lines.append(f"\n[{num_col}]")
        lines.append(f"  Total : {col_total:,.2f}")
        lines.append(f"  Avg   : {col_avg:,.2f}")
        lines.append(f"  Max   : {col_max:,.2f}")
        lines.append(f"  Min   : {col_min:,.2f}")

        if dim:
            try:
                dim_series = df[dim].dropna()
                if len(dim_series) == 0:
                    continue
                ranked = df.groupby(dim)[num_col].sum().sort_values(ascending=not positive)
                lines.append(f"  Ranked by {dim} ({'DESC' if positive else 'ASC'}):")
                for rank_i, (name, val) in enumerate(ranked.items(), 1):
                    pct = (val / col_total * 100) if col_total != 0 else 0
                    lines.append(f"    #{rank_i}: {name} = {val:,.2f}  ({pct:.1f}% of total)")
            except Exception:
                pass

    # Row-level data — cap at 50 rows to avoid token overflow
    lines.append("\nFull data table (up to 50 rows):")
    lines.append(df.head(50).to_string(index=False))
    return "\n".join(lines)


def interpret_results(user_request: str, df: pd.DataFrame, kpi: str = "", positive: bool = False) -> dict:
    columns = df.columns.tolist()
    # ── Pre-compute ALL facts in Python — LLM never calculates ────────────────
    computed_facts = _precompute_analysis_facts(df, positive)

    if positive:
        summary_rule   = "State the top performer and its key strength with one real number. Max 15 words."
        section_ex     = "e.g. 'Why West Region Leads'"
        finding_titles = ["'Highest Profitability'", "'Low Discounting'", "'Strong Sales Volume'"]
        causes_label   = "key_drivers"
        causes_rule    = "max 3, each under 7 words — WHY this performer leads"
        rec_rule       = "max 3, each under 10 words, start with action verb — how OTHER regions can replicate this"
        bench_rule     = "identify the top performer and its single strongest metric"
        summary_named  = "must name the BEST performer and include one real number from the data"
    else:
        summary_rule   = "State the key problem with one real number. Max 15 words."
        section_ex     = "e.g. 'Why Central Region Underperforms'"
        finding_titles = ["'Lowest Profitability'", "'Heavy Discounting'", "'Product Mix Issue'"]
        causes_label   = "root_causes"
        causes_rule    = "max 3, each under 7 words, no filler words"
        rec_rule       = "max 3, each under 10 words, start with action verb"
        bench_rule     = "identify the best performer and one standout metric"
        summary_named  = "must name the worst performer and include one real number from the data"

    prompt = f"""
You are a Senior Data Analyst writing a concise executive briefing.

YOUR ROLE:
- Read the pre-computed facts below
- Interpret and explain them in structured JSON
- You are an INTERPRETER, NOT a calculator

ABSOLUTE RULES — violating any rule makes the response invalid:
- Use ONLY the numbers from "Pre-computed facts" — copy them exactly as shown
- Do NOT generate, calculate, estimate, or round any number yourself
- Do NOT add metrics, percentages, or values not present in the facts
- If a fact is unavailable, write "data not available" — never invent a substitute
- Every bullet "Label: value" must use an exact value from the pre-computed facts

Question: "{user_request}"
Columns available: {columns}

Pre-computed facts (these are the ONLY numbers you may use):
{computed_facts}

Return ONLY this JSON (no markdown, no backticks):
{{
  "summary": "One sentence. {summary_rule}",
  "section_title": "3-6 words {section_ex}",
  "key_findings": [
    {{
      "title": "2-4 word title e.g. {finding_titles[0]}",
      "explanation": "One sentence using exact numbers from the data above.",
      "bullets": ["Label: [exact value from data]", "Label: [exact value from data]", "Label: [exact value from data]"],
      "conclusion": "One sentence business takeaway supported by the data."
    }},
    {{
      "title": "2-4 word title e.g. {finding_titles[1]}",
      "explanation": "One sentence using exact numbers from the data above.",
      "bullets": ["Label: [exact value from data]", "Label: [exact value from data]", "Label: [exact value from data]"],
      "conclusion": "One sentence business takeaway supported by the data."
    }},
    {{
      "title": "2-4 word title e.g. {finding_titles[2]}",
      "explanation": "One sentence using exact numbers from the data above.",
      "bullets": ["Label: [exact value from data]", "Label: [exact value from data]", "Label: [exact value from data]"],
      "conclusion": "One sentence business takeaway supported by the data."
    }}
  ],
  "{causes_label}": ["Under 7 words, data-backed", "Under 7 words, data-backed", "Under 7 words, data-backed"],
  "recommendations": ["Action verb + specific what", "Action verb + specific what", "Action verb + specific what"],
  "benchmark": "{'Top' if positive else 'Best'} performer: [Name from data] — [exact metric and value from data]."
}}

Rules:
- summary: {summary_named}
- key_findings: exactly 3 findings — focus on {'strengths and drivers of success' if positive else 'problems and weaknesses'}
- bullets: exactly 3 per finding, format "Label: value" — values must match the data exactly
- {causes_label}: {causes_rule}
- recommendations: {rec_rule}
- benchmark: {bench_rule} — use exact numbers from data
- Return ONLY the raw JSON object — no markdown, no backticks, no bold (**), no extra text before or after
- Start your response with {{ and end with }} — nothing else
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    try:
        # Remove markdown bold wrappers e.g. **{...}** or **json{...}**
        clean = re.sub(r'^\*+\s*', '', raw)
        clean = re.sub(r'\s*\*+$', '', clean)
        # Strip code fences: ```json, ```JSON, ``` etc.
        clean = re.sub(r"^```[a-zA-Z]*\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()
        # Extract the first { ... } JSON object
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if json_match:
            clean = json_match.group(0)
        result = json.loads(clean)
        # Ensure required keys exist with safe defaults
        result.setdefault("summary", "Analysis complete.")
        result.setdefault("section_title", "Analysis")
        result.setdefault("key_findings", [])
        result.setdefault("recommendations", [])
        result.setdefault("benchmark", "")
        if "key_drivers" in result and "root_causes" not in result:
            result["root_causes"] = result.pop("key_drivers")
        result.setdefault("root_causes", [])
        result["_positive"] = positive
        return result
    except Exception:
        # JSON completely failed — build a readable natural-language fallback
        # from whatever the LLM returned, stripping any JSON artifacts
        clean_text = re.sub(r'[{}"\\]', '', raw)
        clean_text = re.sub(r'\b(summary|section_title|key_findings|root_causes|recommendations|benchmark|explanation|bullets|conclusion|title)\s*:', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        # Split into sentences for bullets
        sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if len(s.strip()) > 20]
        findings = [{"title": "Finding", "explanation": s + ".", "bullets": [], "conclusion": ""} for s in sentences[:3]]
        return {
            "summary": sentences[0] + "." if sentences else "Analysis could not be formatted.",
            "section_title": "Analysis Summary",
            "key_findings": findings,
            "root_causes": [],
            "recommendations": [],
            "benchmark": "",
            "_positive": positive,
        }


def handle_why_question(user_request: str, df_input: pd.DataFrame = None, kpi: str = "", existing_sql: str = None) -> dict:
    df = df_input if df_input is not None else df_global
    positive = is_positive_question(user_request)
    try:
        sql = existing_sql if existing_sql else generate_analysis_sql(user_request, df, positive=positive)

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
        explanation = interpret_results(user_request, result_df, kpi, positive=positive)
        return {
            "status": "success",
            "sql": sql,
            "dataframe": result_df,
            "explanation": explanation,
            "positive": positive,
        }
    except Exception as e:
        import re as _re
        err_str = str(e)
        if "rate_limit_exceeded" in err_str or "429" in err_str or "Rate limit" in err_str:
            wait_match = _re.search(r"try again in ([0-9hms\.]+)", err_str)
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
