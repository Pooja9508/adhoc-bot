import os
import json
import duckdb
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

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


def interpret_results(user_request: str, df: pd.DataFrame, kpi: str = "", positive: bool = False) -> dict:
    data_summary = df.to_string(index=False)
    columns = df.columns.tolist()

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
You are a Senior Data Analyst writing a concise executive briefing. One glance = full picture.

Question: "{user_request}"
Columns: {columns}
Data:
{data_summary}

Return ONLY this JSON (no markdown, no backticks):
{{
  "summary": "One sentence. {summary_rule}",
  "section_title": "3-6 words {section_ex}",
  "key_findings": [
    {{
      "title": "2-4 word title e.g. {finding_titles[0]}",
      "explanation": "One sentence explaining this finding with a specific number.",
      "bullets": ["Label: value", "Label: value", "Label: value"],
      "conclusion": "One sentence takeaway or impact."
    }},
    {{
      "title": "2-4 word title e.g. {finding_titles[1]}",
      "explanation": "One sentence explaining this finding with a specific number.",
      "bullets": ["Label: value", "Label: value", "Label: value"],
      "conclusion": "One sentence takeaway or impact."
    }},
    {{
      "title": "2-4 word title e.g. {finding_titles[2]}",
      "explanation": "One sentence explaining this finding with a specific number.",
      "bullets": ["Label: value", "Label: value", "Label: value"],
      "conclusion": "One sentence takeaway or impact."
    }}
  ],
  "{causes_label}": ["Under 7 words", "Under 7 words", "Under 7 words"],
  "recommendations": ["Action verb + specific what", "Action verb + specific what", "Action verb + specific what"],
  "benchmark": "{'Top' if positive else 'Best'} performer: [Name] — [key metric and value]."
}}

Rules:
- summary: {summary_named}
- key_findings: exactly 3 findings — focus on {'strengths and drivers of success' if positive else 'problems and weaknesses'}
- bullets: exactly 3 per finding, format "Label: value" only
- {causes_label}: {causes_rule}
- recommendations: {rec_rule}
- benchmark: {bench_rule}
- ONLY valid JSON
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        result = json.loads(raw)
        # Normalise key: both "root_causes" and "key_drivers" → stored as "root_causes"
        if "key_drivers" in result and "root_causes" not in result:
            result["root_causes"] = result.pop("key_drivers")
        result["_positive"] = positive  # carry the flag through to the UI
        return result
    except Exception:
        return {
            "summary": raw,
            "section_title": "Analysis",
            "key_findings": [],
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
