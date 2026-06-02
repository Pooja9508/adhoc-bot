import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
from dotenv import load_dotenv
from sql_agent import refine_query
import os
import io
import re
from triage import classify_request
from sql_agent import handle_data_pull, df_global
from analyst_agent import handle_why_question
from validator import validate_chart_dimensions


def split_top_level_commas(s: str) -> list:
    """Split on commas that are NOT inside parentheses."""
    parts, depth, current = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def format_sql(sql: str) -> str:
    clause_keywords = ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY",
                       "HAVING", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "JOIN", "LIMIT", "WITH"]
    # Insert newline before each clause keyword (whole-word, case-insensitive)
    formatted = sql.strip()
    for kw in clause_keywords:
        formatted = re.sub(rf"(?i)\b({re.escape(kw)})\b", r"\n\1", formatted)

    result = []
    for line in formatted.splitlines():
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        is_clause = any(upper.startswith(kw) for kw in clause_keywords)

        if upper.startswith("SELECT"):
            after_select = line[6:].strip()  # everything after SELECT
            if not after_select:
                result.append("SELECT")
                continue
            cols = split_top_level_commas(after_select)
            if not cols:
                result.append(line)
                continue
            result.append("SELECT " + cols[0] + (",") * (len(cols) > 1))
            for col in cols[1:-1]:
                result.append("    " + col + ",")
            if len(cols) > 1:
                result.append("    " + cols[-1])
        elif is_clause:
            result.append(line)
        else:
            result.append("    " + line)

    return "\n".join(result)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Ad-Hoc Data Request Bot", page_icon="🤖", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Page background ── */
[data-testid="stAppViewContainer"] {
    background: #f8f9fb;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Hide Streamlit default branding ── */
#MainMenu, footer { visibility: hidden; }

/* ── Main content max-width wrapper ── */
.block-container {
    max-width: 960px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ── Page header ── */
.app-header {
    background: linear-gradient(135deg, #1a237e 0%, #1565c0 60%, #0288d1 100%);
    border-radius: 14px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.8rem;
    color: white;
}
.app-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
    color: white;
}
.app-header p {
    font-size: 0.95rem;
    opacity: 0.85;
    margin: 0;
    color: white;
}
.badge-row { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: white;
}

/* ── Form card ── */
[data-testid="stForm"] {
    background: white;
    border: 1px solid #e0e4ef;
    border-radius: 12px;
    padding: 1.6rem 1.8rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* ── Input labels ── */
label[data-testid="stWidgetLabel"] p {
    font-weight: 600;
    font-size: 0.85rem;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Text input & textarea ── */
input[type="text"], textarea {
    border-radius: 8px !important;
    border: 1.5px solid #d1d5db !important;
    font-size: 0.95rem !important;
    background: #fafafa !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #1565c0 !important;
    box-shadow: 0 0 0 3px rgba(21,101,192,0.12) !important;
    background: white !important;
}

/* ── Submit button ── */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #1565c0, #0288d1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.2rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    white-space: nowrap !important;
    transition: opacity 0.2s !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    opacity: 0.88 !important;
}

/* ── Divider ── */
hr { border-color: #e5e7eb !important; margin: 1.5rem 0 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e0e4ef;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
[data-testid="metric-container"] label {
    font-size: 0.78rem !important;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #1a237e !important;
}

/* ── Section subheaders ── */
h2, h3 { color: #1a237e !important; }

/* ── Info / success / error boxes ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #e0e4ef !important;
    border-radius: 10px !important;
    background: white !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid #e0e4ef !important;
}

/* ── Refine bar ── */
.refine-bar {
    background: white;
    border: 1.5px solid #e0e4ef;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 0.5rem;
}


/* ── Caption / footer ── */
.footer-text {
    text-align: center;
    color: #9ca3af;
    font-size: 0.78rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>📊 Ad-Hoc Data Request Bot</h1>
  <p>Ask any business question in plain English — get instant answers, charts, and downloadable data.</p>
  <div class="badge-row">
    <span class="badge">✦ Powered by LLaMA 3.3</span>
    <span class="badge">✦ Superstore Dataset</span>
    <span class="badge">✦ SQL + AI Analysis</span>
  </div>
</div>
""", unsafe_allow_html=True)

if "step" not in st.session_state:
    st.session_state.step = "input"
if "request" not in st.session_state:
    st.session_state.request = ""
if "name" not in st.session_state:
    st.session_state.name = ""
if "df_input" not in st.session_state:
    st.session_state.df_input = None
if "request_type" not in st.session_state:
    st.session_state.request_type = ""
if "refined_sql" not in st.session_state:
    st.session_state.refined_sql = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cached_result" not in st.session_state:
    st.session_state.cached_result = None
if "result_history" not in st.session_state:
    st.session_state.result_history = []
if "refinement_type_override" not in st.session_state:
    st.session_state.refinement_type_override = None
if "refinement_request_override" not in st.session_state:
    st.session_state.refinement_request_override = None
if "refine_input_counter" not in st.session_state:
    st.session_state.refine_input_counter = 0
if "main_form_counter" not in st.session_state:
    st.session_state.main_form_counter = 0

KPI_OPTIONS = ["All", "Sales", "Profit", "Quantity", "Discount"]
KPI_KEYWORDS = {
    "Sales":    ["sales", "revenue", "total_sales"],
    "Profit":   ["profit", "total_profit", "net_profit"],
    "Quantity": ["quantity", "qty", "units"],
    "Discount": ["discount", "avg_discount", "discount_pct"],
}


def find_kpi_column(df: pd.DataFrame, kpi: str) -> str:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None
    if kpi == "All":
        return numeric_cols[0]
    keywords = KPI_KEYWORDS.get(kpi, [kpi.lower()])
    for col in numeric_cols:
        for kw in keywords:
            if kw in col.lower():
                return col
    return numeric_cols[0]


def user_wants_visual(user_request: str) -> bool:
    visual_keywords = [
        "chart", "graph", "plot", "visual", "visualize", "pie",
        "bar", "line", "scatter", "histogram", "treemap", "funnel", "box", "area"
    ]
    return any(word in user_request.lower() for word in visual_keywords)


def detect_chart_type(user_request: str, df: pd.DataFrame) -> str:
    """Rule-based chart type detection — checks request keywords then data shape."""
    req = user_request.lower()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols    = df.select_dtypes(include="object").columns.tolist()
    num_rows = len(df)
    if num_rows == 0 or not numeric_cols:
        return "bar"
    num_categories = df[text_cols[0]].nunique() if text_cols else 0

    # ── 1. Explicit chart type requested ─────────────────────────────────────
    explicit = {
        "pie chart": "pie", "bar chart": "bar", "line chart": "line",
        "scatter": "scatter", "histogram": "histogram", "treemap": "treemap",
        "funnel": "funnel", "area chart": "area", "box plot": "box",
    }
    for kw, ct in explicit.items():
        if kw in req:
            return ct

    # ── 2. Time-series detection ──────────────────────────────────────────────
    time_request_kws = ["trend", "over time", "monthly", "yearly", "by month",
                        "by year", "by quarter", "by week", "growth", "change over",
                        "year over year", "month over month", "daily", "weekly"]
    if any(kw in req for kw in time_request_kws):
        return "line"

    # Check if the x-axis column looks like a date/period string
    if text_cols and num_rows > 0:
        sample = str(df[text_cols[0]].dropna().iloc[0]) if len(df[text_cols[0]].dropna()) > 0 else ""
        x_col_lower = text_cols[0].lower()
        is_time_col = (
            bool(re.search(r'\d{4}[-/]\d{2}', sample))          # "2023-01" or "2023/01"
            or bool(re.search(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', sample.lower()))
            or any(kw in x_col_lower for kw in ["date", "month", "year", "quarter", "week", "period", "time"])
        )
        if is_time_col:
            return "line"

    # ── 3. Distribution ───────────────────────────────────────────────────────
    if any(kw in req for kw in ["distribution", "spread", "frequency", "histogram"]):
        return "histogram"

    # ── 4. Correlation / relationship ─────────────────────────────────────────
    if any(kw in req for kw in [" vs ", " versus ", "correlation", "relationship between",
                                 "scatter", "compared to"]):
        return "scatter"

    # ── 5. Part-of-whole ─────────────────────────────────────────────────────
    if any(kw in req for kw in ["share", "proportion", "breakdown", "split",
                                 "contribution", "percentage of", "pie", "makeup",
                                 "how much of", "what percent"]):
        return "pie" if num_categories <= 7 else "treemap"

    # ── 6. Hierarchy / nested ─────────────────────────────────────────────────
    if any(kw in req for kw in ["hierarchy", "treemap", "nested", "drill"]):
        return "treemap"

    # ── 7. Funnel / stages ────────────────────────────────────────────────────
    if any(kw in req for kw in ["funnel", "conversion", "stages", "pipeline", "drop-off"]):
        return "funnel"

    # ── 8. Box / outliers ────────────────────────────────────────────────────
    if any(kw in req for kw in ["outlier", "quartile", "box", "whisker", "variance"]):
        return "box"

    # ── 9. Cumulative / running ───────────────────────────────────────────────
    if any(kw in req for kw in ["cumulative", "running total", "area"]):
        return "area"

    # ── 10. Data-shape fallbacks ──────────────────────────────────────────────
    # Few categories + single metric → pie is more readable than bar
    if num_categories <= 5 and len(numeric_cols) == 1 and num_rows <= 5:
        return "pie"

    # Many rows with a text dimension → horizontal bar is cleaner
    if num_rows > 15:
        return "bar"

    # Default → bar
    return "bar"


def _build_fig(df, chart_type, x, y, title):
    if chart_type == "bar":
        fig = px.bar(df, x=x, y=y, title=title, color=x, text=y,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_tickangle=-30, uniformtext_minsize=8)
    elif chart_type == "line":
        fig = px.line(df, x=x, y=y, title=title, markers=True,
                      color_discrete_sequence=["#2196F3"])
        fig.update_traces(line_width=2, marker_size=8)
    elif chart_type == "pie":
        if df[x].nunique() <= 8:
            fig = px.pie(df, names=x, values=y, title=title, hole=0.35,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(textposition="inside", textinfo="percent+label")
        else:
            fig = px.bar(df, x=x, y=y, title=title, color=x,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False)
    elif chart_type == "scatter":
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        y2 = numeric_cols[1] if len(numeric_cols) >= 2 else numeric_cols[0] if numeric_cols else y
        fig = px.scatter(df, x=y, y=y2, color=x, title=title, size_max=15,
                         color_discrete_sequence=px.colors.qualitative.Set2)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=y, title=title, nbins=20,
                           color_discrete_sequence=["#2196F3"])
    elif chart_type == "box":
        fig = px.box(df, x=x, y=y, title=title,
                     color_discrete_sequence=px.colors.qualitative.Set2)
    elif chart_type == "area":
        fig = px.area(df, x=x, y=y, title=title,
                      color_discrete_sequence=["#2196F3"])
    elif chart_type == "treemap":
        fig = px.treemap(df, path=[x], values=y, title=title,
                         color_discrete_sequence=px.colors.qualitative.Set2)
    elif chart_type == "funnel":
        fig = px.funnel(df, x=y, y=x, title=title,
                        color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        fig = px.bar(df, x=x, y=y, title=title, color=x,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False)
    fig.update_layout(
        title_font_size=16,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig


def render_chart(df: pd.DataFrame, user_request: str, title: str = "Chart"):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if len(df) == 0 or not numeric_cols:
        st.caption("Not enough data to render a chart.")
        return
    chart_check = validate_chart_dimensions(df)
    if not chart_check["valid"]:
        st.caption(chart_check.get("warning", "Not enough data to render a chart."))
        return
    if len(df) < 2:
        st.caption("Not enough data to render a chart.")
        return
    if len(df) > 50:
        df = df.head(50)
    chart_type = detect_chart_type(user_request, df)
    # For all-numeric results use the first numeric col as x-axis
    x = text_cols[0] if text_cols else (numeric_cols[0] if len(numeric_cols) > 1 else None)
    y = numeric_cols[0] if numeric_cols else None
    if not x or not y or x == y:
        st.caption("No suitable columns found for chart.")
        return
    # Time-series charts must be sorted by X (date/period), not by Y value
    if chart_type in ("line", "area"):
        df = df.copy().sort_values(by=x, ascending=True).reset_index(drop=True)
    else:
        df = df.copy().sort_values(by=y, ascending=False).reset_index(drop=True)
    try:
        fig = _build_fig(df, chart_type, x, y, title)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Could not render chart: {e}")


def render_chart_for_kpi(df: pd.DataFrame, user_request: str, kpi_col: str, title: str):
    text_cols = df.select_dtypes(include="object").columns.tolist()
    x = text_cols[0] if text_cols else None
    if not x or not kpi_col or kpi_col not in df.columns:
        render_chart(df, user_request, title)
        return
    chart_type = detect_chart_type(user_request, df)
    # Sort by X for time-series, by KPI value for everything else
    if chart_type in ("line", "area"):
        df_plot = df.copy().sort_values(by=x, ascending=True).reset_index(drop=True)
    else:
        df_plot = df.copy().sort_values(by=kpi_col, ascending=False).reset_index(drop=True)
    col_label = kpi_col.replace("_", " ").title()
    try:
        fig = _build_fig(df_plot, chart_type, x, kpi_col, f"{col_label} by {x.replace('_', ' ').title()}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Could not render chart: {e}")


def clean_col_name(col: str) -> str:
    import re
    col = re.sub(r"^sum\((.+)\)$", r"Total \1", col, flags=re.IGNORECASE)
    col = re.sub(r"^avg\((.+)\)$", r"Avg \1", col, flags=re.IGNORECASE)
    col = col.replace("_pct", "").replace("_", " ")
    return col.title()


def fmt_value(val, col_name: str):
    col_lower = col_name.lower()
    is_pct = any(k in col_lower for k in ["margin", "pct", "discount", "rate", "percent"])
    is_currency = any(k in col_lower for k in ["sales", "profit", "revenue", "amount", "cost"])
    try:
        v = float(val)
    except (TypeError, ValueError):
        return val
    import math
    if math.isnan(v) or math.isinf(v):
        return "—"
    if is_pct:
        if -1.0 <= v <= 1.0:
            return f"{v * 100:.1f}%"
        return f"{v:.1f}%"
    if is_currency:
        return f"${v:,.0f}"
    return f"{v:,.0f}"


def format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    display.columns = [clean_col_name(c) for c in display.columns]
    numeric_cols = display.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        display[col] = display[col].apply(lambda v: fmt_value(v, col))
    return display


def highlight_and_format(df: pd.DataFrame):
    """Apply min/max highlighting AND value formatting in one styler pass.
    Must receive the raw numeric dataframe (before format_display_df)."""
    display = df.copy()
    display.columns = [clean_col_name(c) for c in display.columns]
    numeric_cols = display.select_dtypes(include="number").columns.tolist()

    def _style(col):
        if col.name not in numeric_cols:
            return [""] * len(col)
        styles = [""] * len(col)
        try:
            non_null = col.dropna()
            if len(non_null) < 2 or non_null.nunique() < 2:
                return styles  # skip all-null or all-same-value columns
            max_i = col.idxmax()
            min_i = col.idxmin()
            styles[max_i] = "background-color: #d4edda; font-weight: bold"
            if min_i != max_i:
                styles[min_i] = "background-color: #f8d7da; font-weight: bold"
        except Exception:
            pass
        return styles

    # Format numeric values as currency / % / number — applied after styling
    fmt_dict = {col: (lambda v, c=col: fmt_value(v, c)) for col in numeric_cols}
    return display.style.apply(_style, axis=0).format(fmt_dict)


def render_analysis(explanation: dict, df: pd.DataFrame, user_request: str,
                    sql: str, name: str, df_raw: pd.DataFrame = None, idx: int = 0):
    if name:
        st.caption(f"Request by: {name}")

    # ── KPI selector ─────────────────────────────────────────────────────────
    st.markdown("**Select the KPI you want to analyze:**")
    selected_kpi = st.radio("", KPI_OPTIONS, horizontal=True,
                            label_visibility="collapsed", key=f"kpi_radio_{idx}")
    st.divider()

    # 1. Summary
    if explanation.get("summary"):
        st.info(f"**{explanation['summary']}**")

    # 2. Section heading + comparison table
    if explanation.get("section_title"):
        st.subheader(explanation["section_title"])

    st.dataframe(highlight_and_format(df), use_container_width=True)
    st.caption("🟢 Green = highest &nbsp;&nbsp; 🔴 Red = lowest")

    st.divider()

    # 3. Key Findings — numbered, with explanation
    if explanation.get("key_findings"):
        st.markdown("## Key Findings")
        for i, finding in enumerate(explanation["key_findings"], 1):
            st.markdown(f"**{i}. {finding.get('title', '')}**")
            if finding.get("explanation"):
                st.markdown(finding["explanation"])
            st.divider()

    # 4. Root Cause Summary / Key Drivers
    if explanation.get("root_causes"):
        is_positive = explanation.get("_positive", False)
        section = explanation.get("section_title", "")
        subject = (
            section.replace("Why ", "")
                   .split(" is ")[0]
                   .split(" Underperform")[0]
                   .split(" Lead")[0]
                   .split(" Excel")[0]
                   .strip()
        )
        if is_positive:
            st.markdown("## Key Drivers")
            st.markdown(f"**{subject}** leads because of:")
        else:
            st.markdown("## Root Cause Summary")
            st.markdown(f"The **{subject}** underperforms because of:")
        for cause in explanation["root_causes"]:
            st.markdown(f"• {cause}")
        st.divider()

    # 5. Recommended Actions
    if explanation.get("recommendations"):
        st.markdown("## Recommended Actions")
        for i, rec in enumerate(explanation["recommendations"], 1):
            st.markdown(f"{i}. {rec}")
        st.divider()

    # 6. Benchmark
    if explanation.get("benchmark"):
        is_positive = explanation.get("_positive", False)
        label = "⭐ Top Performer" if is_positive else "🏆 Best Performer Benchmark"
        st.success(f"**{label}:** {explanation['benchmark']}")

    st.divider()

    # 7. Supporting Chart
    kpi_col = find_kpi_column(df, selected_kpi)
    with st.expander("📊 Supporting Chart"):
        if selected_kpi == "All":
            render_chart(df, user_request, explanation.get("section_title", "Analysis Chart"))
        else:
            render_chart_for_kpi(df, user_request, kpi_col,
                                 explanation.get("section_title", "Analysis Chart"))

    # 8. SQL Query Used
    with st.expander("🔍 SQL Query Used"):
        st.markdown("**Query powering this analysis:**")
        st.code(format_sql(sql), language="sql")

    # 9. Download Supporting Data (tabs)
    with st.expander("📥 Download Supporting Data"):
        tab_labels = ["📊 Comparison Table"]
        if df_raw is not None:
            tab_labels.append("📋 Raw Data")
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            st.markdown(f"**All results — multi-KPI comparison ({len(df)} rows)**")
            st.dataframe(format_display_df(df), use_container_width=True)
            buf = io.BytesIO()
            format_display_df(df).to_excel(buf, index=False)
            st.download_button(
                "Download Comparison Table",
                buf.getvalue(),
                file_name="comparison_table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_comparison_{idx}",
            )

        if df_raw is not None and len(tabs) > 1:
            with tabs[1]:
                st.markdown(f"**Full dataset — {len(df_raw):,} rows**")
                st.dataframe(df_raw.head(500), use_container_width=True)
                buf2 = io.BytesIO()
                df_raw.to_excel(buf2, index=False)
                st.download_button(
                    "Download Raw Data",
                    buf2.getvalue(),
                    file_name="raw_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_raw_{idx}",
                )

    st.divider()
    st.markdown('<p class="footer-text">🤖 Ad-Hoc Data Request Bot &nbsp;·&nbsp; Built with Streamlit · Groq · LLaMA 3.3 · DuckDB</p>', unsafe_allow_html=True)


def render_grouped_table(df: pd.DataFrame) -> bool:
    """Render a per-group ranked result as a styled HTML table with colored group headers.
    Returns True if the grouped layout was used, False if it should fall back to plain table."""
    rank_col = next((c for c in df.columns if c.lower() in ("rank", "rnk")), None)
    text_cols = df.select_dtypes(include="object").columns.tolist()

    group_col = None
    for col in text_cols:
        if df[col].duplicated().any():
            group_col = col
            break

    if not group_col or not rank_col:
        return False

    group_colors = [
        ("#1a237e", "#e8eaf6"), ("#1565c0", "#e3f2fd"),
        ("#00695c", "#e0f2f1"), ("#6a1b9a", "#f3e5f5"),
        ("#bf360c", "#fbe9e7"), ("#e65100", "#fff3e0"),
    ]
    groups = df[group_col].unique()
    display_cols = [c for c in df.columns if c != group_col]
    clean_headers = [clean_col_name(c) for c in display_cols]
    num_cols = len(display_cols)

    header_cells = "".join(
        f'<th style="padding:10px 16px;text-align:left;font-size:0.78rem;'
        f'text-transform:uppercase;letter-spacing:0.5px;color:#6b7280;font-weight:600;">{h}</th>'
        for h in clean_headers
    )

    rows_html = ""
    for i, group in enumerate(groups):
        bg_dark, bg_light = group_colors[i % len(group_colors)]
        group_df = df[df[group_col] == group].reset_index(drop=True)

        rows_html += (
            f'<tr><td colspan="{num_cols}" style="background:{bg_dark};color:white;'
            f'padding:9px 16px;font-weight:700;font-size:0.88rem;letter-spacing:0.3px;">'
            f'▸ {group}</td></tr>'
        )

        for row_i, (_, row) in enumerate(group_df.iterrows()):
            row_bg = "white" if row_i % 2 == 0 else "#fafbfc"
            cells = ""
            for col in display_cols:
                raw = row[col]
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = fmt_value(raw, col)
                    align = "right"
                else:
                    val = str(raw) if pd.notna(raw) else "—"
                    align = "left"
                weight = "700" if col.lower() in ("rank", "rnk") else "400"
                cells += (
                    f'<td style="padding:9px 16px;border-bottom:1px solid #f0f2f5;'
                    f'text-align:{align};font-weight:{weight};">{val}</td>'
                )
            rows_html += f'<tr style="background:{row_bg};">{cells}</tr>'

        rows_html += (
            f'<tr><td colspan="{num_cols}" style="height:4px;background:{bg_light};"></td></tr>'
        )

    html = f"""
    <div style="border:1px solid #e0e4ef;border-radius:12px;overflow:hidden;
                margin-bottom:1rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
      <table style="width:100%;border-collapse:collapse;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:0.9rem;">
        <thead>
          <tr style="background:#f8f9fb;border-bottom:2px solid #e0e4ef;">{header_cells}</tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    return True


def _fmt_dimension_val(col: str, val) -> str:
    """Format dimension values — e.g. hour integers → AM/PM labels."""
    if col.lower() in ("hour_of_day", "hour", "sale_hour"):
        try:
            h = int(val)
            if h == 0:   return "12 AM"
            if h < 12:   return f"{h} AM"
            if h == 12:  return "12 PM"
            return f"{h - 12} PM"
        except Exception:
            pass
    return str(val)


def _best_dimension(df: pd.DataFrame) -> str | None:
    """Pick the most meaningful text column for grouping.
    Prefers columns with 2–50 unique values (granular but not row-level IDs)."""
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if not text_cols:
        return None
    # Score: prefer columns whose unique count is between 2 and 50
    # Among those, prefer higher unique count (more granular)
    candidates = [(c, df[c].nunique()) for c in text_cols]
    # Filter to meaningful range
    meaningful = [(c, n) for c, n in candidates if 2 <= n <= 50]
    if meaningful:
        return max(meaningful, key=lambda x: x[1])[0]
    # Fallback: column with most unique values that isn't every row unique
    non_unique_id = [(c, n) for c, n in candidates if n < len(df)]
    if non_unique_id:
        return max(non_unique_id, key=lambda x: x[1])[0]
    return text_cols[0]


def _precompute_facts(df: pd.DataFrame) -> str:
    """Compute all facts from the DataFrame in Python. LLM only explains these."""
    lines = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    dim = _best_dimension(df)

    lines.append(f"Total rows: {len(df):,}")

    for num_col in numeric_cols[:4]:
        col_total = df[num_col].sum()
        col_avg   = df[num_col].mean()
        lines.append(f"\n{num_col} — total: {col_total:,.2f}, avg per row: {col_avg:,.2f}")

        if dim:
            try:
                grouped = df.groupby(dim)[num_col].sum().sort_values(ascending=False)
                if len(grouped) == 1:
                    label = _fmt_dimension_val(dim, grouped.index[0])
                    lines.append(f"  {dim}: {label} = {grouped.iloc[0]:,.2f}")
                else:
                    lines.append(f"  Ranked by {dim}:")
                    for name, val in grouped.items():
                        pct = (val / col_total * 100) if col_total else 0
                        label = _fmt_dimension_val(dim, name)
                        lines.append(f"    {label}: {val:,.2f} ({pct:.1f}%)")
            except Exception:
                pass

    return "\n".join(lines)


def generate_data_summary(user_request: str, row_count: int, df: pd.DataFrame) -> str:
    # ── Step 1: Compute ALL facts in Python — LLM never calculates ────────────
    computed_facts = _precompute_facts(df)

    # ── Step 2: LLM only writes natural language explanation ──────────────────
    prompt = f"""
You are a data analyst. Your ONLY job is to write a clear, well-formatted markdown explanation of pre-computed facts.

YOUR ROLE:
- Read the pre-computed facts below
- Format the answer as structured markdown for easy reading
- You are an interpreter, NOT a calculator

ABSOLUTE RULES:
- Use ONLY the numbers listed in "Pre-computed facts" — copy them exactly as shown
- Do NOT generate, calculate, or estimate any number yourself
- Do NOT add any metric, percentage, or value not present in the facts
- If a fact is missing, do not mention it

Business question: "{user_request}"

Pre-computed facts (these are the ONLY numbers you may use):
{computed_facts}

Choose the best format based on the question and data:

QUESTION TYPE → FORMAT:

1. RANKING / COMPARISON ("which is highest", "top N", "by category/store/region"):
   **[Direct answer naming top performer and its value]**
   | # | [dimension] | [metric] | Share |
   |---|---|---|---|
   | 1 | ... | ... | ...% |
   💡 **Key Insights:**
   - [pattern or gap visible in ranking]

2. SINGLE VALUE ("how many", "what is the total", "count of"):
   **[Direct answer with the value]**
   - [metric]: [value]
   - [supporting metric if available]
   💡 **Insight:** [one sentence if meaningful]

3. TREND / TIME SERIES ("over time", "monthly", "daily", "trend"):
   **[Direct answer about the trend direction]**
   - Peak: [period] = [value]
   - Lowest: [period] = [value]
   - Overall change: [start value] → [end value]
   💡 **Insight:** [trend observation]

4. LIST / DETAIL ("show me", "give me records", "filter"):
   **[Direct answer: how many records match and key total]**
   💡 **Insight:** [one notable observation from the data]

RULES (apply to all formats):
- Pick the format that fits the question — do not force a table when it adds no value
- All values must be copied exactly from the pre-computed facts — no rounding or reformatting
- Never show Share = 100% (meaningless for single-group results — skip the table)
- Never invent a value, percentage, or insight not in the facts
- Max 10 rows in any table
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        err_str = str(e)
        if "rate_limit_exceeded" in err_str or "429" in err_str or "Rate limit" in err_str:
            return f"Found {row_count:,} records matching your request."
        return computed_facts  # fallback: show raw computed facts


def render_refine_bar(current_sql: str, df_input: pd.DataFrame):
    st.divider()
    # Show conversation thread
    if st.session_state.chat_history:
        st.markdown("**💬 Conversation**")
        for i, item in enumerate(st.session_state.chat_history):
            icon = "🔵" if i == 0 else "➕"
            st.markdown(f"{icon} {item}")
        st.markdown("")

    st.markdown("**✏️ Refine your results — add a condition without restarting:**")
    with st.form("refine_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            refinement = st.text_input(
                "", placeholder="e.g. Only West region  |  Filter by Corporate segment  |  Top 5 only",
                label_visibility="collapsed",
                key=f"refine_input_{st.session_state.refine_input_counter}"
            )
        with col2:
            apply = st.form_submit_button("Apply →", use_container_width=True)

    # New Request — outside the form, full-width subtle button below
    st.markdown("""
    <style>
    div[data-testid="stButton"].new-req-btn > button {
        background: transparent !important;
        border: 1.5px dashed #d1d5db !important;
        color: #6b7280 !important;
        border-radius: 8px !important;
        width: 100%;
        font-size: 0.85rem !important;
        padding: 0.4rem 1rem !important;
        margin-top: 0.4rem;
    }
    div[data-testid="stButton"].new-req-btn > button:hover {
        border-color: #1565c0 !important;
        color: #1565c0 !important;
        background: #f0f4ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    new_req_col, _ = st.columns([2, 5])
    with new_req_col:
        new_request = st.button("🔄 Start a New Request", key="new_request_btn", use_container_width=True)

    if new_request:
        for key in ["request", "name", "df_input", "request_type", "refined_sql",
                    "chat_history", "cached_result", "result_history",
                    "refinement_type_override", "refinement_request_override"]:
            st.session_state[key] = [] if key in ("chat_history", "result_history") else None
        st.session_state.refine_input_counter = 0
        st.session_state.main_form_counter += 1    # force a fresh empty form
        st.rerun()

    if apply and refinement.strip():
        ref_lower = refinement.strip().lower()
        orig_type = st.session_state.request_type

        # Filter-condition phrases: short additions that modify the existing SQL
        # e.g. "Only West region", "Filter by Corporate", "Top 5 only", "Exclude Central"
        FILTER_STARTERS = [
            "only ", "filter by ", "filter for ", "exclude ", "just ",
            "limit to ", "restrict to ", "where ", "add filter",
            "in the ", "from the ", "for the ",
        ]
        is_filter_condition = any(ref_lower.startswith(kw) for kw in FILTER_STARTERS)

        try:
            with st.spinner("Understanding your request..."):
                if is_filter_condition:
                    # Short filter addition — modify existing SQL in place
                    ref_type = orig_type
                else:
                    # Standalone question — classify fresh
                    ref_type = classify_request(refinement.strip())

            with st.spinner("Applying..."):
                if not is_filter_condition and ref_type == "WHY_QUESTION":
                    st.session_state.refinement_type_override = "WHY_QUESTION"
                    st.session_state.refinement_request_override = refinement.strip()
                    st.session_state.refined_sql = None
                elif not is_filter_condition and ref_type == "DATA_PULL":
                    st.session_state.refinement_type_override = "DATA_PULL"
                    st.session_state.refinement_request_override = refinement.strip()
                    st.session_state.refined_sql = None
                else:
                    # Filter condition — add AND clause to existing SQL
                    new_sql = refine_query(current_sql, refinement.strip(), df_input)
                    st.session_state.refined_sql = new_sql
                    st.session_state.refinement_type_override = None
                    st.session_state.refinement_request_override = None

            st.session_state.cached_result = None
            st.session_state.chat_history.append(refinement.strip())
            st.session_state.refine_input_counter += 1
            st.rerun()

        except Exception as _refine_err:
            import re as _re
            err_str = str(_refine_err)
            if "rate_limit_exceeded" in err_str or "429" in err_str or "Rate limit" in err_str:
                wait_match = _re.search(r"try again in ([0-9hms\.]+)", err_str)
                wait_time = wait_match.group(1) if wait_match else "a few minutes"
                st.warning(
                    f"⏳ **Groq API rate limit reached.** You've used today's free-tier token quota.\n\n"
                    f"Please wait **{wait_time}** and try again.\n\n"
                    f"💡 *Tip: Upgrade to Groq Dev Tier at console.groq.com for higher limits.*"
                )
            else:
                st.error(f"Error applying refinement: {err_str}")


def show_file_summary(df: pd.DataFrame):
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())
    st.subheader("Preview")
    st.dataframe(df.head(10))
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if len(numeric_cols) > 0 and len(text_cols) > 0:
        st.subheader("Quick Charts")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            top = df[text_cols[0]].value_counts().head(10).reset_index()
            top.columns = [text_cols[0], "Count"]
            fig = px.bar(top, x=text_cols[0], y="Count", title=f"Top 10 by {text_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)
    if len(numeric_cols) > 0:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())


# ── Input Form (always visible) ───────────────────────────────────────────────
with st.form(f"request_form_{st.session_state.main_form_counter}"):
    name = ""
    request_input = st.text_area("Your Request", height=110,
        placeholder="e.g. Which region is underperforming and why?  |  Show me top 10 products by sales  |  Total units sold in CA")
    uploaded_file = st.file_uploader("Upload your own dataset (optional — defaults to Superstore)", type=["csv", "xlsx"])
    submitted = st.form_submit_button("🔍  Get My Answer", use_container_width=False)

if submitted and request_input:
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            raw_bytes = uploaded_file.read()
            df_input = None
            for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
                try:
                    import io as _io
                    df_input = pd.read_csv(_io.BytesIO(raw_bytes), encoding=enc)
                    break
                except Exception:
                    continue
            if df_input is None:
                st.error("Could not read the CSV file. Please ensure it is a valid UTF-8 or Latin-1 encoded file.")
                st.stop()
        else:
            try:
                df_input = pd.read_excel(uploaded_file)
            except Exception as _xe:
                st.error(f"Could not read Excel file: {_xe}")
                st.stop()
    else:
        df_input = df_global

    try:
        with st.spinner("Understanding your request..."):
            request_type = classify_request(request_input)
    except Exception as _triage_err:
        import re as _re2
        _err = str(_triage_err)
        if "rate_limit_exceeded" in _err or "429" in _err or "Rate limit" in _err:
            _wait = _re2.search(r"try again in ([0-9hms\.]+)", _err)
            _wait_str = _wait.group(1) if _wait else "a few minutes"
            st.warning(
                f"⏳ **Groq API rate limit reached.** Please wait **{_wait_str}** and try again.\n\n"
                f"💡 *Tip: Upgrade to Groq Dev Tier at console.groq.com for higher limits.*"
            )
            st.stop()
        else:
            st.error(f"Error classifying request: {_err}")
            st.stop()

    st.session_state.name = name
    st.session_state.request = request_input
    st.session_state.df_input = df_input
    st.session_state.request_type = request_type
    st.session_state.refined_sql = None
    st.session_state.chat_history = [request_input]
    st.session_state.cached_result = None
    st.session_state.result_history = []
    st.session_state.refinement_type_override = None
    st.session_state.refinement_request_override = None
    st.session_state.refine_input_counter = 0


# ── Results (shown below the form when available) ────────────────────────────
def _run_and_cache():
    import re as _re
    request = st.session_state.request
    df_input = st.session_state.df_input
    refined_sql = st.session_state.refined_sql

    # Check if a refinement switched the intent (e.g. WHY → DATA_PULL)
    type_override = st.session_state.get("refinement_type_override")
    request_override = st.session_state.get("refinement_request_override")
    effective_type = type_override or st.session_state.request_type
    effective_request = request_override or request

    try:
        if effective_type == "WHY_QUESTION":
            with st.spinner("Analyzing your question..."):
                result = handle_why_question(effective_request, df_input, existing_sql=refined_sql)
        elif effective_type == "DATA_PULL":
            with st.spinner("Pulling data..."):
                result = handle_data_pull(effective_request, df_input, existing_sql=refined_sql)
        else:
            result = {"status": "file_analysis"}
    except Exception as e:
        err_str = str(e)
        # Detect Groq rate limit error
        if "rate_limit_exceeded" in err_str or "429" in err_str or "Rate limit" in err_str:
            wait_match = _re.search(r"try again in ([0-9hms\.]+)", err_str)
            wait_time = wait_match.group(1) if wait_match else "a few minutes"
            result = {
                "status": "error",
                "sql": "",
                "error": (
                    f"⏳ **Groq API rate limit reached.** You've used today's free-tier token quota.\n\n"
                    f"Please wait **{wait_time}** and try again.\n\n"
                    f"💡 *Tip: Upgrade to Groq Dev Tier at console.groq.com for higher limits.*"
                )
            }
        else:
            result = {"status": "error", "sql": "", "error": err_str}

    if result.get("status") == "success" and not refined_sql:
        st.session_state.refined_sql = result.get("sql", "")

    # Clear overrides after use
    st.session_state.refinement_type_override = None
    st.session_state.refinement_request_override = None

    st.session_state.cached_result = result
    label = st.session_state.chat_history[-1] if st.session_state.chat_history else request
    st.session_state.result_history.append({
        "label": label,
        "result": result,
        "request_type": effective_type,
        "request": effective_request,
        "df_input": df_input,
    })


def render_single_result(result, request, request_type, df_input, name="", idx=0):
    """Render one result (WHY or DATA_PULL)."""
    if request_type == "WHY_QUESTION":
        if result["status"] == "success":
            render_analysis(result["explanation"], result["dataframe"],
                            request, result["sql"], name, df_raw=df_input, idx=idx)
        else:
            msg = result.get("error", "Unknown error")
            if "rate limit" in msg.lower() or "⏳" in msg or "429" in msg:
                st.warning(msg)
            else:
                st.error(f"Error: {msg}")

    elif request_type == "DATA_PULL":
        if result["status"] == "success":
            for w in result.get("warnings", []):
                st.warning(w)
            df_result = result["dataframe"]
            is_agg = result.get("is_aggregate", False)
            df_detail = result.get("detail_dataframe")
            df_display = df_detail if (is_agg and df_detail is not None) else df_result

            with st.spinner("Summarizing..."):
                summary = generate_data_summary(request, result["rows"], df_result)
            st.markdown(summary)
            st.divider()

            if is_agg:
                numeric_cols = df_result.select_dtypes(include="number").columns.tolist()
                metric_cols = st.columns(max(len(numeric_cols), 1))
                for i, col in enumerate(numeric_cols):
                    val = df_result[col].iloc[0]
                    label = col.replace("_", " ").title()
                    if isinstance(val, float) and val == int(val):
                        formatted = f"{int(val):,}"
                    elif isinstance(val, float):
                        formatted = f"{val:,.2f}"
                    else:
                        formatted = f"{val:,}"
                    metric_cols[i].metric(label, formatted)
                st.divider()

            if user_wants_visual(request):
                st.subheader("Visual")
                render_chart(df_display, request, "Results Chart")
                st.divider()

            st.subheader(f"Supporting Records ({result['rows']:,})")
            is_ranked = any(w in request.lower() for w in ["top", "bottom", "best", "worst", "highest", "lowest", "ranking"])
            df_formatted = format_display_df(df_display.copy())
            already_has_rank = any(c.lower() in ("rank", "rnk") for c in df_display.columns)
            grouped_rendered = render_grouped_table(df_display)
            if not grouped_rendered:
                if is_ranked and len(df_formatted) <= 50 and not already_has_rank and "Rank" not in df_formatted.columns:
                    df_formatted.insert(0, "Rank", range(1, len(df_formatted) + 1))
                st.dataframe(df_formatted, use_container_width=True, hide_index=True)

            buf = io.BytesIO()
            df_formatted.to_excel(buf, index=False)
            st.download_button(
                "Download Excel File", buf.getvalue(),
                file_name="results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_excel_{idx}",
            )
            st.divider()

            if not user_wants_visual(request):
                with st.expander("📊 Supporting Chart"):
                    render_chart(df_display, request, "Results Chart")
            with st.expander("🔍 SQL Query Used"):
                st.code(format_sql(result["sql"]), language="sql")
        else:
            msg = result.get("error", "Unknown error")
            if "rate limit" in msg.lower() or "⏳" in msg or "429" in msg:
                st.warning(msg)
            else:
                st.error(f"Error: {msg}")

    else:
        # Fallback for DASHBOARD / unknown — treat as DATA_PULL
        try:
            with st.spinner("Pulling data..."):
                fallback = handle_data_pull(request, df_input)
            st.session_state.cached_result = fallback
            st.rerun()
        except Exception as _e:
            st.error(f"Error: {_e}")


if st.session_state.get("request") and st.session_state.get("df_input") is not None:
    if st.session_state.get("cached_result") is None:
        _run_and_cache()

    history = st.session_state.get("result_history", [])
    df_input = st.session_state.df_input
    name = st.session_state.name

    for i, item in enumerate(history):
        st.divider()
        if i == 0:
            st.markdown(f"### 🔵 {item['label']}")
        else:
            st.markdown(
                f"""<div style="background:#e8f4fd;border-left:4px solid #1565c0;
                    padding:10px 16px;border-radius:6px;margin-bottom:8px;">
                    ➕ <strong>Refined:</strong> {item['label']}</div>""",
                unsafe_allow_html=True,
            )
        render_single_result(item["result"], item["request"],
                             item["request_type"], item["df_input"], name, idx=i)

    # Refine bar — always at the bottom using the latest SQL
    latest_sql = st.session_state.refined_sql or ""
    if history and history[-1]["result"].get("status") == "success":
        latest_sql = history[-1]["result"].get("sql", latest_sql)
    render_refine_bar(latest_sql, df_input)

    st.divider()
    st.markdown('<p class="footer-text">🤖 Ad-Hoc Data Request Bot &nbsp;·&nbsp; Built with Streamlit · Groq · LLaMA 3.3 · DuckDB</p>', unsafe_allow_html=True)
