# app.py
# GenAI-Powered EDA Tool — Automated CSV Analysis with Business Insights
# Author: Robert Borkar
# Date: March 2026

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import anthropic
import json
import io
import re
from datetime import datetime

st.set_page_config(
    page_title="EDA Insight Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode state init
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ═══════════════════════════════════════════
   LIGHT MODE — Virtual Sports ref (Image 1)
   BG: #F2F2F7  |  Card: #FFFFFF  |  Sidebar: #FFFFFF
   Text: #111111 primary, #6B7280 secondary
   Accent: #6366F1 indigo
   ═══════════════════════════════════════════ */

/* Base */
.stApp {
    background-color: #F2F2F7 !important;
}
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
section[data-testid="stSidebar"] * {
    color: #374151 !important;
}

/* ── Brand ── */
.brand-wrap {
    padding: 8px 0 24px 0;
}
.brand-name {
    font-family: 'Inter', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    color: #111111 !important;
    letter-spacing: -0.8px;
    line-height: 1.1;
}
.brand-accent {
    color: #6366F1 !important;
}
.brand-tagline {
    font-size: 0.8rem;
    font-weight: 400;
    color: #9CA3AF !important;
    margin-top: 6px;
    letter-spacing: 0.01em;
}

/* ── Hallucination badge ── */
.guard-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    border-radius: 8px;
    padding: 6px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    color: #6366F1;
    letter-spacing: 0.04em;
    margin-bottom: 20px;
}
.guard-dot {
    width: 7px; height: 7px;
    background: #6366F1;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Sidebar divider ── */
section[data-testid="stSidebar"] hr {
    border-color: #F3F4F6 !important;
}

/* ── Section headers ── */
.sec-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin: 32px 0 18px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #E5E7EB;
}

/* ── Metric cards ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin-bottom: 8px;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 22px 20px;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s, transform 0.15s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366F1, #A5B4FC);
    opacity: 0;
    transition: opacity 0.2s;
}
.metric-card:hover::before { opacity: 1; }
.metric-card:hover {
    box-shadow: 0 4px 16px rgba(99,102,241,0.1);
    transform: translateY(-2px);
}
.metric-value {
    font-family: 'Inter', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #111111;
    line-height: 1;
    letter-spacing: -1.5px;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 8px;
}
.metric-card.highlight .metric-value { color: #6366F1; }
.metric-card.warn .metric-value { color: #F59E0B; }

/* ── Insight cards ── */
.insight-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 14px;
    border-left: 4px solid #6366F1;
    transition: transform 0.15s, box-shadow 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.insight-card:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
}
.insight-card.anomaly  { border-left-color: #EF4444; }
.insight-card.quality  { border-left-color: #F59E0B; }
.insight-card.segment  { border-left-color: #8B5CF6; }
.insight-card.structure { border-left-color: #3B82F6; }
.insight-card.trend    { border-left-color: #6366F1; }

.insight-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    font-weight: 500;
    color: #D1D5DB;
    margin-bottom: 8px;
    letter-spacing: 0.08em;
}
.insight-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #111111;
    margin-bottom: 10px;
    line-height: 1.35;
}
.insight-body {
    font-size: 0.9rem;
    font-weight: 400;
    color: #6B7280;
    line-height: 1.75;
    margin-bottom: 16px;
}
.insight-action {
    font-size: 0.85rem;
    font-weight: 600;
    color: #6366F1;
    padding-top: 14px;
    border-top: 1px solid #F3F4F6;
}
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
    margin-right: 6px;
}
.badge-trend     { background: #EEF2FF; color: #6366F1; }
.badge-quality   { background: #FFFBEB; color: #D97706; }
.badge-anomaly   { background: #FEF2F2; color: #DC2626; }
.badge-segment   { background: #F5F3FF; color: #7C3AED; }
.badge-structure { background: #EFF6FF; color: #2563EB; }

/* ── Confidence bar ── */
.conf-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 12px 0;
}
.conf-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #9CA3AF;
    min-width: 76px;
}
.conf-track {
    flex: 1;
    background: #F3F4F6;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.conf-fill {
    height: 6px;
    border-radius: 4px;
    transition: width 0.6s ease;
}

/* ── Quality verdict card ── */
.verdict-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.verdict-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 10px;
}
.verdict-grade {
    font-family: 'Inter', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 8px;
    letter-spacing: -1px;
}
.verdict-text {
    font-size: 0.92rem;
    font-weight: 400;
    color: #6B7280;
    line-height: 1.7;
}
.flag-item {
    font-size: 0.84rem;
    font-weight: 500;
    color: #D97706;
    margin-top: 8px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}

/* ── Inputs ── */
div[data-testid="stTextInput"] label,
div[data-testid="stFileUploader"] label {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    letter-spacing: 0.01em !important;
}
div[data-testid="stTextInput"] input {
    background: #F9FAFB !important;
    border: 1.5px solid #E5E7EB !important;
    color: #111111 !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.84rem !important;
    padding: 11px 16px !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    background: #FFFFFF !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #D1D5DB !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] > div {
    background: #FFFFFF !important;
    border: 2px dashed #E5E7EB !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
div[data-testid="stFileUploader"] > div:hover {
    border-color: #A5B4FC !important;
    background: #FAFAFE !important;
}
div[data-testid="stFileUploader"] p {
    color: #9CA3AF !important;
    font-size: 0.92rem !important;
    font-weight: 400 !important;
}
div[data-testid="stFileUploader"] small {
    color: #D1D5DB !important;
    font-size: 0.78rem !important;
}
/* Uploaded filename — make it visible */
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploaderFileName"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
div[data-testid="stFileUploader"] .uploadedFileName,
div[data-testid="stFileUploader"] li span,
div[data-testid="stFileUploader"] li {
    color: #111111 !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}

/* ── Button ── */
.stButton > button {
    background: #6366F1 !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 11px 28px !important;
    letter-spacing: 0.01em !important;
    transition: background 0.2s, transform 0.1s, box-shadow 0.2s !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
}
.stButton > button:hover {
    background: #4F46E5 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: transparent !important;
    color: #6366F1 !important;
    border: 1.5px solid #C7D2FE !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
}
.stDownloadButton > button:hover {
    background: #EEF2FF !important;
    border-color: #A5B4FC !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; }
div[data-testid="stSuccess"] {
    background: #F0FDF4 !important;
    border: 1px solid #BBF7D0 !important;
    color: #15803D !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
div[data-testid="stInfo"] {
    background: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    color: #1D4ED8 !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
div[data-testid="stWarning"] {
    background: #FFFBEB !important;
    border: 1px solid #FDE68A !important;
    color: #92400E !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
}
div[data-testid="stError"] {
    background: #FEF2F2 !important;
    border: 1px solid #FECACA !important;
    color: #991B1B !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

/* ── Sidebar how-it-works ── */
.how-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 12px;
}
.step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    color: #6366F1;
    min-width: 20px;
    padding-top: 2px;
}
.step-text {
    font-size: 0.86rem;
    font-weight: 400;
    color: #6B7280;
    line-height: 1.5;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 6rem 2rem;
}
.empty-icon {
    font-size: 3rem;
    opacity: 0.1;
    margin-bottom: 18px;
}
.empty-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    color: #D1D5DB;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Spinner ── */
div[data-testid="stSpinner"] p {
    color: #6B7280 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
}

/* ── Streamlit default text overrides ── */
.stMarkdown p {
    color: #374151 !important;
    font-size: 0.92rem !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #6366F1 !important;
    font-weight: 600 !important;
}

/* ── Plotly chart container ── */
div[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border-radius: 16px;
    border: 1px solid #E5E7EB;
    padding: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* ── Kill Streamlit internal UI leaks ── */
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
[title="keyboard_double_arrow_left"],
[title="keyboard_double_arrow_right"] { display: none !important; }
button[kind="header"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }

/* Theme toggle button styling */
div[data-testid="stButton"]:has(button[key="theme_btn"]) button,
button[data-testid="theme_btn"] {
    background: #F9FAFB !important;
    color: #374151 !important;
    border: 1px solid #E5E7EB !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    text-align: left !important;
}
button[data-testid="theme_btn"]:hover {
    background: #F3F4F6 !important;
    transform: none !important;
}

::-webkit-scrollbar-track { background: #F2F2F7; }
::-webkit-scrollbar-thumb { background: #E5E7EB; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #D1D5DB; }
</style>
""", unsafe_allow_html=True)

# Dark mode CSS — injected dynamically
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stApp { background-color: #0F0F14 !important; }
    section[data-testid="stSidebar"] {
        background-color: #16161F !important;
        border-right: 1px solid #2A2A3A !important;
    }
    section[data-testid="stSidebar"] * { color: #C9C9D3 !important; }
    .brand-name { color: #F0F0FF !important; }
    .sec-header { color: #555570 !important; border-bottom-color: #2A2A3A !important; }
    .metric-card {
        background: #16161F !important;
        border-color: #2A2A3A !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
    }
    .metric-value { color: #F0F0FF !important; }
    .insight-card {
        background: #16161F !important;
        border-color: #2A2A3A !important;
    }
    .insight-title { color: #F0F0FF !important; }
    .insight-body { color: #9999B0 !important; }
    .insight-action { color: #818CF8 !important; border-top-color: #2A2A3A !important; }
    .verdict-card { background: #16161F !important; border-color: #2A2A3A !important; }
    .verdict-text { color: #9999B0 !important; }
    .conf-track { background: #2A2A3A !important; }
    div[data-testid="stFileUploader"] > div {
        background: #16161F !important;
        border-color: #2A2A3A !important;
    }
    div[data-testid="stFileUploader"] > div:hover {
        border-color: #6366F1 !important;
        background: #1A1A28 !important;
    }
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] li,
    div[data-testid="stFileUploader"] li span,
    div[data-testid="stFileUploaderFileName"] { color: #F0F0FF !important; opacity: 1 !important; }
    div[data-testid="stPlotlyChart"] {
        background: #16161F !important;
        border-color: #2A2A3A !important;
    }
    div[data-testid="stDataFrame"] {
        border-color: #2A2A3A !important;
    }
    .stMarkdown p { color: #C9C9D3 !important; }
    ::-webkit-scrollbar-track { background: #0F0F14 !important; }
    ::-webkit-scrollbar-thumb { background: #2A2A3A !important; }
    /* Toggle button dark override */
    .theme-toggle-wrap { background: #1E1E2E !important; border-color: #3A3A5A !important; }
    .theme-toggle-label { color: #C9C9D3 !important; }
    </style>
    """, unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def sanitise_col(name: str) -> str:
    return re.sub(r'[^\w\s\-]', '', str(name))[:60]


def validate_upload(f):
    if f is None:
        return False, "No file uploaded."
    if not f.name.lower().endswith('.csv'):
        return False, "File must have a .csv extension."
    allowed = {'text/csv', 'application/csv', 'text/plain', 'application/octet-stream'}
    if f.type not in allowed:
        return False, "Invalid file type. Please upload a CSV."
    if f.size > 10 * 1024 * 1024:
        return False, f"File too large ({f.size/1024/1024:.1f}MB). Max is 10MB."
    return True, ""


def load_csv(f):
    try:
        raw = f.read()
        f.seek(0)
        df = None
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            return None, "Could not decode file. Try saving as UTF-8 CSV."
        if df.empty:
            return None, "The CSV file is empty."
        if len(df.columns) == 0:
            return None, "No columns found in CSV."
        return df, ""
    except pd.errors.EmptyDataError:
        return None, "The CSV file appears empty or has no parseable data."
    except pd.errors.ParserError:
        return None, "Could not parse CSV. File may be malformed."
    except Exception:
        return None, "An error occurred reading the file. Please check the format."


def profile_dataframe(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    profile = {
        'shape': {'rows': int(df.shape[0]), 'cols': int(df.shape[1])},
        'duplicates': int(df.duplicated().sum()),
        'total_missing_pct': round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
        'numeric_cols': numeric_cols,
        'categorical_cols': cat_cols,
        'columns': {}
    }

    for col in df.columns:
        safe = sanitise_col(col)
        null_pct = round(float(df[col].isnull().mean() * 100), 2)
        cp = {
            'dtype': str(df[col].dtype),
            'null_pct': null_pct,
            'unique_count': int(df[col].nunique()),
            'unreliable': bool(null_pct > 30)
        }
        if col in numeric_cols:
            desc = df[col].describe()
            cp.update({
                'mean':   round(float(desc.get('mean', 0)), 4),
                'median': round(float(df[col].median()), 4),
                'std':    round(float(desc.get('std', 0)), 4),
                'min':    round(float(desc.get('min', 0)), 4),
                'max':    round(float(desc.get('max', 0)), 4),
                'skewness': round(float(df[col].skew()), 4) if len(df[col].dropna()) > 2 else None
            })
        else:
            top_vals = df[col].value_counts().head(5).to_dict()
            cp['top_values'] = {str(k): int(v) for k, v in top_vals.items()}
        profile['columns'][safe] = cp

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        pairs, seen = [], set()
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i+1:]:
                key = tuple(sorted([c1, c2]))
                if key not in seen:
                    val = corr.loc[c1, c2]
                    if not np.isnan(val):
                        pairs.append({
                            'col1': sanitise_col(c1), 'col2': sanitise_col(c2),
                            'correlation': round(float(val), 4),
                            'strength': 'strong' if abs(val) >= 0.7 else ('moderate' if abs(val) >= 0.4 else 'weak')
                        })
                    seen.add(key)
        pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        profile['top_correlations'] = pairs[:10]
    else:
        profile['top_correlations'] = []

    return profile


def build_prompt(profile: dict, filename: str) -> str:
    return f"""You are a senior data analyst reviewing a dataset for business stakeholders.
Analyze the following dataset profile and return structured business insights.

Dataset: {filename}
Profile:
{json.dumps(profile, indent=2)}

STRICT RULES:
1. NEVER claim causation. Use "associated with", "correlates with", "suggests" only.
2. Correlations with absolute value below 0.4 MUST be flagged as "weak signal — insufficient for business decisions".
3. Columns with null_pct > 30 are marked unreliable=true. Do NOT base primary insights on these without stating they are unreliable.
4. Do NOT invent numbers or patterns not in the profile.
5. Confidence scores must reflect actual data quality.

Return ONLY valid JSON — no markdown, no preamble:

{{
  "dataset_summary": "2-3 sentence plain English summary",
  "data_quality_verdict": {{
    "overall_grade": "A/B/C/D",
    "summary": "1-2 sentences on data quality",
    "flags": ["specific issues found"]
  }},
  "insights": [
    {{
      "title": "Short insight title",
      "type": "trend|anomaly|quality|segment|structure",
      "body": "2-3 sentences in plain business language",
      "action": "Specific recommended action",
      "confidence_score": 0.0,
      "confidence_reason": "Why this confidence level"
    }}
  ]
}}

Generate exactly 5 insights ordered by business impact, highest first."""


def call_claude(api_key: str, prompt: str):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        return json.loads(raw), ""
    except anthropic.AuthenticationError:
        return None, "Invalid API key. Check your Anthropic API key and try again."
    except anthropic.RateLimitError:
        return None, "Rate limit reached. Wait a moment and try again."
    except anthropic.APITimeoutError:
        return None, "Request timed out. Try a smaller file."
    except anthropic.APIConnectionError:
        return None, "Could not connect to Anthropic API. Check your internet connection."
    except json.JSONDecodeError:
        return None, "Malformed response from API. Please try again."
    except Exception:
        return None, "Unexpected error generating insights. Please try again."


# ── Chart config — light, clean, Inter font ───────────────────────────────────

BG_CHART = '#FFFFFF'
PAPER_C  = '#FFFFFF'
GRID_C   = '#F3F4F6'
TEXT_C   = '#6B7280'
INDIGO   = '#6366F1'
VIOLET   = '#8B5CF6'
BLUE_C   = '#3B82F6'
AMBER_C  = '#F59E0B'
RED_C    = '#EF4444'
SLATE_C  = '#94A3B8'

# Cohesive palette — no jarring neons
PALETTE  = [INDIGO, VIOLET, BLUE_C, SLATE_C, '#A5B4FC', '#C4B5FD']

def base_layout(title="", height=None):
    l = dict(
        title=dict(
            text=title,
            font=dict(family='Inter', size=13, color='#374151'),
            x=0, xanchor='left'
        ),
        paper_bgcolor=PAPER_C,
        plot_bgcolor=BG_CHART,
        font=dict(family='Inter', color=TEXT_C, size=12),
        margin=dict(l=44, r=20, t=44, b=44),
        xaxis=dict(
            gridcolor=GRID_C, zerolinecolor=GRID_C,
            linecolor='#E5E7EB', tickfont=dict(size=11, color='#9CA3AF'),
            tickcolor='#E5E7EB'
        ),
        yaxis=dict(
            gridcolor=GRID_C, zerolinecolor=GRID_C,
            linecolor='#E5E7EB', tickfont=dict(size=11, color='#9CA3AF'),
            tickcolor='#E5E7EB'
        ),
        showlegend=False
    )
    if height:
        l['height'] = height
    return l


def render_charts(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # distributions
    if numeric_cols:
        st.markdown('<div class="sec-header">// distributions</div>', unsafe_allow_html=True)
        cols_per_row = 3
        batches = [numeric_cols[i:i+cols_per_row] for i in range(0, min(len(numeric_cols), 9), cols_per_row)]
        for batch in batches:
            chart_cols = st.columns(len(batch))
            for idx, col in enumerate(batch):
                with chart_cols[idx]:
                    fig = px.histogram(df, x=col, nbins=30,
                                       color_discrete_sequence=[PALETTE[idx % len(PALETTE)]])
                    fig.update_layout(**base_layout(col, height=220))
                    fig.update_traces(marker_line_width=0, opacity=0.8)
                    st.plotly_chart(fig, use_container_width=True)

    # missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if len(missing) > 0:
        st.markdown('<div class="sec-header">// missing values</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=missing.values, y=missing.index.tolist(),
            orientation='h',
            marker=dict(color=AMBER_C, line=dict(width=0), opacity=0.85),
            text=missing.values, textposition='outside',
            textfont=dict(size=11, color='#374151')
        ))
        layout = base_layout("Missing Value Count by Column", height=max(200, len(missing) * 38))
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    # correlation heatmap
    if len(numeric_cols) >= 2:
        st.markdown('<div class="sec-header">// correlation matrix</div>', unsafe_allow_html=True)
        corr = df[numeric_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale=[[0, RED_C], [0.5, '#F3F4F6'], [1, INDIGO]],
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=10, color='#374151'),
            showscale=True,
            colorbar=dict(
                tickfont=dict(color=TEXT_C, size=10),
                outlinewidth=0,
                bgcolor=PAPER_C
            )
        ))
        layout = base_layout("Feature Correlation Heatmap", height=max(300, len(numeric_cols) * 54))
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        # top scatter pairs
        pairs, seen = [], set()
        for c1 in numeric_cols:
            for c2 in numeric_cols:
                if c1 != c2:
                    key = tuple(sorted([c1, c2]))
                    if key not in seen:
                        v = abs(corr.loc[c1, c2])
                        if not np.isnan(v):
                            pairs.append((c1, c2, v))
                        seen.add(key)
        pairs.sort(key=lambda x: x[2], reverse=True)
        top = pairs[:3]

        if top:
            st.markdown('<div class="sec-header">// top correlated pairs</div>', unsafe_allow_html=True)
            scatter_cols = st.columns(len(top))
            for idx, (c1, c2, val) in enumerate(top):
                with scatter_cols[idx]:
                    sample = df[[c1, c2]].dropna().sample(min(500, len(df)))
                    fig = px.scatter(sample, x=c1, y=c2,
                                     color_discrete_sequence=[PALETTE[idx % len(PALETTE)]],
                                     opacity=0.55)
                    fig.update_layout(**base_layout(f"{c1[:14]} × {c2[:14]}  r={val:.2f}", height=240))
                    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
                    st.plotly_chart(fig, use_container_width=True)


def render_insight_card(insight: dict, idx: int):
    itype = insight.get('type', 'structure').lower()
    conf = float(insight.get('confidence_score', 0))
    conf_pct = int(conf * 100)
    conf_color = INDIGO if conf_pct >= 70 else (AMBER_C if conf_pct >= 40 else RED_C)

    st.markdown(f"""
    <div class="insight-card {itype}">
        <div class="insight-num">INSIGHT {idx:02d}</div>
        <span class="badge badge-{itype}">{itype}</span>
        <div class="insight-title">{insight.get('title','')}</div>
        <div class="insight-body">{insight.get('body','')}</div>
        <div class="conf-wrap">
            <span class="conf-label">Confidence</span>
            <div class="conf-track">
                <div class="conf-fill" style="width:{conf_pct}%; background:{conf_color};"></div>
            </div>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.74rem; font-weight:600; color:{conf_color}; min-width:36px;">{conf_pct}%</span>
        </div>
        <div style="font-size:0.78rem; color:#D1D5DB; margin-top:4px; font-style:italic;">{insight.get('confidence_reason','')}</div>
        <div class="insight-action">→ {insight.get('action','')}</div>
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
        <div class="brand-name">EDA<br><span class="brand-accent">Insight</span><br>Engine</div>
        <div class="brand-tagline">CSV → Profile → Business Insights</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guard-badge">
        <span class="guard-dot"></span>
        HALLUCINATION GUARD ACTIVE
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # load from secrets.toml if available — no typing needed locally
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-api03-...",
        help="Never stored or logged."
    )
    if api_key:
        if not api_key.startswith("sk-ant-"):
            st.warning("Key format looks off — double check it")

    st.markdown("---")

    st.markdown("""
    <div style="margin-bottom: 10px;">
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; font-weight:600; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.15em;">How it works</span>
    </div>
    <div class="how-step"><span class="step-num">01</span><span class="step-text">Upload any CSV (max 10MB)</span></div>
    <div class="how-step"><span class="step-num">02</span><span class="step-text">Review auto-generated data profile</span></div>
    <div class="how-step"><span class="step-num">03</span><span class="step-text">Generate AI business insights</span></div>
    <div class="how-step"><span class="step-num">04</span><span class="step-text">Download full JSON report</span></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; font-weight:400; color:#D1D5DB; line-height:1.8;">
        Built by Robert Borkar · March 2026<br>
        Powered by Claude Haiku 4.5
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    is_dark = st.session_state.dark_mode
    track_bg = "#6366F1" if is_dark else "#E5E7EB"
    knob_pos = "translateX(20px)" if is_dark else "translateX(0px)"
    mode_text = "Dark Mode" if is_dark else "Light Mode"

    st.markdown(f"""
    <style>
    .theme-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: {'#1E1E2E' if is_dark else '#F9FAFB'};
        border: 1px solid {'#3A3A5A' if is_dark else '#E5E7EB'};
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 2px;
        cursor: pointer;
    }}
    .theme-row-label {{
        font-size: 0.84rem;
        font-weight: 600;
        color: {'#C9C9D3' if is_dark else '#374151'};
        font-family: 'Inter', sans-serif;
    }}
    .theme-knob-track {{
        width: 44px; height: 24px;
        background: {track_bg};
        border-radius: 24px;
        position: relative;
        transition: background 0.3s;
        flex-shrink: 0;
    }}
    .theme-knob {{
        position: absolute;
        width: 18px; height: 18px;
        background: white;
        border-radius: 50%;
        top: 3px; left: 3px;
        transform: {knob_pos};
        transition: transform 0.3s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.25);
    }}
    /* Make st.button for theme invisible but clickable on top */
    div[data-testid="stButton"] button[key="theme_btn"],
    .stButton:has(button[key="theme_btn"]) > button {{
        position: absolute !important;
        opacity: 0 !important;
        width: 100% !important;
        height: 46px !important;
        margin-top: -50px !important;
        cursor: pointer !important;
        z-index: 10 !important;
    }}
    </style>
    <div class="theme-row">
        <span class="theme-row-label">{mode_text}</span>
        <div class="theme-knob-track"><div class="theme-knob"></div></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("toggle", key="theme_btn", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">// upload dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a CSV file here",
    type=['csv'],
    help="Max 10MB. UTF-8 or Latin-1 encoding supported."
)

for k in ['last_filename', 'df', 'profile', 'insights']:
    if k not in st.session_state:
        st.session_state[k] = None

if uploaded_file is not None:
    valid, err = validate_upload(uploaded_file)
    if not valid:
        st.error(err)
    else:
        if uploaded_file.name != st.session_state.last_filename:
            df, load_err = load_csv(uploaded_file)
            if load_err:
                st.error(load_err)
            else:
                st.session_state.df = df
                st.session_state.last_filename = uploaded_file.name
                st.session_state.profile = profile_dataframe(df)
                st.session_state.insights = None

        df = st.session_state.df
        profile = st.session_state.profile

        if df is not None and profile is not None:

            # metric cards
            st.markdown('<div class="sec-header">// dataset overview</div>', unsafe_allow_html=True)

            missing_pct = profile['total_missing_pct']
            warn_class = 'warn' if missing_pct > 10 else 'highlight'

            st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-card highlight">
                    <div class="metric-value">{profile['shape']['rows']:,}</div>
                    <div class="metric-label">Rows</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{profile['shape']['cols']}</div>
                    <div class="metric-label">Columns</div>
                </div>
                <div class="metric-card {warn_class}">
                    <div class="metric-value">{missing_pct}%</div>
                    <div class="metric-label">Missing</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{profile['duplicates']}</div>
                    <div class="metric-label">Duplicates</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(profile['numeric_cols'])}</div>
                    <div class="metric-label">Numeric Cols</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # column profile table
            st.markdown('<div class="sec-header">// column profile</div>', unsafe_allow_html=True)
            rows = []
            for col_name, cp in profile['columns'].items():
                row = {
                    'Column': col_name,
                    'Type': cp['dtype'],
                    'Null %': f"{cp['null_pct']}%",
                    'Unique': cp['unique_count'],
                    'Reliable': '✓' if not cp.get('unreliable') else '⚠  >30% null'
                }
                if 'mean' in cp:
                    row.update({'Mean': cp['mean'], 'Std': cp['std'], 'Skew': cp.get('skewness', '—')})
                else:
                    tops = list(cp.get('top_values', {}).keys())
                    row['Top Value'] = tops[0] if tops else '—'
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # charts
            render_charts(df)

            # insights
            st.markdown('<div class="sec-header">// ai insights</div>', unsafe_allow_html=True)

            if not api_key:
                st.info("Enter your Anthropic API key in the sidebar to generate insights.")
            else:
                if st.button("⚡  Generate Business Insights"):
                    with st.spinner("Analysing with Claude Haiku 4.5 — usually takes 5–10 seconds..."):
                        result, api_err = call_claude(api_key, build_prompt(profile, uploaded_file.name))
                        if api_err:
                            st.error(api_err)
                        else:
                            st.session_state.insights = result

                if st.session_state.insights is not None:
                    ins = st.session_state.insights

                    if 'dataset_summary' in ins:
                        st.markdown(f"""
                        <div class="verdict-card">
                            <div class="verdict-label">Dataset Summary</div>
                            <div class="verdict-text">{ins['dataset_summary']}</div>
                        </div>""", unsafe_allow_html=True)

                    if 'data_quality_verdict' in ins:
                        dqv = ins['data_quality_verdict']
                        grade = dqv.get('overall_grade', 'B')
                        gc = {'A': '#10B981', 'B': INDIGO, 'C': AMBER_C, 'D': RED_C}.get(grade, TEXT_C)
                        flags_html = ''.join([
                            f'<div class="flag-item"><span>⚠</span><span>{f}</span></div>'
                            for f in dqv.get('flags', [])
                        ])
                        st.markdown(f"""
                        <div class="verdict-card">
                            <div style="display:flex; align-items:center; gap:16px; margin-bottom:10px;">
                                <div class="verdict-label" style="margin-bottom:0;">Data Quality Grade</div>
                                <div class="verdict-grade" style="color:{gc};">{grade}</div>
                            </div>
                            <div class="verdict-text">{dqv.get('summary','')}</div>
                            {flags_html}
                        </div>""", unsafe_allow_html=True)

                    if 'insights' in ins and isinstance(ins['insights'], list):
                        for i, insight in enumerate(ins['insights'][:5], 1):
                            render_insight_card(insight, i)

                    report = {
                        'generated_at': datetime.now().isoformat(),
                        'filename': uploaded_file.name,
                        'dataset_profile': profile,
                        'ai_insights': ins
                    }
                    st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
                    st.download_button(
                        label="↓ Download Full Report (JSON)",
                        data=json.dumps(report, indent=2),
                        file_name=uploaded_file.name.replace('.csv', '') + '_insights.json',
                        mime="application/json"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">◈</div>
        <div class="empty-text">Upload a CSV to begin analysis</div>
    </div>""", unsafe_allow_html=True)
