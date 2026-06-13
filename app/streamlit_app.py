from __future__ import annotations

import json
import sys
from pathlib import Path
from importlib import resources

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


from risklens_ai.artifacts import load_artifacts
from risklens_ai.explainability import explain_with_shap, shap_explanation
from risklens_ai.memo import generate_investment_memo
from risklens_ai.pdf_report import write_text_pdf
from risklens_ai.questions import generate_due_diligence_questions
from risklens_ai.scoring import score_company
from risklens_ai.excel_parser import parse_financial_excel


ROOT = Path(__file__).resolve().parents[1]

TIER_COLOR = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_sample() -> tuple[str, dict[str, float]]:
    sample_text = (
        resources.files("risklens_ai.samples")
        .joinpath("sample_company.json")
        .read_text(encoding="utf-8")
    )
    sample = json.loads(sample_text)
    return sample["company_name"], sample["financials"]


def gauge(score: float) -> go.Figure:
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f2937"},
                "steps": [
                    {"range": [0, 35],  "color": "#d1fae5"},
                    {"range": [35, 65], "color": "#fde68a"},
                    {"range": [65, 100],"color": "#fecaca"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "value": score,
                },
            },
        )
    )


def _wrap(text: str, width: int = 28) -> str:
    """Wrap a long label at word boundaries, joining with <br> for Plotly."""
    words, lines, current = text.split(), [], []
    for word in words:
        if sum(len(w) + 1 for w in current) + len(word) > width and current:
            lines.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


# ── Shared dark-theme constants ───────────────────────────────────────────────
_BG      = "#0e1117"   # Streamlit dark page background
_PAPER   = "#161b27"   # slightly lighter frame
_GRID    = "#2d3348"   # grid lines
_BORDER  = "#3d4461"   # axis lines
_TEXT    = "#e2e8f0"   # primary labels
_SUBTEXT = "#94a3b8"   # secondary / axis titles
_RED     = "#f87171"   # risk ↑  (softer red, readable on dark)
_GREEN   = "#34d399"   # risk ↓  (emerald, readable on dark)
_BLUE    = "#60a5fa"   # reference lines


def _dark_layout(**overrides) -> dict:
    """Return a base dark-theme layout dict, merged with any overrides."""
    base = dict(
        plot_bgcolor  = _BG,
        paper_bgcolor = _PAPER,
        font          = {"family": "Inter, sans-serif", "size": 12, "color": _TEXT},
        xaxis=dict(
            gridcolor      = _GRID,
            zerolinecolor  = _BORDER,
            tickfont       = {"color": _SUBTEXT, "size": 11},
            title_font     = {"color": _SUBTEXT, "size": 12},
            showline       = True,
            linecolor      = _BORDER,
        ),
        yaxis=dict(
            gridcolor   = _GRID,
            tickfont    = {"color": _TEXT, "size": 12},
            automargin  = True,
            showgrid    = False,
            showline    = False,
        ),
        legend        = {"visible": False},
        hoverlabel    = dict(
            bgcolor    = "#1e2535",
            bordercolor= _BORDER,
            font_color = _TEXT,
            font_size  = 12,
        ),
        margin        = {"l": 12, "r": 100, "t": 60, "b": 50},
    )
    base.update(overrides)
    return base


def _waterfall(shap_result) -> go.Figure:
    """
    SHAP waterfall — stacked bar cascade on a dark theme.
    Top 10 features by |SHAP|, remainder collapsed.
    Values rounded to 3 decimal places.
    """
    from risklens_ai.explainability import _label as _feat_label

    sv   = shap_result.shap_values[0]
    fn   = shap_result.feature_names
    base = shap_result.base_value
    pred = shap_result.predicted_probability

    label_map = dict(zip(
        shap_result.contribution_df["Feature"],
        shap_result.contribution_df["Label"],
    ))

    def _get_label(feat: str) -> str:
        raw = label_map.get(feat) or _feat_label(feat)
        return _wrap(raw, 30)

    # ── Top-10 + "Other" ──────────────────────────────────────────────────────
    TOP_N    = 10
    order    = np.argsort(np.abs(sv))[::-1]
    top_idx  = list(order[:TOP_N])
    rest     = order[TOP_N:]

    labels = [_get_label(fn[i]) for i in top_idx]
    values = [float(sv[i]) for i in top_idx]

    other_val = float(sv[rest].sum()) if len(rest) > 0 else 0.0
    if abs(other_val) > 1e-6:
        labels.append("All other<br>features")
        values.append(other_val)

    # Reverse: largest |SHAP| at top
    labels = labels[::-1]
    values = values[::-1]

    # ── Cascade geometry ──────────────────────────────────────────────────────
    running = base
    lefts, widths, colors, hover_texts = [], [], [], []
    for v in values:
        w = abs(v)
        left = running if v >= 0 else running + v
        lefts.append(left)
        widths.append(w)
        colors.append(_RED if v >= 0 else _GREEN)
        sign = "+" if v >= 0 else "−"
        hover_texts.append(
            f"<b>{sign}{w:.3f}</b><br>"
            f"{'Increases' if v >= 0 else 'Reduces'} bankruptcy probability"
        )
        running += v

    bar_labels = [f"{'+' if v >= 0 else ''}{v:.3f}" for v in values]

    fig = go.Figure()

    # Invisible spacer (positions each bar correctly)
    fig.add_trace(go.Bar(
        y=labels, x=lefts, orientation="h",
        marker={"color": "rgba(0,0,0,0)"},
        hoverinfo="skip", showlegend=False,
    ))

    # Visible contribution bars
    fig.add_trace(go.Bar(
        y=labels, x=widths, orientation="h",
        marker={"color": colors, "opacity": 0.90,
                "line": {"color": "rgba(0,0,0,0)", "width": 0}},
        text=bar_labels,
        textposition="outside",
        textfont={"size": 12, "color": _TEXT},
        customdata=hover_texts,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ))

    # Base probability reference line
    fig.add_vline(
        x=base, line_dash="dot", line_color=_SUBTEXT, line_width=1.5,
        annotation_text=f"Base {base:.3f}",
        annotation_font={"size": 11, "color": _SUBTEXT},
        annotation_position="top left",
    )
    # Final prediction line
    fig.add_vline(
        x=pred, line_dash="dash", line_color=_BLUE, line_width=2,
        annotation_text=f"Prediction {pred:.3f}",
        annotation_font={"size": 11, "color": _BLUE},
        annotation_position="bottom right",
    )

    # X-axis range with padding
    all_x = lefts + [l + w for l, w in zip(lefts, widths)]
    x_min = min(min(all_x), base) - 0.008
    x_max = max(max(all_x), pred) + 0.025

    fig.update_layout(**_dark_layout(
        barmode = "stack",
        title   = {
            "text": "SHAP Waterfall  ·  Feature contributions to bankruptcy probability",
            "font": {"size": 15, "color": _TEXT},
            "x": 0.01,
        },
        xaxis = dict(
            title      = "Bankruptcy probability",
            tickformat = ".3f",
            gridcolor  = _GRID,
            zerolinecolor = _BORDER,
            tickfont   = {"color": _SUBTEXT, "size": 11},
            title_font = {"color": _SUBTEXT, "size": 12},
            range      = [x_min, x_max],
            showline   = True,
            linecolor  = _BORDER,
        ),
        yaxis = dict(
            tickfont   = {"color": _TEXT, "size": 12},
            automargin = True,
            showgrid   = False,
        ),
        height = max(700, 58 * len(labels) + 120),
        margin = {"l": 12, "r": 110, "t": 65, "b": 55},
    ))
    return fig


def _bar_chart(shap_result, top_n: int = 15) -> go.Figure:
    """
    Horizontal importance bar chart — top features by |SHAP|.
    Sorted descending. Red = increases risk, green = reduces risk.
    """
    from risklens_ai.explainability import _label as _feat_label

    sv    = shap_result.shap_values[0]
    fn    = shap_result.feature_names
    order = np.argsort(np.abs(sv))[::-1][:top_n]

    label_map = dict(zip(
        shap_result.contribution_df["Feature"],
        shap_result.contribution_df["Label"],
    ))

    # Largest at top → reversed for horizontal chart
    raw_labels = [(label_map.get(fn[i]) or _feat_label(fn[i])) for i in order]
    vals       = [float(sv[i]) for i in order]

    labels_r = [_wrap(l, 32) for l in reversed(raw_labels)]
    vals_r   = list(reversed(vals))
    colors_r = [_RED if v >= 0 else _GREEN for v in vals_r]
    text_r   = [f"{'+' if v >= 0 else ''}{v:.3f}" for v in vals_r]

    x_max = max(abs(v) for v in vals) if vals else 0.01

    fig = go.Figure(go.Bar(
        x=vals_r, y=labels_r,
        orientation="h",
        marker={"color": colors_r, "opacity": 0.88,
                "line": {"color": "rgba(0,0,0,0)", "width": 0}},
        text=text_r,
        textposition="outside",
        textfont={"size": 12, "color": _TEXT},
        hovertemplate=(
            "<b>%{y}</b><br>"
            "SHAP: %{text}<br>"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_color=_BORDER, line_width=1.5)

    fig.update_layout(**_dark_layout(
        title = {
            "text": f"Top {top_n} Features by SHAP Importance  ·  Red = risk ↑  Green = risk ↓",
            "font": {"size": 15, "color": _TEXT},
            "x": 0.01,
        },
        xaxis = dict(
            title      = "SHAP value (probability contribution)",
            tickformat = ".3f",
            gridcolor  = _GRID,
            tickfont   = {"color": _SUBTEXT, "size": 11},
            title_font = {"color": _SUBTEXT, "size": 12},
            range      = [-x_max * 1.45, x_max * 1.45],
            zerolinecolor = _BORDER,
            showline   = True,
            linecolor  = _BORDER,
        ),
        yaxis = dict(
            tickfont   = {"color": _TEXT, "size": 12},
            automargin = True,
            showgrid   = False,
        ),
        height = max(520, 42 * top_n + 100),
        margin = {"l": 12, "r": 100, "t": 65, "b": 55},
    ))
    return fig


# ── page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="RiskLens AI", layout="wide")
st.title("RiskLens AI")
st.caption("Private Equity Due Diligence Copilot")

artifacts = load_artifacts(ROOT / "artifacts")
sample_name, sample_financials = load_sample()

# ── sidebar ───────────────────────────────────────────────────────────────────

# ── Ratio catalogue ──────────────────────────────────────────────────────────
# Each entry: (attr_key, display_name, tooltip, group, default)
# Groups: Profitability | Liquidity | Leverage | Cash Flow | Growth | Risk Indicators
_RATIOS: list[tuple[str, str, str, str, float]] = [
    # Profitability
    ("Attr1",  "Return on Assets (ROA)",
     "Net profit ÷ total assets. Measures how efficiently the company generates profit "
     "from its asset base. Negative ROA means the company is loss-making.",
     "Profitability", 0.04),
    ("Attr7",  "EBIT / Total Assets",
     "Earnings before interest & tax divided by total assets. Measures operating "
     "profitability before financing costs — the purest measure of business performance.",
     "Profitability", 0.06),
    ("Attr13", "EBITDA Margin",
     "(Gross profit + depreciation) ÷ sales. Cash earnings per £1 of revenue. "
     "Below 10% leaves little room for debt service.",
     "Profitability", 0.12),
    ("Attr19", "Gross Profit Margin",
     "Gross profit ÷ sales. Revenue remaining after direct cost of goods sold. "
     "Negative means the company sells below cost.",
     "Profitability", 0.22),
    ("Attr23", "Net Profit Margin",
     "Net profit ÷ sales. Final bottom-line margin after all costs, interest, and tax. "
     "Persistent negatives deplete cash reserves.",
     "Profitability", 0.04),
    ("Attr6",  "Retained Earnings / Assets",
     "Accumulated historical profits divided by total assets. Negative signals "
     "a sustained history of losses eroding equity.",
     "Profitability", 0.15),
    ("Attr24", "3-Year Avg Gross Profit / Assets",
     "Three-year average gross profit divided by total assets. Smoothed profitability "
     "metric that normalises single-year anomalies.",
     "Profitability", 0.09),
    ("Attr18", "Gross Profit / Total Assets",
     "Gross profit divided by total assets. Measures product-level returns on the "
     "asset base before operating expenses.",
     "Profitability", 0.10),

    # Liquidity
    ("Attr4",  "Current Ratio",
     "Current assets ÷ short-term liabilities. Values below 1.0 mean current "
     "obligations exceed liquid assets — an immediate liquidity warning.",
     "Liquidity", 1.80),
    ("Attr3",  "Working Capital / Assets",
     "(Current assets − current liabilities) ÷ total assets. Altman Z-score "
     "component. Negative values indicate the company cannot fund daily operations.",
     "Liquidity", 0.12),
    ("Attr40", "Cash Ratio",
     "(Current assets − inventory − receivables) ÷ short-term liabilities. "
     "Strictest liquidity test — counts only cash and near-cash.",
     "Liquidity", 0.25),
    ("Attr46", "Quick Ratio",
     "(Current assets − inventory) ÷ short-term liabilities. Excludes slow-moving "
     "inventory. Below 0.8 signals near-term liquidity stress.",
     "Liquidity", 1.20),
    ("Attr5",  "Cash Conversion Days",
     "Days the company can survive on liquid assets at current spending. "
     "Under 30 days is critical.",
     "Liquidity", 60.0),
    ("Attr55", "Working Capital (Absolute £)",
     "Current assets minus current liabilities in absolute monetary terms. "
     "Relevant for covenant calculations set in absolute amounts.",
     "Liquidity", 200_000.0),

    # Leverage
    ("Attr2",  "Debt Ratio (Total Liabilities / Assets)",
     "Total liabilities ÷ total assets. Above 0.80 is highly leveraged; "
     "above 1.0 means liabilities exceed assets — technically insolvent on book value.",
     "Leverage", 0.45),
    ("Attr8",  "Equity / Total Liabilities",
     "Book value of equity ÷ total liabilities. Altman Z-score component. "
     "Below 0.5 indicates debt dominates the capital structure.",
     "Leverage", 1.20),
    ("Attr10", "Equity Ratio",
     "Total equity ÷ total assets. The complement of the debt ratio. "
     "Below 0.20 leaves very little equity cushion.",
     "Leverage", 0.55),
    ("Attr59", "Long-term Debt / Equity",
     "Long-term liabilities ÷ equity. Above 2.0 is generally high leverage. "
     "Undefined when equity is negative.",
     "Leverage", 0.80),
    ("Attr17", "Solvency Ratio (Assets / Liabilities)",
     "Total assets ÷ total liabilities. Below 1.05 leaves almost no margin. "
     "Direct insolvency indicator.",
     "Leverage", 2.20),
    ("Attr51", "Short-term Liabilities / Assets",
     "Short-term debt as a fraction of total assets. High values create rollover risk — "
     "lenders refusing to renew credit triggers instant distress.",
     "Leverage", 0.25),

    # Cash Flow
    ("Attr27", "Interest Coverage Ratio",
     "Operating profit ÷ financial expenses. Below 1.5× is a standard covenant trigger; "
     "below 1.0× means operations cannot cover interest payments.",
     "Cash Flow", 4.50),
    ("Attr26", "Cash Flow / Total Liabilities",
     "(Net profit + depreciation) ÷ total liabilities. How much of total debt "
     "can be covered by operating cash flow. Below 0.05 is severe.",
     "Cash Flow", 0.12),
    ("Attr16", "Debt Coverage Ratio",
     "(Gross profit + depreciation) ÷ total liabilities. Cash earnings relative "
     "to total debt. The inverse of debt payback days.",
     "Cash Flow", 0.25),
    ("Attr15", "Debt Payback Days",
     "Total liabilities × 365 ÷ (gross profit + depreciation). How many days of "
     "EBITDA-equivalent profit it would take to repay all liabilities.",
     "Cash Flow", 400.0),
    ("Attr41", "Debt Repayment Period (Months)",
     "Total liabilities ÷ monthly EBITDA equivalent. Directly maps to Debt/EBITDA "
     "in months — the primary PE leverage metric.",
     "Cash Flow", 25.0),

    # Growth
    ("Attr21", "Revenue Growth Rate",
     "Sales(year n) ÷ sales(year n−1). Values above 1.0 signal growth; "
     "below 1.0 means declining revenue — accelerating fixed-cost pressure.",
     "Growth", 1.05),
    ("Attr9",  "Asset Turnover",
     "Sales ÷ total assets. Altman Z-score component. How efficiently the company "
     "converts its asset base into revenue.",
     "Growth", 1.20),
    ("Attr60", "Inventory Turnover (×)",
     "Sales ÷ inventory. How many times per year inventory is sold and replaced. "
     "Low values indicate slow-moving or potentially obsolete stock.",
     "Growth", 6.00),
    ("Attr44", "Days Sales Outstanding",
     "Receivables × 365 ÷ sales. How long customers take to pay. "
     "Above 90 days signals collection problems.",
     "Growth", 45.0),
    ("Attr20", "Days Inventory Outstanding",
     "Inventory × 365 ÷ sales. Days of sales sitting in inventory. "
     "High values tie up cash and may indicate unsellable stock.",
     "Growth", 55.0),

    # Risk Indicators
    ("Attr58", "Cost Ratio (Total Costs / Revenue)",
     "Total costs ÷ total sales. Above 1.0 means costs exceed revenue — "
     "the company is burning cash on every unit of sales.",
     "Risk Indicators", 0.82),
    ("Attr29", "Company Size (Log Assets)",
     "Natural log of total assets. Larger companies have better capital market access "
     "and historically lower default rates.",
     "Risk Indicators", 13.5),
    ("Attr25", "Excess Equity / Assets",
     "(Equity − share capital) ÷ total assets. Negative means accumulated losses "
     "have wiped out the equity premium over par value.",
     "Risk Indicators", 0.10),
    ("Attr38", "Constant Capital / Assets",
     "(Equity + long-term liabilities) ÷ total assets. Fraction of assets funded "
     "by stable long-term capital rather than short-term borrowing.",
     "Risk Indicators", 0.75),
    ("Attr30", "Net Debt / Sales",
     "(Total liabilities − cash) ÷ sales. Net debt relative to revenue. "
     "Values above 2.0 indicate severe leverage.",
     "Risk Indicators", 0.80),
]

# Group order for display
_GROUP_ORDER = [
    "Profitability",
    "Liquidity",
    "Leverage",
    "Cash Flow",
    "Growth",
    "Risk Indicators",
]

# Group icons
_GROUP_ICON = {
    "Profitability":   "📈",
    "Liquidity":       "💧",
    "Leverage":        "⚖️",
    "Cash Flow":       "💵",
    "Growth":          "📊",
    "Risk Indicators": "🚨",
}


with st.sidebar:
    st.header("Company Inputs")
    company_name = st.text_input("Company name", value=sample_name)

    # ── Excel analyst workflow ────────────────────────────────────────────────
    st.download_button(
        "📥 Download Excel Template",
        data=open(ROOT / "risklens_excel_template.xlsx", "rb").read(),
        file_name="risklens_excel_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    uploaded_excel = st.file_uploader(
        "📊 Upload Financial Statements",
        type=["xlsx"],
    )

    with st.expander("⚙️ Developer / API Upload", expanded=False):
        uploaded_json = st.file_uploader(
            "Advanced JSON Upload",
            type=["json"],
        )

    if uploaded_excel:
        parsed = parse_financial_excel(uploaded_excel)
        financials = parsed["ratios"]

        st.success(f"Loaded {len(financials)} computed ratios from Excel.")

        if parsed["warnings"]:
            for warning in parsed["warnings"]:
                st.warning(warning)

        with st.expander("📋 Extracted Financial Statements", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    parsed["line_items"].items(),
                    columns=["Line Item", "Value"],
                ),
                use_container_width=True,
            )

        with st.expander("📊 Computed Ratios", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    parsed["ratios"].items(),
                    columns=["Ratio", "Value"],
                ),
                use_container_width=True,
            )

    elif uploaded_json:
        payload = json.loads(uploaded_json.read().decode("utf-8"))
        financials = payload.get("financials", payload)

        st.success(
            f"Loaded {len(financials)} ratios from JSON. "
            "Edit values below to override individual figures."
        )

    else:
        financials = sample_financials.copy()

    st.divider()

    # ── Search / filter ───────────────────────────────────────────────────────
    search_query = st.text_input(
        "🔍 Search ratios",
        placeholder="e.g. debt, margin, coverage…",
        help="Filter displayed ratios by name or keyword.",
    ).strip().lower()

    # ── Render grouped ratio toolkit ──────────────────────────────────────────
    by_group: dict[str, list] = {g: [] for g in _GROUP_ORDER}
    for entry in _RATIOS:
        by_group[entry[3]].append(entry)

    for group in _GROUP_ORDER:
        entries = by_group[group]

        if search_query:
            entries = [
                e for e in entries
                if search_query in e[1].lower() or search_query in e[2].lower()
            ]

        if not entries:
            continue

        icon = _GROUP_ICON[group]
        expanded = group in ("Profitability", "Liquidity")

        with st.expander(f"{icon} {group} ({len(entries)} ratios)", expanded=expanded):
            for attr, label, tooltip, _grp, default in entries:
                current_val = financials.get(attr, default)

                try:
                    current_val = float(current_val)
                except (TypeError, ValueError):
                    current_val = float(default)

                financials[attr] = st.number_input(
                    label=label,
                    value=current_val,
                    format="%.5f",
                    help=tooltip,
                    key=f"input_{attr}",
                )

    # Surface any Attr keys from JSON/Excel that are not shown in the manual toolkit
    catalogue_keys = {e[0] for e in _RATIOS}

    for key, val in list(financials.items()):
        if key not in catalogue_keys and key.startswith("Attr"):
            try:
                financials[key] = float(val)
            except (TypeError, ValueError):
                pass

# ── scoring & explainability ──────────────────────────────────────────────────

score, x_model = score_company(financials, artifacts, company_name=company_name)
score          = explain_with_shap(score, x_model, artifacts)
shap_result    = shap_explanation(x_model, artifacts)   # raw SHAP data for tab
memo           = generate_investment_memo(score)
questions      = generate_due_diligence_questions(score)

if score.used_demo_scoring:
    st.warning(
        "No trained model artifact found — dashboard is running in deterministic demo mode."
    )

# ── header KPIs ───────────────────────────────────────────────────────────────

left, center, right = st.columns([1.15, 1, 1])
with left:
    st.plotly_chart(gauge(score.risk_score), use_container_width=True)
with center:
    st.metric("Risk tier", score.risk_tier)
    st.metric("Bankruptcy probability", f"{score.probability:.1%}")
with right:
    st.metric("ROC-AUC",  score.metrics.get("roc_auc", "—"))
    st.metric("PR-AUC",   score.metrics.get("pr_auc",  "—"))
    st.metric("Recall",   score.metrics.get("recall",  "—"))

# ── tabs ──────────────────────────────────────────────────────────────────────

tab_shap, tab_risk, tab_questions, tab_memo, tab_export = st.tabs(
    ["SHAP Explainability", "Risk Drivers", "Due Diligence", "Investment Memo", "PDF Report"]
)

# ── SHAP tab ──────────────────────────────────────────────────────────────────

with tab_shap:
    if shap_result is None:
        st.warning(
            "SHAP explainer artifact not found. "
            "Place `explainer.joblib` in the `artifacts/` directory to enable this tab."
        )
    else:
        # ── Section header ────────────────────────────────────────────────────
        st.markdown("## SHAP Explainability")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) shows exactly which financial "
            "features drove this company's risk score, and by how much. "
            "Values are in **probability space** — a SHAP of +0.05 means that "
            "feature raised bankruptcy probability by 5 percentage points."
        )

        # ── Summary metric cards ──────────────────────────────────────────────
        prob_change = shap_result.predicted_probability - shap_result.base_value
        change_sign = "+" if prob_change >= 0 else ""

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            label="Base Probability",
            value=f"{shap_result.base_value:.1%}",
            help="Average bankruptcy probability across the training population.",
        )
        c2.metric(
            label="Predicted Probability",
            value=f"{shap_result.predicted_probability:.1%}",
            help="Model's bankruptcy probability estimate for this company.",
        )
        c3.metric(
            label="Probability Change",
            value=f"{change_sign}{prob_change:.1%}",
            delta=f"{change_sign}{prob_change:.1%}",
            delta_color="inverse",
            help="How much this company deviates from the population base.",
        )
        c4.metric(
            label="Features Analysed",
            value=len(shap_result.feature_names),
            help="Total features passed through the SHAP explainer.",
        )

        st.divider()

        # ── Waterfall chart ───────────────────────────────────────────────────
        st.markdown("### Waterfall Chart")
        st.caption(
            "Each bar shows a single feature's contribution to the final "
            "prediction. Bars cascade left-to-right from the base probability. "
            "🔴 Red = pushes toward bankruptcy · 🟢 Green = reduces risk."
        )
        st.plotly_chart(
            _waterfall(shap_result),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.divider()

        # ── Bar chart ─────────────────────────────────────────────────────────
        st.markdown("### Feature Importance Ranking")
        st.caption(
            "Top 15 features ranked by absolute SHAP value — "
            "how much each feature moves the bankruptcy probability on average."
        )
        st.plotly_chart(
            _bar_chart(shap_result, top_n=15),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.divider()

        # ── Risk drivers & mitigants tables ───────────────────────────────────
        st.markdown("### Risk Drivers & Mitigants")

        col_risk, col_mit = st.columns(2, gap="large")

        def _driver_table(features, shaps, feat_vals, title, color_hex):
            rows = []
            for feat, sv_val, fv_val in zip(features[:5], shaps[:5], feat_vals[:5]):
                rows.append({
                    "Feature":       feat,
                    "Direction":     "↑ Increases Risk" if sv_val >= 0 else "↓ Reduces Risk",
                    "SHAP Impact":   f"{'+' if sv_val >= 0 else ''}{sv_val:.3f}",
                    "Current Value": f"{fv_val:.4f}",
                })
            if rows:
                df_tbl = pd.DataFrame(rows)
                st.markdown(
                    f"<span style='color:{color_hex};font-size:1rem;font-weight:600'>"
                    f"{title}</span>",
                    unsafe_allow_html=True,
                )
                st.dataframe(df_tbl, use_container_width=True, hide_index=True)

        with col_risk:
            _driver_table(
                shap_result.positive_features,
                shap_result.positive_shap,
                shap_result.positive_values,
                "🔴 Top Risk Drivers",
                _RED,
            )

        with col_mit:
            _driver_table(
                shap_result.negative_features,
                shap_result.negative_shap,
                shap_result.negative_values,
                "🟢 Top Mitigating Factors",
                _GREEN,
            )

        st.divider()

        # ── Full contribution table ────────────────────────────────────────────
        with st.expander("📋 Full Feature Contribution Table (top 30)", expanded=False):
            display_df = shap_result.contribution_df[[
                "Label", "SHAP Value", "Feature Value", "Direction"
            ]].copy()
            display_df.columns = ["Feature", "SHAP Impact", "Current Value", "Direction"]
            display_df["SHAP Impact"]   = display_df["SHAP Impact"].apply(
                lambda v: f"{'+' if v >= 0 else ''}{v:.3f}"
            )
            display_df["Current Value"] = display_df["Current Value"].apply(
                lambda v: f"{v:.4f}"
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.caption(
            "ℹ️ SHAP values are in probability space. "
            "The waterfall bars sum exactly to: "
            f"Base ({shap_result.base_value:.3f}) + contributions = "
            f"Prediction ({shap_result.predicted_probability:.3f})."
        )

# ── Risk Drivers tab (unchanged) ──────────────────────────────────────────────

with tab_risk:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top risk drivers")
        if score.top_risks:
            st.dataframe(
                pd.DataFrame([f.__dict__ for f in score.top_risks]),
                use_container_width=True,
            )
        else:
            st.info("No risk factors computed — ensure explainer.joblib is present.")
    with c2:
        st.subheader("Mitigating factors")
        if score.top_mitigants:
            st.dataframe(
                pd.DataFrame([f.__dict__ for f in score.top_mitigants]),
                use_container_width=True,
            )
        else:
            st.info("No mitigating factors computed.")

# ── Due Diligence tab (unchanged) ─────────────────────────────────────────────

with tab_questions:
    for idx, question in enumerate(questions, start=1):
        st.write(f"{idx}. {question}")

# ── Investment Memo tab (unchanged) ───────────────────────────────────────────

with tab_memo:
    st.text_area("Memo", memo, height=520)

# ── PDF Report tab (unchanged) ────────────────────────────────────────────────

with tab_export:
    output_path = (
        ROOT / "outputs"
        / f"{company_name.lower().replace(' ', '_')}_risklens_report.pdf"
    )
    if st.button("Generate PDF"):
        created = write_text_pdf(memo, output_path)
        st.success(f"PDF generated: {created}")
    if output_path.exists():
        st.download_button(
            "Download PDF",
            data=output_path.read_bytes(),
            file_name=output_path.name,
            mime="application/pdf",
        )
