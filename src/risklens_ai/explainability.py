"""Real SHAP explainability for RiskLens AI.

Replaces any previous placeholder implementation.

Public API
----------
explain_with_shap(score, x_model, artifacts) -> RiskScore
    Populates score.top_risks and score.top_mitigants in-place and returns
    the mutated RiskScore.  Called by streamlit_app.py immediately after
    score_company().

shap_explanation(x_model, artifacts) -> SHAPResult | None
    Returns raw SHAP data needed by the dashboard SHAP tab.
    Returns None if explainer artifact is missing (graceful degradation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .scoring import RiskFactor, RiskScore, factor_from_contribution

# ---------------------------------------------------------------------------
# Feature-label map (short names for charts)
# ---------------------------------------------------------------------------
_SHORT_LABELS: dict[str, str] = {
    "Attr1":  "ROA",
    "Attr2":  "Debt Ratio",
    "Attr3":  "Working Capital/Assets",
    "Attr4":  "Current Ratio",
    "Attr5":  "Cash Runway Days",
    "Attr6":  "Retained Earnings/Assets",
    "Attr7":  "EBIT/Assets",
    "Attr8":  "Equity/Liabilities",
    "Attr9":  "Asset Turnover",
    "Attr10": "Equity/Assets",
    "Attr12": "Gross Profit/ST Liabilities",
    "Attr13": "EBITDA Margin",
    "Attr15": "Debt Payback Days",
    "Attr16": "Debt Coverage",
    "Attr17": "Solvency Ratio",
    "Attr19": "Gross Margin",
    "Attr20": "Days Inventory",
    "Attr21": "Revenue Growth",
    "Attr22": "Operating Profit/Assets",
    "Attr23": "Net Margin",
    "Attr24": "3yr Avg Profit/Assets",
    "Attr25": "Excess Equity/Assets",
    "Attr26": "Cash Flow/Liabilities",
    "Attr27": "Interest Coverage",
    "Attr28": "Working Capital/Fixed Assets",
    "Attr29": "Company Size (log assets)",
    "Attr30": "Net Debt/Sales",
    "Attr32": "Days Payable",
    "Attr33": "OpEx/ST Liabilities",
    "Attr34": "OpEx/Total Liabilities",
    "Attr35": "Operating Profit/Assets",
    "Attr38": "Constant Capital/Assets",
    "Attr39": "Operating Margin",
    "Attr40": "Cash Ratio",
    "Attr41": "Debt Repayment Months",
    "Attr42": "Operating Profit/Assets",
    "Attr43": "Receivables+Inventory Days",
    "Attr44": "Days Receivable",
    "Attr45": "Net Profit/Inventory",
    "Attr46": "Quick Ratio",
    "Attr47": "Days Inventory (alt)",
    "Attr48": "EBIT/Assets (alt)",
    "Attr49": "EBIT Margin",
    "Attr50": "Liquid Coverage",
    "Attr51": "ST Liabilities/Assets",
    "Attr52": "ST Payable Days",
    "Attr53": "Equity/Fixed Assets",
    "Attr54": "Constant Capital/Fixed Assets",
    "Attr55": "Working Capital (abs)",
    "Attr57": "Net Cash Surplus",
    "Attr58": "Cost Ratio",
    "Attr59": "LT Debt/Equity",
    "Attr60": "Inventory Turnover",
    "Attr61": "Receivables Turnover",
    "Attr62": "ST Liabilities/Sales Days",
    "Attr63": "Sales/ST Liabilities",
    "Attr64": "Sales/Fixed Assets",
    "altman_z":          "Altman Z-Score",
    "altman_zone":       "Altman Zone",
    "flag_roa_neg":      "Flag: Negative ROA",
    "flag_debt_high":    "Flag: High Leverage",
    "flag_insolvent":    "Flag: Balance-Sheet Insolvent",
    "flag_wc_neg":       "Flag: Negative Working Capital",
    "flag_ebit_neg":     "Flag: Negative EBIT",
    "flag_margin_neg":   "Flag: Negative Gross Margin",
    "flag_cost_over_1":  "Flag: Costs Exceed Revenue",
}


def _label(feature: str) -> str:
    """Return a short analyst-readable label for any feature name."""
    if feature in _SHORT_LABELS:
        return _SHORT_LABELS[feature]
    if feature.endswith("_missing"):
        base = feature[:-8]
        return f"{_label(base)} — data missing"
    if feature.endswith("_log"):
        base = feature[:-4]
        return f"{_label(base)} (log)"
    return feature


# ---------------------------------------------------------------------------
# SHAPResult — raw data bundle returned to the dashboard
# ---------------------------------------------------------------------------

@dataclass
class SHAPResult:
    """All SHAP artefacts needed to render charts in the dashboard."""

    feature_names: list[str]
    shap_values: np.ndarray          # shape (1, n_features)  — probability space
    feature_values: np.ndarray       # shape (1, n_features)  — preprocessed values
    base_value: float                # explainer expected value (base probability)
    predicted_probability: float

    # Sorted views (populated by _build_sorted)
    positive_features: list[str] = field(default_factory=list)   # increase risk
    positive_shap: list[float]   = field(default_factory=list)
    positive_values: list[float] = field(default_factory=list)

    negative_features: list[str] = field(default_factory=list)   # reduce risk
    negative_shap: list[float]   = field(default_factory=list)
    negative_values: list[float] = field(default_factory=list)

    contribution_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self._build_sorted()

    def _build_sorted(self, top_n: int = 15) -> None:
        sv  = self.shap_values[0]
        fv  = self.feature_values[0]
        fn  = self.feature_names

        # Sort all features by absolute SHAP descending
        order = np.argsort(np.abs(sv))[::-1]

        pos_f, pos_s, pos_v = [], [], []
        neg_f, neg_s, neg_v = [], [], []

        for idx in order:
            name  = fn[idx]
            label = _label(name)
            val   = float(fv[idx])
            s     = float(sv[idx])
            if s >= 0 and len(pos_f) < top_n:
                pos_f.append(label); pos_s.append(s); pos_v.append(val)
            elif s < 0 and len(neg_f) < top_n:
                neg_f.append(label); neg_s.append(s); neg_v.append(val)
            if len(pos_f) >= top_n and len(neg_f) >= top_n:
                break

        self.positive_features = pos_f
        self.positive_shap     = pos_s
        self.positive_values   = pos_v
        self.negative_features = neg_f
        self.negative_shap     = neg_s
        self.negative_values   = neg_v

        # Full contribution table
        rows = []
        for idx in np.argsort(np.abs(sv))[::-1][:30]:
            rows.append({
                "Feature":      fn[idx],
                "Label":        _label(fn[idx]),
                "SHAP Value":   round(float(sv[idx]), 5),
                "Feature Value":round(float(fv[idx]), 4),
                "Direction":    "increases risk" if sv[idx] >= 0 else "reduces risk",
            })
        self.contribution_df = pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def shap_explanation(
    x_model:   pd.DataFrame,
    artifacts: Any,
) -> SHAPResult | None:
    """
    Compute SHAP values for a single preprocessed row.

    Parameters
    ----------
    x_model   : preprocessed feature DataFrame (1 row, 141 cols)
    artifacts : RiskLensArtifacts — must have .explainer and .model set

    Returns
    -------
    SHAPResult if explainer is present, else None.
    """
    if artifacts.explainer is None or artifacts.model is None:
        return None

    feature_names = list(x_model.columns)
    X = x_model.values  # (1, n_features)

    # Compute SHAP values in probability space
    sv = artifacts.explainer.shap_values(X)       # (1, n_features)
    if isinstance(sv, list):
        # Some SHAP versions return [neg_class, pos_class]
        sv = sv[1]
    sv = np.atleast_2d(sv)

    base_val  = float(np.atleast_1d(artifacts.explainer.expected_value)[0])
    pred_prob = float(artifacts.model.predict_proba(X)[:, 1][0])

    return SHAPResult(
        feature_names        = feature_names,
        shap_values          = sv,
        feature_values       = X,
        base_value           = base_val,
        predicted_probability= pred_prob,
    )


def explain_with_shap(
    score:     RiskScore,
    x_model:   pd.DataFrame,
    artifacts: Any,
    top_n:     int = 5,
) -> RiskScore:
    """
    Populate score.top_risks and score.top_mitigants using real SHAP values.

    Falls back to empty lists (demo behaviour) if explainer is absent.
    Always returns the score object.
    """
    result = shap_explanation(x_model, artifacts)
    if result is None:
        return score

    sv = result.shap_values[0]
    fv = result.feature_values[0]
    fn = result.feature_names

    order = np.argsort(np.abs(sv))[::-1]
    top_risks: list[RiskFactor]     = []
    top_mitigants: list[RiskFactor] = []

    for idx in order:
        if len(top_risks) >= top_n and len(top_mitigants) >= top_n:
            break
        rf = factor_from_contribution(fn[idx], fv[idx], float(sv[idx]))
        if sv[idx] >= 0 and len(top_risks) < top_n:
            top_risks.append(rf)
        elif sv[idx] < 0 and len(top_mitigants) < top_n:
            top_mitigants.append(rf)

    score.top_risks     = top_risks
    score.top_mitigants = top_mitigants
    return score
