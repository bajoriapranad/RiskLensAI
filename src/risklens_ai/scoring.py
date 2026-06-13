"""Risk scoring contracts and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .data_dictionary import feature_label


@dataclass
class RiskFactor:
    feature: str
    label: str
    value: float | None
    contribution: float
    direction: str
    interpretation: str


@dataclass
class RiskScore:
    company_name: str
    probability: float
    risk_score: float
    risk_tier: str
    model_version: str
    metrics: dict[str, float] = field(default_factory=dict)
    top_risks: list[RiskFactor] = field(default_factory=list)
    top_mitigants: list[RiskFactor] = field(default_factory=list)
    used_demo_scoring: bool = False


def tier_from_score(score: float) -> str:
    if score >= 65:
        return "HIGH"
    if score >= 35:
        return "MEDIUM"
    return "LOW"


def _to_frame(raw_financials: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    row = {feature: raw_financials.get(feature, np.nan) for feature in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def _demo_probability(raw_financials: dict[str, Any]) -> float:
    """Deterministic fallback for UI testing when trained artifacts are absent."""
    debt = float(raw_financials.get("Attr2", 0.5) or 0.5)
    roa = float(raw_financials.get("Attr1", 0.0) or 0.0)
    wc = float(raw_financials.get("Attr3", 0.0) or 0.0)
    ebit = float(raw_financials.get("Attr7", 0.0) or 0.0)
    cost = float(raw_financials.get("Attr58", 0.9) or 0.9)
    z = 2.7 * debt - 3.0 * roa - 1.8 * wc - 2.2 * ebit + 1.4 * max(cost - 1.0, 0.0) - 1.0
    return float(1.0 / (1.0 + np.exp(-z)))


def score_company(
    raw_financials: dict[str, Any],
    artifacts: Any,
    company_name: str = "Target Company",
) -> tuple[RiskScore, pd.DataFrame]:
    """Score one company using loaded artifacts, falling back to demo mode if absent."""
    feature_names = artifacts.feature_names
    x_raw = _to_frame(raw_financials, feature_names)
    x_model = artifacts.preprocessor.transform(x_raw) if artifacts.preprocessor is not None else x_raw

    used_demo = artifacts.model is None
    if artifacts.model is not None:
        probability = float(artifacts.model.predict_proba(x_model)[:, 1][0])
    else:
        probability = _demo_probability(raw_financials)

    risk_score = round(probability * 100, 1)
    score = RiskScore(
        company_name=company_name,
        probability=probability,
        risk_score=risk_score,
        risk_tier=tier_from_score(risk_score),
        model_version=artifacts.metadata.get("model_version", "artifact-missing-demo-mode"),
        metrics=artifacts.metadata.get("metrics", {}),
        used_demo_scoring=used_demo,
    )
    return score, pd.DataFrame(x_model, columns=getattr(x_model, "columns", feature_names))


def factor_from_contribution(feature: str, value: Any, contribution: float) -> RiskFactor:
    direction = "increases risk" if contribution >= 0 else "reduces risk"
    label = feature_label(feature)
    interpretation = f"{label} {direction} for this company."
    try:
        numeric_value = None if pd.isna(value) else float(value)
    except TypeError:
        numeric_value = None
    return RiskFactor(
        feature=feature,
        label=label,
        value=numeric_value,
        contribution=float(contribution),
        direction=direction,
        interpretation=interpretation,
    )

