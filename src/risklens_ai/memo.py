"""Investment memo generation for analyst review."""

from __future__ import annotations

from datetime import date

from .questions import generate_due_diligence_questions
from .scoring import RiskScore


def generate_investment_memo(score: RiskScore) -> str:
    risks = "\n".join(
        f"- {factor.label}: {factor.interpretation} Contribution={factor.contribution:.3f}"
        for factor in score.top_risks
    ) or "- No material model risk drivers available."
    mitigants = "\n".join(
        f"- {factor.label}: {factor.interpretation} Contribution={factor.contribution:.3f}"
        for factor in score.top_mitigants
    ) or "- No model mitigants available."
    questions = "\n".join(f"- {q}" for q in generate_due_diligence_questions(score))

    recommendation = {
        "HIGH": "Proceed only with enhanced financial diligence and downside-case underwriting.",
        "MEDIUM": "Proceed with targeted diligence focused on the listed risk drivers.",
        "LOW": "Proceed with standard diligence; monitor model-identified watch items.",
    }[score.risk_tier]

    return f"""RiskLens AI Investment Memo
Date: {date.today().isoformat()}
Company: {score.company_name}
Model version: {score.model_version}

Executive View
Risk tier: {score.risk_tier}
Risk score: {score.risk_score:.1f}/100
Estimated bankruptcy probability: {score.probability:.1%}
Recommendation: {recommendation}

Model Context
ROC-AUC: {score.metrics.get("roc_auc", "n/a")}
PR-AUC: {score.metrics.get("pr_auc", "n/a")}
Recall: {score.metrics.get("recall", "n/a")}

Primary Risk Drivers
{risks}

Mitigating Factors
{mitigants}

Due Diligence Questions
{questions}

Analyst Note
RiskLens AI is a prioritization aid. It does not replace investment judgment,
quality of earnings work, legal review, management interviews, or investment
committee approval.
"""

