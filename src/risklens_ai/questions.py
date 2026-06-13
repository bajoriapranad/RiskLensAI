"""Due diligence question generation from model explanations."""

from __future__ import annotations

from .scoring import RiskFactor, RiskScore


def questions_for_factor(factor: RiskFactor) -> list[str]:
    label = factor.label.lower()
    feature = factor.feature
    if feature == "Attr2":
        return [
            "What portion of liabilities is near-term versus long-term?",
            "Are there covenant breaches, refinancing events, or maturity walls in the next 24 months?",
        ]
    if feature in {"Attr1", "Attr7", "Attr23"}:
        return [
            "What is driving weak profitability: pricing, volume, cost inflation, or one-off items?",
            "How much of the margin weakness is expected to reverse under the investment plan?",
        ]
    if feature in {"Attr3", "Attr4"}:
        return [
            "Is working capital stress seasonal, structural, or caused by customer/vendor concentration?",
            "What liquidity sources are available if collections slow or suppliers tighten terms?",
        ]
    if feature == "Attr58":
        return [
            "Which cost lines are rising faster than revenue?",
            "Are there credible cost actions already validated by management?",
        ]
    return [
        f"What business event explains the unfavorable movement in {label}?",
        f"Is {label} consistent with peer companies and prior-year performance?",
    ]


def generate_due_diligence_questions(score: RiskScore, max_questions: int = 8) -> list[str]:
    questions: list[str] = []
    for factor in score.top_risks:
        questions.extend(questions_for_factor(factor))
    if score.risk_tier == "HIGH":
        questions.append("What downside case should the investment committee underwrite before proceeding?")
    return questions[:max_questions]

