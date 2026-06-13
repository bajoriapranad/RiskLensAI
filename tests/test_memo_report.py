from pathlib import Path
import tempfile

from risklens_ai.explainability import explain_with_directional_fallback
from risklens_ai.memo import generate_investment_memo
from risklens_ai.pdf_report import write_text_pdf
from risklens_ai.scoring import RiskScore


def test_memo_and_pdf_generation():
    score = RiskScore(
        company_name="Unit Test Co.",
        probability=0.72,
        risk_score=72.0,
        risk_tier="HIGH",
        model_version="test",
    )
    row = {"Attr1": -0.1, "Attr2": 0.9, "Attr3": -0.2, "Attr7": -0.05, "Attr58": 1.1}
    import pandas as pd

    score = explain_with_directional_fallback(score, pd.DataFrame([row]))
    memo = generate_investment_memo(score)
    assert "Unit Test Co." in memo
    assert "Due Diligence Questions" in memo

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf = write_text_pdf(memo, Path(tmpdir) / "report.pdf")
        assert pdf.exists()
        assert pdf.read_bytes().startswith(b"%PDF")

