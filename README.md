# RiskLens AI

Private Equity Due Diligence Copilot for financial distress screening.

This repository intentionally does not retrain models. It is the application
layer around the completed XGBoost model:

1. Load trained artifacts from `artifacts/`
2. Score company financial ratios
3. Explain risk drivers with SHAP
4. Generate due diligence questions and investment memos
5. Export analyst-ready PDF reports
6. Present the workflow in Streamlit

## Current Inputs Reviewed

- `risklens_eda.ipynb`: EDA-only notebook for `3year.arff`
- `3year.arff`: Polish bankruptcy dataset, 10,503 rows, 64 features, target `class`
- `I am building an ML project called RiskLens AI.docx`: architecture and product planning notes

## Expected Artifacts

Place completed model artifacts here:

```text
artifacts/
  model.joblib
  preprocessor.joblib        # optional, if preprocessing was saved separately
  explainer.joblib           # optional, fitted SHAP explainer
  metadata.json
```

`metadata.json` should include at least:

```json
{
  "model_name": "XGBoost",
  "model_version": "xgb-polish-3year-v1",
  "feature_names": ["Attr1", "Attr2", "..."],
  "metrics": {
    "roc_auc": 0.869,
    "pr_auc": 0.495,
    "recall": 0.707
  }
}
```

## Run Dashboard

Install the project once in editable mode:

```powershell
python -m pip install -e .
```

Then run tests and the dashboard without setting `PYTHONPATH`:

```powershell
pytest
```

```powershell
streamlit run app/streamlit_app.py
```

If artifacts are missing, the app still runs in demo mode using deterministic
rule-based scoring so the UI, memo, and PDF workflow can be tested.

## Package Layout

```text
src/risklens_ai/          # installable Python package
app/streamlit_app.py      # Streamlit entrypoint
artifacts/                # local trained model artifacts, not committed
tests/                    # pytest suite
```
