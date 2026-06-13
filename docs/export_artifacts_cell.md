# Export Trained RiskLens Artifacts

Search result: no trained model artifact was found in this project workspace or
the checked RiskLens files in Downloads.

The embedded SHAP notebook text in `I am building an ML project called RiskLens AI.docx`
expects these older artifacts:

```text
risklens_artifacts/
  xgb_model.json
  imputer.joblib
  metadata.json
  X_train.npy
  X_test.npy
  y_test.npy
```

Those files were not present.

## Where To Run This

Run the cell below in the completed ML pipeline notebook, immediately after the
final XGBoost test-set evaluation cell that produced:

```text
ROC-AUC = 0.869
PR-AUC  = 0.495
```

It must be run after these fitted objects/variables exist in notebook memory:

- `model`: final fitted XGBoost model, or replace with your final XGBoost object name
- `imputer`: fitted `SimpleImputer`
- `feature_names`
- `impute_cols`
- `log_features`
- `drop_high_missing`
- `winsor_bounds`
- `drop_corr`
- optionally `optimal_threshold`

The available SHAP notebook text identifies the runtime model object as `model`
and the preprocessing state as `imputer` plus the metadata fields above. I did
not find a single sklearn `Pipeline` object in the available files.

## Export Cell

```python
# RiskLens AI: export trained production artifacts
# Run this AFTER the final XGBoost evaluation cell.
# This cell does not retrain anything.

from pathlib import Path
import json
import joblib

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# If your final trained XGBoost object is not named `model`, change this line.
production_model = model

# The available notebooks used a fitted SimpleImputer plus metadata, rather than
# a single sklearn Pipeline. RiskLens app code loads this as `preprocessor`.
preprocessor = imputer

joblib.dump(production_model, ARTIFACTS_DIR / "model.joblib")
joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")

metadata = {
    "model_name": "XGBoost",
    "model_version": "xgb-polish-3year-v1",
    "dataset": "Polish Companies Bankruptcy Dataset - 3year.arff",
    "target": "class",
    "positive_label": 1,
    "metrics": {
        "roc_auc": 0.869,
        "pr_auc": 0.495,
        "recall": 0.707,
    },
    "feature_names": list(feature_names),
    "impute_cols": list(impute_cols),
    "log_features": list(log_features),
    "drop_high_missing": list(drop_high_missing),
    "winsor_bounds": {
        key: [float(value[0]), float(value[1])]
        for key, value in winsor_bounds.items()
    },
    "drop_corr": list(drop_corr),
    "optimal_threshold": float(globals().get("optimal_threshold", 0.48)),
}

with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Exported RiskLens artifacts:")
print(f"  {ARTIFACTS_DIR / 'model.joblib'}")
print(f"  {ARTIFACTS_DIR / 'preprocessor.joblib'}")
print(f"  {ARTIFACTS_DIR / 'metadata.json'}")
```

## If Your Object Names Differ

Change only these aliases:

```python
production_model = model
preprocessor = imputer
```

For example:

```python
production_model = xgb_model
preprocessor = preprocessing_pipeline
```
