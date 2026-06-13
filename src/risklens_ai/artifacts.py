"""Load trained RiskLens artifacts without retraining models."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass
import json
from typing import Any

import joblib

@dataclass(frozen=True)
class RiskLensArtifacts:
    model: Any | None
    preprocessor: Any | None
    explainer: Any | None
    metadata: dict[str, Any]
    artifact_dir: Path

    @property
    def is_model_ready(self) -> bool:
        return self.model is not None

    @property
    def feature_names(self) -> list[str]:
        names = self.metadata.get("feature_names")
        if names:
            return list(names)
        return [f"Attr{i}" for i in range(1, 65)]


def _load_joblib(path: Path) -> Any | None:
    return joblib.load(path) if path.exists() else None


def load_artifacts(artifact_dir: str | Path = "artifacts") -> RiskLensArtifacts:
    """Load model, preprocessing, SHAP, and metadata artifacts if present."""
    root = Path(artifact_dir)
    metadata_path = root / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return RiskLensArtifacts(
        model=_load_joblib(root / "model.joblib"),
        preprocessor=_load_joblib(root / "preprocessor.joblib"),
        explainer=_load_joblib(root / "explainer.joblib"),
        metadata=metadata,
        artifact_dir=root,
    )

