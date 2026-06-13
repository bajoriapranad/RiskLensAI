"""RiskLensPreprocessor — standalone module so joblib can unpickle it."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RiskLensPreprocessor(BaseEstimator, TransformerMixin):
    DROP_COLS = ["Attr37"]
    DUP_COLS  = ["Attr36", "Attr56"]
    LOG_COLS  = ["Attr4","Attr15","Attr17","Attr20","Attr27",
                 "Attr32","Attr44","Attr50","Attr60","Attr61"]

    def fit(self, X: pd.DataFrame, y=None):
        df = X.drop(columns=[c for c in self.DROP_COLS + self.DUP_COLS if c in X.columns]).copy()
        self.winsor_bounds_: dict = {}
        for col in df.columns:
            self.winsor_bounds_[col] = (float(df[col].quantile(0.01)),
                                        float(df[col].quantile(0.99)))
        df_w = df.copy()
        for col, (lo, hi) in self.winsor_bounds_.items():
            df_w[col] = df_w[col].clip(lo, hi)
        self.medians_: dict = df_w.median().to_dict()
        self.base_cols_: list = df.columns.tolist()
        self.feature_names_out_: list = self._compute_feature_names()
        return self

    def _compute_feature_names(self) -> list:
        names = list(self.base_cols_)
        names += [f"{c}_missing" for c in self.base_cols_]
        names += [f"{c}_log" for c in self.LOG_COLS if c in self.base_cols_]
        names += ["altman_z","altman_zone",
                  "flag_roa_neg","flag_debt_high","flag_insolvent",
                  "flag_wc_neg","flag_ebit_neg","flag_margin_neg","flag_cost_over_1"]
        return names

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.drop(columns=[c for c in self.DROP_COLS + self.DUP_COLS if c in X.columns]).copy()
        for col in self.base_cols_:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.base_cols_].copy()

        miss = {f"{c}_missing": df[c].isna().astype(float) for c in self.base_cols_}

        for col, (lo, hi) in self.winsor_bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        for col, med in self.medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        logs = {}
        for col in self.LOG_COLS:
            if col in df.columns:
                v = df[col].values.astype(float)
                logs[f"{col}_log"] = np.sign(v) * np.log1p(np.abs(v))

        zw = {"Attr3":1.2,"Attr6":1.4,"Attr7":3.3,"Attr8":0.6,"Attr9":1.0}
        altman_z = (sum(w * df[c].values for c, w in zw.items())
                    if all(c in df.columns for c in zw) else np.zeros(len(df)))
        altman_zone = np.where(altman_z < 1.81, 0, np.where(altman_z < 2.99, 1, 2)).astype(float)

        def _col(name, default=0.0):
            return df[name].values if name in df.columns else np.full(len(df), default)

        flags = {
            "flag_roa_neg":     (_col("Attr1")  < 0).astype(float),
            "flag_debt_high":   (_col("Attr2")  > 0.80).astype(float),
            "flag_insolvent":   (_col("Attr2")  >= 1.0).astype(float),
            "flag_wc_neg":      (_col("Attr3")  < 0).astype(float),
            "flag_ebit_neg":    (_col("Attr7")  < 0).astype(float),
            "flag_margin_neg":  (_col("Attr19") < 0).astype(float),
            "flag_cost_over_1": (_col("Attr58") > 1.0).astype(float),
        }

        out = df.copy()
        for k, v in miss.items():   out[k] = v.values
        for k, v in logs.items():   out[k] = v
        out["altman_z"]    = altman_z
        out["altman_zone"] = altman_zone
        for k, v in flags.items():  out[k] = v
        for col in self.feature_names_out_:
            if col not in out.columns:
                out[col] = 0.0
        return out[self.feature_names_out_]

    def get_feature_names_out(self):
        return self.feature_names_out_
