"""Excel parser for RiskLens AI analyst financial template.

This module converts the user-facing Excel financial statement template into the
Attr1..Attr64 ratio dictionary expected by the existing RiskLens scoring pipeline.

Expected workbook format:
- Sheet: Inputs
- Columns: Line Item | Current Year | Prior Year | Unit | Required? | Notes

Usage:
    from risklens_ai.excel_parser import parse_financial_excel
    result = parse_financial_excel(uploaded_file)
    financials = result["ratios"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Any
import math

import pandas as pd


REQUIRED_LINE_ITEMS = [
    "Revenue / Sales",
    "Cost of Sales",
    "Operating Expenses",
    "Net Profit",
    "Gross Profit",
    "Operating Profit / EBIT",
    "EBITDA",
    "Depreciation",
    "Financial Expenses / Interest",
    "Total Assets",
    "Fixed Assets",
    "Current Assets",
    "Inventory",
    "Receivables",
    "Cash + Short-term Securities",
    "Total Liabilities",
    "Short-term Liabilities",
    "Long-term Liabilities",
    "Equity",
    "Retained Earnings",
    "Working Capital",
]

OPTIONAL_DEFAULTS = {
    "Profit on Sales": None,
    "Extraordinary Items": 0.0,
    "Share Capital": 0.0,
    "Constant Capital": None,
    "Three-year Gross Profit": None,
}


def _clean_number(value: Any) -> float | None:
    """Convert Excel cell value to float or None."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if value in {"", "-", "n/a", "N/A", "NA", "na"}:
            return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = numerator / denominator
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _safe_log(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return math.log(value)


@dataclass
class ExcelParseResult:
    line_items: dict[str, float | None]
    ratios: dict[str, float | None]
    warnings: list[str]
    missing_required: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_items": self.line_items,
            "ratios": self.ratios,
            "warnings": self.warnings,
            "missing_required": self.missing_required,
        }


def parse_financial_excel(file: str | BinaryIO) -> dict[str, Any]:
    """Parse analyst Excel template and return ratios for RiskLens scoring.

    Parameters
    ----------
    file:
        Path, file-like object, or Streamlit UploadedFile.

    Returns
    -------
    dict with keys:
        - line_items: extracted financial statement values
        - ratios: Attr1..Attr64 mapping
        - warnings: validation warnings
        - missing_required: missing required line items
    """
    df = pd.read_excel(file, sheet_name="Inputs", header=2)

    expected_cols = {"Line Item", "Current Year", "Prior Year"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            "Invalid RiskLens template. Missing columns: " + ", ".join(sorted(missing_cols))
        )

    df = df.dropna(subset=["Line Item"])
    line_items: dict[str, float | None] = {}
    prior_items: dict[str, float | None] = {}

    for _, row in df.iterrows():
        label = str(row["Line Item"]).strip()
        if not label:
            continue
        line_items[label] = _clean_number(row.get("Current Year"))
        prior_items[label] = _clean_number(row.get("Prior Year"))

    # Defaults / derived fallback values
    for key, default in OPTIONAL_DEFAULTS.items():
        line_items.setdefault(key, default)
        prior_items.setdefault(key, default)

    # If omitted, derive common optional values.
    if line_items.get("Working Capital") is None:
        ca = line_items.get("Current Assets")
        stl = line_items.get("Short-term Liabilities")
        line_items["Working Capital"] = None if ca is None or stl is None else ca - stl

    if line_items.get("Constant Capital") is None:
        eq = line_items.get("Equity")
        ltl = line_items.get("Long-term Liabilities")
        line_items["Constant Capital"] = None if eq is None or ltl is None else eq + ltl

    if line_items.get("Profit on Sales") is None:
        line_items["Profit on Sales"] = line_items.get("Operating Profit / EBIT")

    if line_items.get("Three-year Gross Profit") is None:
        gp = line_items.get("Gross Profit")
        line_items["Three-year Gross Profit"] = None if gp is None else gp * 3

    missing_required = [k for k in REQUIRED_LINE_ITEMS if line_items.get(k) is None]
    warnings: list[str] = []
    if missing_required:
        warnings.append("Missing required line items: " + ", ".join(missing_required))

    def v(name: str) -> float | None:
        return line_items.get(name)

    def vp(name: str) -> float | None:
        return prior_items.get(name)

    # Shorthands
    S = v("Revenue / Sales")
    S0 = vp("Revenue / Sales")
    COGS = v("Cost of Sales")
    OPEx = v("Operating Expenses")
    NP = v("Net Profit")
    GP = v("Gross Profit")
    PS = v("Profit on Sales")
    EBIT = v("Operating Profit / EBIT")
    EBITDA = v("EBITDA")
    Dep = v("Depreciation")
    Fin = v("Financial Expenses / Interest")
    Extra = v("Extraordinary Items") or 0.0
    TA = v("Total Assets")
    FA = v("Fixed Assets")
    CA = v("Current Assets")
    Inv = v("Inventory")
    Rec = v("Receivables")
    Cash = v("Cash + Short-term Securities")
    TL = v("Total Liabilities")
    STL = v("Short-term Liabilities")
    LTL = v("Long-term Liabilities")
    Eq = v("Equity")
    SC = v("Share Capital") or 0.0
    RE = v("Retained Earnings")
    CC = v("Constant Capital")
    WC = v("Working Capital")
    GP3 = v("Three-year Gross Profit")

    def sub(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else a - b

    def add(*args: float | None) -> float | None:
        if any(x is None for x in args):
            return None
        return float(sum(args))

    def mul(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else a * b

    ratios: dict[str, float | None] = {
        "Attr1": _safe_div(NP, TA),
        "Attr2": _safe_div(TL, TA),
        "Attr3": _safe_div(WC, TA),
        "Attr4": _safe_div(CA, STL),
        "Attr5": _safe_div(add(Cash, Rec, -STL if STL is not None else None), sub(OPEx, Dep)) * 365 if _safe_div(add(Cash, Rec, -STL if STL is not None else None), sub(OPEx, Dep)) is not None else None,
        "Attr6": _safe_div(RE, TA),
        "Attr7": _safe_div(EBIT, TA),
        "Attr8": _safe_div(Eq, TL),
        "Attr9": _safe_div(S, TA),
        "Attr10": _safe_div(Eq, TA),
        "Attr11": _safe_div(add(GP, Extra, Fin), TA),
        "Attr12": _safe_div(GP, STL),
        "Attr13": _safe_div(EBITDA, S),
        "Attr14": _safe_div(add(GP, Fin), TA),
        "Attr15": _safe_div(mul(TL, 365), EBITDA),
        "Attr16": _safe_div(EBITDA, TL),
        "Attr17": _safe_div(TA, TL),
        "Attr18": _safe_div(GP, TA),
        "Attr19": _safe_div(GP, S),
        "Attr20": _safe_div(mul(Inv, 365), S),
        "Attr21": _safe_div(S, S0),
        "Attr22": _safe_div(EBIT, TA),
        "Attr23": _safe_div(NP, S),
        "Attr24": _safe_div(GP3, TA),
        "Attr25": _safe_div(sub(Eq, SC), TA),
        "Attr26": _safe_div(add(NP, Dep), TL),
        "Attr27": _safe_div(EBIT, Fin),
        "Attr28": _safe_div(WC, FA),
        "Attr29": _safe_log(TA),
        "Attr30": _safe_div(sub(TL, Cash), S),
        "Attr31": _safe_div(add(GP, Fin), S),
        "Attr32": _safe_div(mul(STL, 365), COGS),
        "Attr33": _safe_div(OPEx, STL),
        "Attr34": _safe_div(OPEx, TL),
        "Attr35": _safe_div(PS, TA),
        "Attr36": _safe_div(S, TA),
        "Attr37": _safe_div(sub(CA, Inv), LTL),
        "Attr38": _safe_div(CC, TA),
        "Attr39": _safe_div(PS, S),
        "Attr40": _safe_div(sub(sub(CA, Inv), Rec), STL),
        "Attr41": _safe_div(TL, mul(EBITDA, 12 / 365) if EBITDA is not None else None),
        "Attr42": _safe_div(EBIT, S),
        "Attr43": None,  # filled below as Attr44 + Attr20 where available
        "Attr44": _safe_div(mul(Rec, 365), S),
        "Attr45": _safe_div(NP, Inv),
        "Attr46": _safe_div(sub(CA, Inv), STL),
        "Attr47": _safe_div(mul(Inv, 365), COGS),
        "Attr48": _safe_div(sub(EBIT, Dep), TA),
        "Attr49": _safe_div(sub(EBIT, Dep), S),
        "Attr50": _safe_div(CA, TL),
        "Attr51": _safe_div(STL, TA),
        "Attr52": _safe_div(mul(STL, 365), COGS),
        "Attr53": _safe_div(Eq, FA),
        "Attr54": _safe_div(CC, FA),
        "Attr55": WC,
        "Attr56": _safe_div(sub(S, COGS), S),
        "Attr57": _safe_div(sub(sub(CA, Inv), STL), sub(sub(S, GP), Dep)),
        "Attr58": _safe_div(add(COGS, OPEx, Fin), S),
        "Attr59": _safe_div(LTL, Eq),
        "Attr60": _safe_div(S, Inv),
        "Attr61": _safe_div(S, Rec),
        "Attr62": _safe_div(mul(STL, 365), S),
        "Attr63": _safe_div(S, STL),
        "Attr64": _safe_div(S, FA),
    }
    if ratios["Attr44"] is not None and ratios["Attr20"] is not None:
        ratios["Attr43"] = ratios["Attr44"] + ratios["Attr20"]

    # Add warnings for blank ratios. The existing preprocessor can handle missing values,
    # but surfacing the issue helps analysts fix their inputs.
    blank_ratio_count = sum(1 for value in ratios.values() if value is None)
    if blank_ratio_count:
        warnings.append(f"{blank_ratio_count} Attr ratios could not be computed and will be treated as missing.")

    return ExcelParseResult(
        line_items=line_items,
        ratios=ratios,
        warnings=warnings,
        missing_required=missing_required,
    ).as_dict()


if __name__ == "__main__":
    import sys
    parsed = parse_financial_excel(sys.argv[1])
    print("Extracted line items:", len(parsed["line_items"]))
    print("Computed ratios:", len(parsed["ratios"]))
    print("Warnings:", parsed["warnings"])
    for k in list(parsed["ratios"])[:10]:
        print(k, parsed["ratios"][k])
