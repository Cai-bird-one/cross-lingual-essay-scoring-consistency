#!/usr/bin/env python3
"""Build a compact model-comparison CSV from metric report files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def model_name_from_report(report: pd.DataFrame, fallback: str) -> str:
    model_rows = report[
        report["group"].astype(str).str.startswith("model=")
        & ~report["group"].astype(str).str.contains("language=", regex=False)
    ]
    if model_rows.empty:
        return fallback
    return str(model_rows.iloc[0]["group"]).replace("model=", "", 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--sort-by",
        default="qwk,accuracy_exact,-mae",
        help=(
            "Comma-separated sort keys. Prefix with '-' for ascending order. "
            "Default sorts by qwk desc, accuracy desc, mae asc."
        ),
    )
    args = parser.parse_args()

    rows = []
    for path in args.metrics:
        report = pd.read_csv(path)
        overall = report[report["group"].eq("overall")]
        if overall.empty:
            raise SystemExit(f"No overall row found in {path}")
        row = overall.iloc[0].to_dict()
        row["model"] = model_name_from_report(report, path.stem)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    columns = [
        "model",
        "n",
        "accuracy_exact",
        "within_1_level_or_point",
        "macro_f1",
        "mae",
        "rmse",
        "mean_bias_pred_minus_gold",
        "pearson",
        "spearman",
        "qwk",
    ]
    comparison = comparison[[column for column in columns if column in comparison.columns]]

    sort_cols: list[str] = []
    ascending: list[bool] = []
    for raw_key in args.sort_by.split(","):
        key = raw_key.strip()
        if not key:
            continue
        if key.startswith("-"):
            sort_cols.append(key[1:])
            ascending.append(True)
        else:
            sort_cols.append(key)
            ascending.append(False)
    if sort_cols:
        comparison = comparison.sort_values(sort_cols, ascending=ascending)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)
    print(comparison.to_string(index=False))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
