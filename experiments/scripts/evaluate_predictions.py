#!/usr/bin/env python3
"""Evaluate model predictions for CEFR and essay-scoring experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


CEFR_TO_NUM = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(rank_average(x), rank_average(y))


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if len(labels) <= 1:
        return float("nan")
    index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    observed = np.zeros((n, n), dtype=float)
    for true, pred in zip(y_true, y_pred):
        observed[index[true], index[pred]] += 1

    true_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    expected = np.outer(true_hist, pred_hist) / max(observed.sum(), 1)

    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            weights[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

    numerator = (weights * observed).sum()
    denominator = (weights * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1 - numerator / denominator)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else float("nan")


def normalize_predictions(df: pd.DataFrame, task: str) -> pd.DataFrame:
    out = df.copy()
    out = out[out["error"].fillna("") == ""].copy()
    out = out[pd.to_numeric(out["pred_numeric"], errors="coerce").notna()].copy()
    out["gold_numeric"] = pd.to_numeric(out["gold_numeric"], errors="coerce")
    out["pred_numeric"] = pd.to_numeric(out["pred_numeric"], errors="coerce")
    out = out[out["gold_numeric"].notna() & out["pred_numeric"].notna()].copy()
    if task == "cefr":
        out["gold_class"] = out["gold_numeric"].astype(int)
        out["pred_class"] = out["pred_numeric"].round().astype(int)
    elif task == "aes100":
        out["gold_class"] = out["gold_numeric"].round().astype(int)
        out["pred_class"] = out["pred_numeric"].round().astype(int)
    else:
        out["gold_class"] = (out["gold_numeric"] * 2).round().astype(int)
        out["pred_class"] = (out["pred_numeric"] * 2).round().astype(int)
    out["diff"] = out["pred_numeric"] - out["gold_numeric"]
    out["abs_diff"] = out["diff"].abs()
    return out


def metrics_for(group: pd.DataFrame) -> dict:
    y_true = group["gold_numeric"].to_numpy(dtype=float)
    y_pred = group["pred_numeric"].to_numpy(dtype=float)
    true_class = group["gold_class"].to_numpy()
    pred_class = group["pred_class"].to_numpy()
    exact = float(np.mean(true_class == pred_class)) if len(group) else float("nan")
    within_1 = float(np.mean(np.abs(y_true - y_pred) <= 1.0)) if len(group) else float("nan")
    return {
        "n": int(len(group)),
        "accuracy_exact": exact,
        "within_1_level_or_point": within_1,
        "macro_f1": macro_f1(true_class, pred_class),
        "mae": float(np.mean(np.abs(y_true - y_pred))) if len(group) else float("nan"),
        "rmse": float(math.sqrt(np.mean((y_true - y_pred) ** 2))) if len(group) else float("nan"),
        "mean_bias_pred_minus_gold": float(np.mean(y_pred - y_true)) if len(group) else float("nan"),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "qwk": qwk(true_class, pred_class),
    }


def build_report(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = [{"group": "overall", **metrics_for(df)}]
    for language, part in df.groupby("language"):
        rows.append({"group": f"language={language}", **metrics_for(part)})
    for model, part in df.groupby("model"):
        rows.append({"group": f"model={model}", **metrics_for(part)})
    for (model, language), part in df.groupby(["model", "language"]):
        rows.append({"group": f"model={model}|language={language}", **metrics_for(part)})

    report = pd.DataFrame(rows)
    diagnostics = {
        "rows_evaluated": int(len(df)),
        "error_free_rows": int(len(df)),
        "languages": df["language"].value_counts().to_dict(),
        "models": df["model"].value_counts().to_dict(),
        "gold_labels": df["gold_label"].value_counts().sort_index().to_dict(),
        "pred_labels": df["pred_label"].value_counts().sort_index().to_dict(),
    }
    return report, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["cefr", "aes", "aes100"])
    parser.add_argument("--output-dir", default=Path("experiments/results/metrics"), type=Path)
    args = parser.parse_args()

    if not args.predictions.exists():
        raise SystemExit(
            f"Prediction file not found: {args.predictions}\n"
            "Run score_with_llm.py first, or check that the model-name suffix in the path matches the scoring output."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.predictions)
    df = normalize_predictions(raw, args.task)
    report, diagnostics = build_report(df)

    stem = args.predictions.stem
    report_path = args.output_dir / f"{stem}_metrics.csv"
    json_path = args.output_dir / f"{stem}_diagnostics.json"
    clean_path = args.output_dir / f"{stem}_clean_predictions.csv"

    report.to_csv(report_path, index=False)
    df.to_csv(clean_path, index=False)
    json_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report.to_string(index=False))
    print(f"\nWrote: {report_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {clean_path}")


if __name__ == "__main__":
    main()
