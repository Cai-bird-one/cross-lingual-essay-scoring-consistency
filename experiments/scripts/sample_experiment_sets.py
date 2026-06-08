#!/usr/bin/env python3
"""Create practical, stratified evaluation subsets from processed datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def sample_n(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed)


def make_merlin_balanced(processed: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(processed / "merlin_cefr.csv")
    # A2 and B1 are the only CEFR levels with enough examples in all three
    # MERLIN languages, so this subset is the cleanest cross-language comparison.
    keep_levels = ["A2", "B1"]
    groups = []
    for (language, label), part in df[df["gold_label"].isin(keep_levels)].groupby(
        ["language", "gold_label"]
    ):
        groups.append(sample_n(part, 150, seed))
    out = pd.concat(groups, ignore_index=True).sample(frac=1, random_state=seed)
    out.to_csv(processed / "merlin_cefr_balanced_a2_b1_900.csv", index=False)
    return out


def make_merlin_full_eval(processed: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(processed / "merlin_cefr.csv")
    groups = []
    for (language, label), part in df.groupby(["language", "gold_label"]):
        groups.append(sample_n(part, 200, seed))
    out = pd.concat(groups, ignore_index=True).sample(frac=1, random_state=seed)
    out.to_csv(processed / "merlin_cefr_eval_max200_per_lang_level.csv", index=False)
    return out


def score_band(score: float) -> str:
    if score <= 2.0:
        return "low_1_2"
    if score <= 3.0:
        return "mid_low_2_5_3"
    if score <= 4.0:
        return "mid_high_3_5_4"
    return "high_4_5_5"


def make_ellipse_stratified(processed: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(processed / "ellipse_aes.csv")
    df["score_band"] = df["gold_numeric"].apply(score_band)
    groups = []
    for _, part in df.groupby("score_band"):
        groups.append(sample_n(part, 150, seed))
    out = pd.concat(groups, ignore_index=True).sample(frac=1, random_state=seed)
    out.to_csv(processed / "ellipse_aes_stratified_600.csv", index=False)
    return out


def summarize(name: str, df: pd.DataFrame) -> dict:
    return {
        "name": name,
        "rows": int(len(df)),
        "languages": df["language"].value_counts().to_dict(),
        "labels": df["gold_label"].value_counts().sort_index().to_dict(),
        "language_by_label": (
            df.groupby(["language", "gold_label"]).size().unstack(fill_value=0).sort_index().to_dict()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[2],
        type=Path,
    )
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    root = args.project_root.resolve()
    processed = root / "experiments" / "data" / "processed"
    summaries_dir = root / "experiments" / "results" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "merlin_cefr_balanced_a2_b1_900": make_merlin_balanced(processed, args.seed),
        "merlin_cefr_eval_max200_per_lang_level": make_merlin_full_eval(processed, args.seed),
        "ellipse_aes_stratified_600": make_ellipse_stratified(processed, args.seed),
    }
    summary = {name: summarize(name, df) for name, df in outputs.items()}
    (summaries_dir / "experiment_subsets_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
