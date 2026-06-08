#!/usr/bin/env python3
"""Download and standardize datasets for the cross-lingual scoring experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
from datasets import load_dataset


CEFR_TO_NUM = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
MERLIN_DATASETS = {
    "merlin_de": "UniversalCEFR/merlin_de",
    "merlin_it": "UniversalCEFR/merlin_it",
    "merlin_cs": "UniversalCEFR/merlin_cs",
}


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "summaries": root / "results" / "summaries",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_merlin(root: Path, dirs: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for short_name, dataset_id in MERLIN_DATASETS.items():
        dataset = load_dataset(dataset_id)["train"]
        rows = [dict(row) for row in dataset]
        write_jsonl(dirs["raw"] / f"{short_name}.jsonl", rows)

        df = pd.DataFrame(rows)
        df = df[df["cefr_level"].isin(CEFR_TO_NUM)].copy()
        df["sample_id"] = [f"{short_name}_{i:05d}" for i in range(len(df))]
        df["dataset"] = dataset_id
        df["language"] = df["lang"]
        df["label_type"] = "CEFR"
        df["gold_label"] = df["cefr_level"]
        df["gold_numeric"] = df["cefr_level"].map(CEFR_TO_NUM).astype(int)
        df["prompt"] = ""
        df["split"] = "train"
        df["text_chars"] = df["text"].fillna("").str.len()
        df["text_words_space"] = df["text"].fillna("").str.split().str.len()
        keep = [
            "sample_id",
            "dataset",
            "language",
            "text",
            "label_type",
            "gold_label",
            "gold_numeric",
            "prompt",
            "source_name",
            "license",
            "split",
            "format",
            "category",
            "title",
            "text_chars",
            "text_words_space",
        ]
        frames.append(df[keep])

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(dirs["processed"] / "merlin_cefr.csv", index=False)
    out.to_json(
        dirs["processed"] / "merlin_cefr.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return out


def ensure_ellipse_test_extracted(project_root: Path) -> Path | None:
    test_csv = project_root / "ELLIPSE-Corpus" / "extracted" / "test" / "ELLIPSE_Final_github_test.csv"
    if test_csv.exists():
        return test_csv

    zip_path = project_root / "ELLIPSE-Corpus" / "ELLIPSE_Final_github_test.zip"
    if not zip_path.exists():
        return None

    test_csv.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(test_csv.parent, pwd=b"ellipse_test")
    return test_csv if test_csv.exists() else None


def normalize_ellipse(project_root: Path, dirs: dict[str, Path]) -> pd.DataFrame:
    train_csv = project_root / "ELLIPSE-Corpus" / "ELLIPSE_Final_github_train.csv"
    test_csv = ensure_ellipse_test_extracted(project_root)
    if not train_csv.exists():
        print(f"Skipping ELLIPSE: {train_csv} not found.")
        return pd.DataFrame()

    paths: list[Path] = [train_csv]
    if test_csv is not None:
        paths.append(test_csv)

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        split = "test" if "test" in path.name else "train"
        normalized = pd.DataFrame(
            {
                "sample_id": "ellipse_" + df["text_id_kaggle"].astype(str),
                "dataset": "ELLIPSE-Corpus",
                "language": "en",
                "text": df["full_text"].fillna(""),
                "label_type": "SCORE_1_5",
                "gold_label": df["Overall"].astype(str),
                "gold_numeric": df["Overall"].astype(float),
                "prompt": df["prompt"].fillna(""),
                "source_name": "ELLIPSE-Corpus",
                "license": "CC BY-NC-SA 4.0",
                "split": df.get("set", pd.Series([split] * len(df))).fillna(split),
                "format": "essay",
                "category": "learner",
                "title": df["prompt"].fillna(""),
            }
        )
        normalized["text_chars"] = normalized["text"].str.len()
        normalized["text_words_space"] = normalized["text"].str.split().str.len()
        frames.append(normalized)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(dirs["processed"] / "ellipse_aes.csv", index=False)
    out.to_json(
        dirs["processed"] / "ellipse_aes.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return out


def write_summary(name: str, df: pd.DataFrame, dirs: dict[str, Path]) -> dict:
    if df.empty:
        summary = {"name": name, "rows": 0}
        path = dirs["summaries"] / f"{name}_summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    summary = {
        "name": name,
        "rows": int(len(df)),
        "languages": df["language"].value_counts().to_dict(),
        "label_type": df["label_type"].value_counts().to_dict(),
        "gold_label_counts": df["gold_label"].value_counts().sort_index().to_dict(),
        "by_language_label": (
            df.groupby(["language", "gold_label"]).size().unstack(fill_value=0).sort_index().to_dict()
        ),
        "text_chars": {
            "min": int(df["text_chars"].min()),
            "median": float(df["text_chars"].median()),
            "mean": float(round(df["text_chars"].mean(), 2)),
            "max": int(df["text_chars"].max()),
        },
    }
    path = dirs["summaries"] / f"{name}_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_combined(merlin: pd.DataFrame, ellipse: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    frames = [df for df in [merlin, ellipse] if not df.empty]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(dirs["processed"] / "all_standardized.csv", index=False)
    combined.to_json(
        dirs["processed"] / "all_standardized.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[2],
        type=Path,
        help="Repository/workspace root.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    src_root = project_root / "src"
    dirs = ensure_dirs(src_root)

    merlin = normalize_merlin(src_root, dirs)
    ellipse = normalize_ellipse(project_root, dirs)
    combined = write_combined(merlin, ellipse, dirs)

    summaries = {
        "merlin_cefr": write_summary("merlin_cefr", merlin, dirs),
        "all_standardized": write_summary("all_standardized", combined, dirs),
    }
    if not ellipse.empty:
        summaries["ellipse_aes"] = write_summary("ellipse_aes", ellipse, dirs)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
