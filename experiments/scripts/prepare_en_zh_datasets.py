#!/usr/bin/env python3
"""Prepare English CEFR and Chinese essay-scoring datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


CEFR_TO_NUM = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
ENGLISH_CEFR_DATASETS = [
    "UniversalCEFR/icle500_en",
    "UniversalCEFR/cefr_asag_en",
]


def base_cefr(label: str) -> str | None:
    label = str(label).strip().upper()
    if label in {"", "NA", "NAN"}:
        return None
    if label.endswith("+"):
        label = label[:-1]
    return label if label in CEFR_TO_NUM else None


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "summaries": root / "results" / "summaries",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def prepare_english_cefr(dirs: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_id in ENGLISH_CEFR_DATASETS:
        short_name = dataset_id.split("/")[-1]
        ds = load_dataset(dataset_id)["train"]
        df = pd.DataFrame(ds)
        df.to_json(
            dirs["raw"] / f"{short_name}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        df = df[df["category"].eq("learner")].copy()
        df["gold_label_original"] = df["cefr_level"]
        df["gold_label"] = df["cefr_level"].map(base_cefr)
        df = df[df["gold_label"].notna()].copy()
        df["sample_id"] = [f"{short_name}_{i:05d}" for i in range(len(df))]
        df["dataset"] = dataset_id
        df["language"] = "en"
        df["label_type"] = "CEFR"
        df["gold_numeric"] = df["gold_label"].map(CEFR_TO_NUM).astype(int)
        df["prompt"] = ""
        df["split"] = "train"
        df["text_chars"] = df["text"].fillna("").str.len()
        df["text_words_space"] = df["text"].fillna("").str.split().str.len()
        df = df[df["text_chars"] >= 50].copy()
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
            "gold_label_original",
        ]
        frames.append(df[keep])

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(dirs["processed"] / "english_cefr_learner.csv", index=False)
    out.to_json(
        dirs["processed"] / "english_cefr_learner.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return out


def read_chinese_scores(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="gbk").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        essay_id, title, score = parts[0].strip(), parts[1].strip(), parts[2].strip()
        rows.append({"essay_id": essay_id, "title": title, "score": float(score)})
    return pd.DataFrame(rows)


def prepare_chinese_aes(project_root: Path, dirs: dict[str, Path]) -> pd.DataFrame:
    dataset_root = project_root / "experiments" / "data" / "raw" / "AES-Dataset"
    scores = read_chinese_scores(dataset_root / "scores.txt")

    rows = []
    for _, row in scores.iterrows():
        essay_id = row["essay_id"]
        essay_path = dataset_root / "essays" / f"{essay_id}.txt"
        text = essay_path.read_text(encoding="utf-8").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title_from_file = lines[0] if lines else row["title"]
        body = "\n".join(lines[1:]) if len(lines) > 1 else text
        rows.append(
            {
                "sample_id": f"chinese_aes_{essay_id}",
                "dataset": "AES-Dataset",
                "language": "zh",
                "text": body,
                "label_type": "SCORE_0_100",
                "gold_label": str(int(row["score"])),
                "gold_numeric": float(row["score"]),
                "prompt": title_from_file,
                "source_name": "declan-haojin/AES-Dataset",
                "license": "MIT",
                "split": "all",
                "format": "essay",
                "category": "learner",
                "title": row["title"],
                "text_chars": len(body),
                "text_words_space": len(body.split()),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(dirs["processed"] / "chinese_aes.csv", index=False)
    out.to_json(
        dirs["processed"] / "chinese_aes.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return out


def sample_english_cefr(df: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    groups = []
    for label, part in df.groupby("gold_label"):
        groups.append(part.sample(n=min(100, len(part)), random_state=42))
    out = pd.concat(groups, ignore_index=True).sample(frac=1, random_state=42)
    out.to_csv(dirs["processed"] / "english_cefr_eval_max100_per_level.csv", index=False)
    return out


def sample_chinese_aes(df: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    out = df.copy()
    out["score_band"] = pd.qcut(out["gold_numeric"], q=4, labels=False, duplicates="drop")
    groups = []
    for _, part in out.groupby("score_band"):
        groups.append(part.sample(n=min(50, len(part)), random_state=42))
    sampled = pd.concat(groups, ignore_index=True).sample(frac=1, random_state=42)
    sampled.to_csv(dirs["processed"] / "chinese_aes_stratified.csv", index=False)
    return sampled


def summarize(name: str, df: pd.DataFrame, dirs: dict[str, Path]) -> dict:
    summary = {
        "name": name,
        "rows": int(len(df)),
        "languages": df["language"].value_counts().to_dict(),
        "label_type": df["label_type"].value_counts().to_dict(),
        "gold_label_counts": df["gold_label"].astype(str).value_counts().sort_index().to_dict(),
        "gold_numeric": {
            "min": float(df["gold_numeric"].min()),
            "median": float(df["gold_numeric"].median()),
            "mean": float(round(df["gold_numeric"].mean(), 3)),
            "max": float(df["gold_numeric"].max()),
        },
        "text_chars": {
            "min": int(df["text_chars"].min()),
            "median": float(df["text_chars"].median()),
            "mean": float(round(df["text_chars"].mean(), 2)),
            "max": int(df["text_chars"].max()),
        },
    }
    (dirs["summaries"] / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def write_combined_cefr(project_root: Path, english: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame | None:
    merlin_path = project_root / "experiments" / "data" / "processed" / "merlin_cefr.csv"
    if not merlin_path.exists():
        return None
    merlin = pd.read_csv(merlin_path)
    combined = pd.concat([merlin, english], ignore_index=True, sort=False)
    combined.to_csv(dirs["processed"] / "cefr_multilingual_en_de_it_cs.csv", index=False)
    combined.to_json(
        dirs["processed"] / "cefr_multilingual_en_de_it_cs.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    groups = []
    for _, part in combined.groupby(["language", "gold_label"]):
        groups.append(part.sample(n=min(100, len(part)), random_state=42))
    sample = pd.concat(groups, ignore_index=True).sample(frac=1, random_state=42)
    sample.to_csv(
        dirs["processed"] / "cefr_multilingual_en_de_it_cs_eval_max100_per_lang_level.csv",
        index=False,
    )
    return combined


def write_combined_aes(project_root: Path, chinese: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame | None:
    ellipse_path = project_root / "experiments" / "data" / "processed" / "ellipse_aes.csv"
    if not ellipse_path.exists():
        return None
    ellipse = pd.read_csv(ellipse_path)
    combined = pd.concat([ellipse, chinese], ignore_index=True, sort=False)
    combined.to_csv(dirs["processed"] / "aes_english_chinese_mixed_scales.csv", index=False)
    combined.to_json(
        dirs["processed"] / "aes_english_chinese_mixed_scales.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return combined


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dirs = ensure_dirs(project_root / "experiments")

    english = prepare_english_cefr(dirs)
    chinese = prepare_chinese_aes(project_root, dirs)
    english_sample = sample_english_cefr(english, dirs)
    chinese_sample = sample_chinese_aes(chinese, dirs)
    combined_cefr = write_combined_cefr(project_root, english, dirs)
    combined_aes = write_combined_aes(project_root, chinese, dirs)

    summaries = {
        "english_cefr_learner": summarize("english_cefr_learner", english, dirs),
        "english_cefr_eval_max100_per_level": summarize(
            "english_cefr_eval_max100_per_level", english_sample, dirs
        ),
        "chinese_aes": summarize("chinese_aes", chinese, dirs),
        "chinese_aes_stratified": summarize("chinese_aes_stratified", chinese_sample, dirs),
    }
    if combined_cefr is not None:
        summaries["cefr_multilingual_en_de_it_cs"] = summarize(
            "cefr_multilingual_en_de_it_cs", combined_cefr, dirs
        )
    if combined_aes is not None:
        summaries["aes_english_chinese_mixed_scales"] = summarize(
            "aes_english_chinese_mixed_scales", combined_aes, dirs
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
