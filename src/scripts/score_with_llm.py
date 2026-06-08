#!/usr/bin/env python3
"""Score standardized experiment CSV files with an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


CEFR_TO_NUM = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
VALID_CEFR = set(CEFR_TO_NUM)
VALID_AES_SCORES = {x / 2 for x in range(2, 11)}


def get_client(args: argparse.Namespace) -> OpenAI:
    api_key = (
        args.api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("AIHUBMIX_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    base_url = (
        args.base_url
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("AIHUBMIX_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
    )
    if not api_key:
        raise SystemExit(
            "Missing API key. Set OPENAI_API_KEY/AIHUBMIX_API_KEY, pass --api-key, "
            "or pass a provider-specific environment variable."
        )
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def load_prompt(task: str, project_root: Path, prompt_file: Path | None = None) -> str:
    if prompt_file is not None:
        path = prompt_file
        if not path.is_absolute():
            path = project_root / path
        return path.read_text(encoding="utf-8")
    if task == "cefr":
        name = "cefr_scoring_prompt.md"
    elif task == "aes100":
        name = "aes_100_scoring_prompt.md"
    else:
        name = "aes_scoring_prompt.md"
    path = project_root / "src" / "prompts" / name
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, row: pd.Series, max_chars: int) -> str:
    text = str(row.get("text", ""))
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
    prompt = str(row.get("prompt", ""))
    return template.replace("{text}", text).replace("{prompt}", prompt)


def extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_prediction(task: str, payload: dict[str, Any]) -> tuple[str, float | None, float | None]:
    if task == "cefr":
        value = str(
            payload.get("predicted_label")
            or payload.get("cefr")
            or payload.get("level")
            or payload.get("label")
            or ""
        ).upper()
        match = re.search(r"\b(A1|A2|B1|B2|C1|C2)\b", value)
        label = match.group(1) if match else value
        numeric = float(CEFR_TO_NUM[label]) if label in CEFR_TO_NUM else None
    elif task == "aes100":
        value = payload.get("predicted_score", payload.get("score", payload.get("overall_score")))
        score = round(float(value))
        score = min(100, max(0, score))
        label = str(score)
        numeric = float(score)
    else:
        value = payload.get("predicted_score", payload.get("score", payload.get("overall_score")))
        score = float(value)
        score = round(score * 2) / 2
        score = min(5.0, max(1.0, score))
        label = f"{score:.1f}"
        numeric = score
    confidence = payload.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    return label, numeric, confidence


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float | None,
    max_retries: int,
    retry_sleep: float,
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a careful evaluator. Return only valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - keep script robust across providers
            last_error = exc
            message = str(exc).lower()
            if "response_format" in message or "json_object" in message:
                try:
                    fallback_kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                    }
                    if temperature is not None:
                        fallback_kwargs["temperature"] = temperature
                    response = client.chat.completions.create(**fallback_kwargs)
                    return response.choices[0].message.content or ""
                except Exception as fallback_exc:  # noqa: BLE001
                    last_error = fallback_exc
            if temperature is not None and "temperature" in message:
                temperature = None
                continue
            if attempt < max_retries:
                time.sleep(retry_sleep * (attempt + 1))
                continue
    raise RuntimeError(f"Model call failed after retries: {last_error}") from last_error


def read_existing(output: Path) -> set[str]:
    if not output.exists():
        return set()
    try:
        df = pd.read_csv(output, usecols=["sample_id"])
    except Exception:
        return set()
    return set(df["sample_id"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", required=True, choices=["cefr", "aes", "aes100"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--prompt-file",
        default=None,
        type=Path,
        help="Optional prompt template path. Relative paths are resolved from project root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the first prompt and exit without calling the API.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[2], type=Path)
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")

    project_root = args.project_root.resolve()
    template = load_prompt(args.task, project_root, args.prompt_file)

    df = pd.read_csv(args.input)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    if args.dry_run:
        if df.empty:
            raise SystemExit("Input has no rows.")
        print(render_prompt(template, df.iloc[0], args.max_chars))
        return

    client = get_client(args)

    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = read_existing(args.output) if args.resume else set()
    write_header = not args.output.exists() or not args.resume

    columns = [
        "sample_id",
        "dataset",
        "language",
        "label_type",
        "gold_label",
        "gold_numeric",
        "model",
        "task",
        "pred_label",
        "pred_numeric",
        "confidence",
        "raw_response",
        "error",
    ]

    with args.output.open("a" if args.resume else "w", encoding="utf-8", newline="") as f:
        if write_header:
            f.write(",".join(columns) + "\n")

        for idx, row in df.iterrows():
            sample_id = str(row["sample_id"])
            if sample_id in done:
                continue

            prompt = render_prompt(template, row, args.max_chars)
            raw_response = ""
            pred_label = ""
            pred_numeric: float | None = None
            confidence: float | None = None
            error = ""

            try:
                raw_response = call_model(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                )
                payload = extract_json(raw_response)
                pred_label, pred_numeric, confidence = normalize_prediction(args.task, payload)
            except Exception as exc:  # noqa: BLE001
                error = str(exc).replace("\n", " ")[:500]

            record = {
                "sample_id": sample_id,
                "dataset": row.get("dataset", ""),
                "language": row.get("language", ""),
                "label_type": row.get("label_type", ""),
                "gold_label": row.get("gold_label", ""),
                "gold_numeric": row.get("gold_numeric", ""),
                "model": args.model,
                "task": args.task,
                "pred_label": pred_label,
                "pred_numeric": pred_numeric if pred_numeric is not None else "",
                "confidence": confidence if confidence is not None else "",
                "raw_response": raw_response,
                "error": error,
            }
            pd.DataFrame([record], columns=columns).to_csv(f, header=False, index=False)
            f.flush()

            if (idx + 1) % 10 == 0:
                print(f"processed {idx + 1}/{len(df)}", file=sys.stderr)


if __name__ == "__main__":
    main()
