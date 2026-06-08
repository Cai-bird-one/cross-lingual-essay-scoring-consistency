#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CEFR_MODELS="${CEFR_MODELS:-gpt-4o-mini}"
AES100_MODELS="${AES100_MODELS:-gpt-4o-mini}"
LIMIT="${LIMIT:-0}"
RUN_SCORING="${RUN_SCORING:-auto}"

has_api_key() {
  [[ -n "${OPENAI_API_KEY:-}" || -n "${AIHUBMIX_API_KEY:-}" || -n "${DASHSCOPE_API_KEY:-}" ]]
}

safe_name() {
  local value="$1"
  value="${value//[^[:alnum:]_]/_}"
  printf '%s' "$value"
}

score_dataset() {
  local input="$1"
  local task="$2"
  local model="$3"
  local prefix="$4"
  local metrics_dir="$5"
  local model_safe
  local output

  model_safe="$(safe_name "$model")"
  output="src/results/predictions/${prefix}_${model_safe}.csv"

  mkdir -p "$(dirname "$output")" "$metrics_dir"
  echo "[score] task=${task} model=${model} input=${input}"

  local limit_args=()
  if [[ "$LIMIT" != "0" ]]; then
    limit_args=(--limit "$LIMIT")
  fi

  python3 src/scripts/score_with_llm.py \
    --input "$input" \
    --task "$task" \
    --model "$model" \
    --output "$output" \
    --resume \
    "${limit_args[@]}"

  echo "[evaluate] $output"
  python3 src/scripts/evaluate_predictions.py \
    --predictions "$output" \
    --task "$task" \
    --output-dir "$metrics_dir"
}

build_comparison_if_possible() {
  local metrics_dir="$1"
  local output="$2"
  shopt -s nullglob
  local metrics=("$metrics_dir"/*_metrics.csv)
  shopt -u nullglob

  if (( ${#metrics[@]} == 0 )); then
    echo "[skip] no metrics found in $metrics_dir"
    return
  fi

  echo "[compare] $output"
  python3 src/scripts/build_model_comparison.py \
    --metrics "${metrics[@]}" \
    --output "$output"
}

echo "[step 1/5] Prepare MERLIN and optional ELLIPSE datasets"
python3 src/scripts/prepare_datasets.py

echo "[step 2/5] Prepare English CEFR and optional Chinese AES datasets"
python3 src/scripts/prepare_en_zh_datasets.py

echo "[step 3/5] Build evaluation subsets"
python3 src/scripts/sample_experiment_sets.py

if [[ "$RUN_SCORING" == "0" ]]; then
  echo "[done] RUN_SCORING=0, stopped after data preparation."
  exit 0
fi

if [[ "$RUN_SCORING" == "auto" ]] && ! has_api_key; then
  echo "[done] No API key found, stopped before model scoring."
  echo "       Set OPENAI_API_KEY, AIHUBMIX_API_KEY, or DASHSCOPE_API_KEY to run scoring."
  exit 0
fi

if ! has_api_key; then
  echo "[error] Model scoring requires OPENAI_API_KEY, AIHUBMIX_API_KEY, or DASHSCOPE_API_KEY." >&2
  exit 1
fi

echo "[step 4/5] Score datasets and evaluate predictions"

CEFR_INPUT="src/data/processed/cefr_multilingual_en_de_it_cs_eval_max100_per_lang_level.csv"
if [[ ! -f "$CEFR_INPUT" ]]; then
  CEFR_INPUT="src/data/processed/merlin_cefr_balanced_a2_b1_900.csv"
fi

if [[ -f "$CEFR_INPUT" ]]; then
  for model in $CEFR_MODELS; do
    score_dataset "$CEFR_INPUT" "cefr" "$model" "cefr" "src/results/metrics/cefr"
  done
else
  echo "[skip] no CEFR evaluation input found"
fi

AES100_INPUT="src/data/processed/chinese_aes_stratified.csv"
if [[ -f "$AES100_INPUT" ]]; then
  for model in $AES100_MODELS; do
    score_dataset "$AES100_INPUT" "aes100" "$model" "chinese_aes" "src/results/metrics/chinese_aes"
  done
else
  echo "[skip] no Chinese AES evaluation input found"
fi

echo "[step 5/5] Build comparison tables"
build_comparison_if_possible "src/results/metrics/cefr" "src/results/metrics/cefr/model_comparison.csv"
build_comparison_if_possible "src/results/metrics/chinese_aes" "src/results/metrics/chinese_aes/model_comparison.csv"

echo "[done] Full experiment pipeline finished."
