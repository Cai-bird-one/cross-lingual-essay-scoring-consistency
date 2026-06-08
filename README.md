# 跨语言作文评分一致性实验代码

本仓库仅保存实验代码与运行所需的提示词模板，不包含论文正文、PDF、图表图片、原始数据、处理后数据、模型预测结果或评测结果。

代码用于通过 OpenAI-compatible API 运行跨语言作文评分一致性实验，支持 CEFR 等级判断任务和作文分数预测任务。

## 功能

- 下载并标准化多语言作文数据；
- 构造平衡实验样本；
- 使用大语言模型进行 CEFR 等级判断和作文评分；
- 计算 Accuracy、MAE、RMSE、Bias、Pearson、Spearman 和 QWK 等指标；
- 汇总多个模型的评测结果。

## 目录结构

```text
.
├── src/
│   ├── prompts/       # 评分任务提示词模板
│   └── scripts/       # 数据处理、模型评分、评测和结果汇总脚本
├── scripts/
│   └── run_full_experiment.sh
├── requirements.txt
└── README.md
```

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API 配置

模型评分脚本会从命令行参数或环境变量读取 API 配置。

环境变量示例：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_base_url"
```

## 常用命令

### 一键运行完整流程

```bash
bash scripts/run_full_experiment.sh
```

这个脚本会按顺序执行数据准备、抽样、模型评分、指标计算和模型结果汇总。若没有设置 API key，脚本会完成数据准备和抽样，然后在评分前停止。

常用环境变量：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_base_url"

CEFR_MODELS="gpt-4o-mini qwen-plus" \
AES100_MODELS="gpt-4o-mini qwen-plus" \
bash scripts/run_full_experiment.sh
```

- `CEFR_MODELS`：用于 CEFR 等级判断任务的模型列表，默认是 `gpt-4o-mini`。
- `AES100_MODELS`：用于中文 0--100 分作文评分任务的模型列表，默认是 `gpt-4o-mini`。
- `LIMIT`：每个评分输入最多处理多少条样本；默认为 `0`，表示不限制。调试时可设为 `LIMIT=5`。
- `RUN_SCORING`：设为 `0` 时只准备数据和抽样，不调用模型 API；默认 `auto`，无 API key 时自动跳过评分。

示例：只跑前 5 条样本检查流程。

```bash
LIMIT=5 bash scripts/run_full_experiment.sh
```

### 分步运行

以下命令适合调试或只复现部分流程。

#### 1. 准备 MERLIN 与可选 ELLIPSE 数据

```bash
python3 src/scripts/prepare_datasets.py
```

作用：

- 从 Hugging Face 下载并标准化 MERLIN 德语、意大利语、捷克语 CEFR 数据；
- 若仓库根目录下存在本地 `ELLIPSE-Corpus/`，则额外整理英文作文评分数据；
- 输出标准化 CSV/JSONL 到 `src/data/processed/`；
- 输出数据摘要到 `src/results/summaries/`。

#### 2. 准备英文 CEFR 与中文 AES 数据

```bash
python3 src/scripts/prepare_en_zh_datasets.py
```

作用：

- 从 Hugging Face 下载并整理英文 CEFR 学习者写作数据；
- 若本地存在 `src/data/raw/AES-Dataset/`，则整理中文 0--100 分作文评分数据；
- 生成英文 CEFR 抽样集、中文 AES 分层抽样集，以及可用的合并数据；
- 缺少中文 AES 本地数据时会跳过中文部分。

#### 3. 构造实验抽样集

```bash
python3 src/scripts/sample_experiment_sets.py
```

作用：

- 基于 `src/data/processed/` 中的标准化数据构造主实验输入；
- 生成 MERLIN A2/B1 平衡集、MERLIN 多等级评测集和 ELLIPSE 分层抽样集；
- 输出抽样文件到 `src/data/processed/`；
- 输出抽样摘要到 `src/results/summaries/`。

#### 4. 调用模型进行 CEFR 评分

```bash
python3 src/scripts/score_with_llm.py \
  --input src/data/processed/cefr_multilingual_en_de_it_cs_eval_max100_per_lang_level.csv \
  --task cefr \
  --model gpt-4o-mini \
  --output src/results/predictions/cefr4lang_balanced_gpt4o_mini.csv
```

作用：

- 读取标准化作文样本；
- 将每条文本填入 `src/prompts/cefr_scoring_prompt.md`；
- 调用指定模型输出 CEFR 标签；
- 将原始响应、预测标签和数值化预测保存到 `src/results/predictions/`。

调试时可以先渲染首条 prompt，不调用 API：

```bash
python3 src/scripts/score_with_llm.py \
  --input src/data/processed/cefr_multilingual_en_de_it_cs_eval_max100_per_lang_level.csv \
  --task cefr \
  --model gpt-4o-mini \
  --dry-run
```

中文 0--100 分作文评分使用 `aes100` 任务：

```bash
python3 src/scripts/score_with_llm.py \
  --input src/data/processed/chinese_aes_stratified.csv \
  --task aes100 \
  --model gpt-4o-mini \
  --output src/results/predictions/chinese_aes_gpt4o_mini.csv
```

#### 5. 评测预测结果

```bash
python3 src/scripts/evaluate_predictions.py \
  --predictions src/results/predictions/cefr4lang_balanced_gpt4o_mini.csv \
  --task cefr \
  --output-dir src/results/metrics/cefr4lang
```

作用：

- 读取模型预测文件；
- 过滤空预测和解析失败样本；
- 计算总体、分语言、分模型、模型-语言交叉维度的指标；
- 输出指标 CSV、诊断 JSON 和清洗后的预测文件到 `src/results/metrics/`。

#### 6. 汇总多个模型结果

```bash
python3 src/scripts/build_model_comparison.py \
  --metrics src/results/metrics/cefr4lang/*_metrics.csv \
  --output src/results/metrics/cefr4lang/cefr4lang_model_comparison.csv
```

作用：

- 读取多个 `*_metrics.csv` 文件；
- 提取每个模型的 overall 指标；
- 按 QWK、Accuracy 和 MAE 排序；
- 输出紧凑的模型对比表。
