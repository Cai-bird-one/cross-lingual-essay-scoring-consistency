# 跨语言作文评分一致性实验代码

本仓库仅保存实验代码与运行所需的提示词模板，不包含论文正文、PDF、图表图片、原始数据、处理后数据、模型预测结果或评测结果。

代码用于通过 OpenAI-compatible API 运行跨语言作文评分一致性实验，支持 CEFR 等级判断任务和作文分数预测任务。

## 功能

- 下载并标准化多语言作文数据；
- 构造平衡实验样本；
- 使用大语言模型进行 CEFR 等级判断和作文评分；
- 计算 Accuracy、MAE、RMSE、Bias、Pearson、Spearman 和 QWK 等指标；
- 汇总多个模型的评测结果；
- 根据汇总数据重新生成论文图表。

## 目录结构

```text
.
├── experiments/
│   ├── prompts/       # 评分任务提示词模板
│   └── scripts/       # 数据处理、模型评分、评测和结果汇总脚本
├── generate_paper_charts.py
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

模型评分脚本会从命令行参数、环境变量或本地 `experiments/api` 文件读取 API 配置。

环境变量示例：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_base_url"
```

本地文件示例：

```text
api_key=your_api_key
url=https://example.com/v1
```

`experiments/api` 已被 `.gitignore` 忽略，不会上传到仓库。

## 常用命令

准备数据：

```bash
python3 experiments/scripts/prepare_datasets.py
python3 experiments/scripts/prepare_en_zh_datasets.py
python3 experiments/scripts/sample_experiment_sets.py
```

运行 CEFR 评分：

```bash
python3 experiments/scripts/score_with_llm.py \
  --input experiments/data/processed/cefr_en_de_it_cs_balanced_a2_b1_400.csv \
  --task cefr \
  --model gpt-4o-mini \
  --output experiments/results/predictions/cefr4lang_balanced_gpt4o_mini.csv
```

评测预测结果：

```bash
python3 experiments/scripts/evaluate_predictions.py \
  --predictions experiments/results/predictions/cefr4lang_balanced_gpt4o_mini.csv \
  --task cefr \
  --output-dir experiments/results/metrics/cefr4lang
```

汇总模型结果：

```bash
python3 experiments/scripts/build_model_comparison.py \
  --input-dir experiments/results/metrics/cefr4lang \
  --output experiments/results/metrics/cefr4lang/cefr4lang_model_comparison.csv
```
