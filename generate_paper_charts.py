import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paper-charts")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


OUT = Path("paper_figures")
OUT.mkdir(exist_ok=True)

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font_manager.fontManager.addfont(FONT)
font_manager.fontManager.addfont(FONT_BOLD)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

COLORS = {
    "blue": "#3B82F6",
    "teal": "#14B8A6",
    "green": "#22C55E",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "purple": "#8B5CF6",
    "slate": "#334155",
    "gray": "#94A3B8",
}


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", dpi=220)
    plt.close(fig)


def annotate_bars(ax, bars, fmt="{:.3f}", y_offset=0.01):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )


def draw_heatmap(ax, data, title, cmap, vmin=None, vmax=None, center=None, fmt=".3f"):
    arr = np.array(data.values, dtype=float)
    if center is not None:
        norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
        im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    else:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_title(title, fontsize=11, pad=10, color="#0F172A", weight="bold")
    ax.tick_params(length=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            color = "white" if abs(val) > 0.45 else "#0F172A"
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7.5, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def cefr_distribution():
    df = pd.DataFrame(
        {
            "A1": [1, 57, 16, 29],
            "A2": [188, 306, 55, 381],
            "B1": [165, 331, 213, 394],
            "B2": [81, 293, 283, 2],
            "C1": [4, 42, 119, 0],
            "C2": [0, 4, 69, 0],
        },
        index=["捷克语 cs", "德语 de", "英语 en", "意大利语 it"],
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bottom = np.zeros(len(df))
    palette = ["#DBEAFE", "#93C5FD", "#60A5FA", "#2DD4BF", "#A78BFA", "#F59E0B"]
    for level, color in zip(df.columns, palette):
        ax.bar(df.index, df[level], bottom=bottom, label=level, color=color, edgecolor="white", linewidth=0.8)
        bottom += df[level].values
    ax.set_title("图1  四语言 CEFR 原始标签分布", fontsize=14, weight="bold", color="#0F172A", pad=16)
    ax.set_ylabel("文本数量", fontsize=10, color="#475569")
    ax.legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
    clean_axes(ax)
    for i, total in enumerate(bottom):
        ax.text(i, total + 20, f"{int(total):,}", ha="center", va="bottom", fontsize=9, color="#334155")
    save(fig, "figure1_cefr_label_distribution.png")


def cefr_overall():
    df = pd.DataFrame(
        {
            "Accuracy": [0.730, 0.708, 0.698, 0.595],
            "MAE": [0.273, 0.293, 0.303, 0.415],
            "Bias": [-0.023, 0.078, 0.078, 0.260],
            "Spearman": [0.528, 0.514, 0.495, 0.513],
            "QWK": [0.513, 0.500, 0.483, 0.446],
        },
        index=["gpt-5", "qwen3-max", "gpt-4o-mini", "qwen-plus"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(df.index))
    bars = axes[0].bar(x, df["Accuracy"], color=[COLORS["blue"], COLORS["teal"], COLORS["purple"], COLORS["amber"]])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df.index, rotation=18, ha="right")
    axes[0].set_ylim(0, 0.85)
    axes[0].set_ylabel("Accuracy", color="#475569")
    axes[0].set_title("CEFR 总体准确率", fontsize=12, weight="bold", color="#0F172A")
    annotate_bars(axes[0], bars, "{:.3f}", 0.012)
    clean_axes(axes[0])

    cmap = LinearSegmentedColormap.from_list("score", ["#EFF6FF", "#60A5FA", "#1E3A8A"])
    draw_heatmap(axes[1], df[["MAE", "Bias", "Spearman", "QWK"]], "误差与一致性指标", cmap, vmin=-0.05, vmax=0.55)
    fig.suptitle("图2  四语言 CEFR 主实验总体表现", fontsize=14, weight="bold", color="#0F172A", y=1.03)
    save(fig, "figure2_cefr_overall_metrics.png")


def cefr_by_language():
    models = {
        "gpt-5": [[0.760, 0.240, 0.180, 0.520], [0.560, 0.450, -0.450, 0.329], [0.820, 0.180, 0.120, 0.654], [0.780, 0.220, 0.060, 0.607]],
        "gpt-4o-mini": [[0.830, 0.170, 0.110, 0.673], [0.660, 0.340, -0.320, 0.320], [0.580, 0.420, 0.380, 0.462], [0.720, 0.280, 0.140, 0.481]],
        "qwen3-max": [[0.750, 0.250, 0.210, 0.554], [0.630, 0.370, -0.370, 0.288], [0.710, 0.290, 0.250, 0.586], [0.740, 0.260, 0.220, 0.536]],
        "qwen-plus": [[0.420, 0.620, 0.600, 0.327], [0.750, 0.250, -0.210, 0.500], [0.630, 0.370, 0.270, 0.570], [0.580, 0.420, 0.380, 0.432]],
    }
    cmap = LinearSegmentedColormap.from_list("bias", ["#2563EB", "#F8FAFC", "#EF4444"])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, (model, values) in zip(axes.flat, models.items()):
        df = pd.DataFrame(values, index=["cs", "de", "en", "it"], columns=["Accuracy", "MAE", "Bias", "QWK"])
        draw_heatmap(ax, df, model, cmap, vmin=-0.65, vmax=0.85, center=0)
    fig.suptitle("图3  各模型在不同语言上的 CEFR 表现", fontsize=14, weight="bold", color="#0F172A", y=1.02)
    save(fig, "figure3_cefr_language_breakdown.png")


def chinese_metrics():
    df = pd.DataFrame(
        {
            "MAE": [7.965, 7.910, 8.185, 9.405],
            "RMSE": [10.120, 10.536, 10.615, 12.146],
            "Bias": [5.445, 2.630, 2.295, 6.175],
            "Pearson": [0.244, 0.003, -0.014, -0.009],
            "Spearman": [0.239, 0.036, 0.005, 0.038],
            "QWK": [0.142, 0.004, -0.017, -0.004],
        },
        index=["gpt-4o-mini", "qwen3-max", "gpt-5", "qwen-plus"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(df.index))
    width = 0.26
    for offset, metric, color in [(-width, "MAE", COLORS["blue"]), (0, "RMSE", COLORS["red"]), (width, "Bias", COLORS["amber"])]:
        axes[0].bar(x + offset, df[metric], width=width, label=metric, color=color)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df.index, rotation=18, ha="right")
    axes[0].set_ylabel("分数误差", color="#475569")
    axes[0].set_title("误差与高估幅度", fontsize=12, weight="bold", color="#0F172A")
    axes[0].legend(frameon=False, fontsize=9)
    clean_axes(axes[0])

    cmap = LinearSegmentedColormap.from_list("corr", ["#DBEAFE", "#F8FAFC", "#14B8A6"])
    draw_heatmap(axes[1], df[["Pearson", "Spearman", "QWK"]], "相关性与一致性", cmap, vmin=-0.04, vmax=0.26)
    fig.suptitle("图4  中文作文评分模型表现", fontsize=14, weight="bold", color="#0F172A", y=1.03)
    save(fig, "figure4_chinese_scoring_metrics.png")


def chinese_distribution():
    df = pd.DataFrame(
        {
            "人工均值": [82.745, 82.745, 82.745, 82.745],
            "预测均值": [88.190, 85.040, 85.375, 88.920],
            "预测标准差": [4.616, 5.981, 5.843, 6.184],
            "预测最小值": [70, 56, 45, 42],
            "预测最大值": [92, 94, 92, 94],
        },
        index=["gpt-4o-mini", "gpt-5", "qwen3-max", "qwen-plus"],
    )
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    y = np.arange(len(df.index))
    ax.hlines(y, df["预测最小值"], df["预测最大值"], color="#CBD5E1", linewidth=8, alpha=0.9, label="预测范围")
    ax.errorbar(df["预测均值"], y, xerr=df["预测标准差"], fmt="o", color=COLORS["blue"], ecolor=COLORS["blue"], capsize=4, label="预测均值 ± 标准差")
    ax.axvline(82.745, color=COLORS["red"], linestyle="--", linewidth=1.8, label="人工均值 82.745")
    for yi, row in enumerate(df.itertuples()):
        ax.text(row.预测最大值 + 0.7, yi, f"{row.预测最小值:.0f}-{row.预测最大值:.0f}", va="center", fontsize=8, color="#475569")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlim(38, 101)
    ax.set_xlabel("作文分数", color="#475569")
    ax.set_title("图5  中文作文预测分数分布", fontsize=14, weight="bold", color="#0F172A", pad=16)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    clean_axes(ax)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    save(fig, "figure5_chinese_prediction_distribution.png")


if __name__ == "__main__":
    cefr_distribution()
    cefr_overall()
    cefr_by_language()
    chinese_metrics()
    chinese_distribution()
    print(f"Saved figures to {OUT.resolve()}")
