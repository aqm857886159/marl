import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from MRRL import (
    calculate_statistics,
    plot_convergence_speed,
    plot_learning_curves,
    plot_performance_distribution,
    plot_performance_radar,
    save_core_data,
    setup_plotting,
)

FINAL_DIR = Path("experiment_outputs_final")
FIG_DIR = FINAL_DIR / "figures"
CORE_DIR = FINAL_DIR / "core_tables"


def load_learning_curves(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Learning curve CSV not found: {path}")

    df = pd.read_csv(path)
    if "episode_return_mean" not in df.columns and "reward" in df.columns:
        df = df.rename(columns={"reward": "episode_return_mean"})
    required_cols = ["algorithm", "seed", "timestep", "episode_return_mean"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"学习曲线缺少必要列: {missing}")
    return df[required_cols].dropna(subset=["episode_return_mean"])


def load_eval_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Eval metric CSV not found: {path}")

    df = pd.read_csv(path)
    id_cols = ["algorithm", "seed", "timestep"]
    metric_cols = [c for c in df.columns if c not in id_cols]
    if not metric_cols:
        raise ValueError("评估指标 CSV 中没有可用的 metric 列")

    melted = df.melt(
        id_vars=id_cols,
        value_vars=metric_cols,
        var_name="metric",
        value_name="value"
    )
    return melted.dropna(subset=["value"])


def main():
    learning_path = FINAL_DIR / "final_learning_curves.csv"
    eval_path = FINAL_DIR / "final_eval_metrics.csv"

    df_curves = load_learning_curves(learning_path)
    df_perf_metrics = load_eval_metrics(eval_path)
    df_convergence = pd.DataFrame(columns=["algorithm", "seed", "steps"])  # 当前暂无收敛速度数据

    setup_plotting()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 3.3 & 3.4
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    plot_learning_curves(df_curves, ax1)
    if df_convergence.empty:
        ax2.set_title("Figure 3.4: Convergence Speed (No Data)")
        ax2.text(0.5, 0.5, "暂无可用数据", ha="center", va="center", fontsize=12)
        ax2.axis("off")
    else:
        plot_convergence_speed(df_convergence, ax2)
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / "figure_3.3_3.4_learning_curves.png", dpi=300)

    # Figure 3.5
    fig2, ax3 = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    plot_performance_radar(df_perf_metrics, ax3)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "figure_3.5_radar_chart.png", dpi=300)

    # Figure 3.6
    plot_performance_distribution(df_perf_metrics, None)
    plt.savefig(FIG_DIR / "figure_3.6_violin_plots.png", dpi=300)

    # Tables (3.5 & 3.6)
    agg_stats, p_value_matrix = calculate_statistics(df_perf_metrics)
    save_core_data(str(CORE_DIR), df_curves, df_perf_metrics, df_convergence, agg_stats, p_value_matrix)

    # 额外备份最终汇总 CSV -> core_tables 目录，方便论文引用
    CORE_DIR.mkdir(parents=True, exist_ok=True)
    df_curves.to_csv(CORE_DIR / "learning_curves_final.csv", index=False)
    df_perf_metrics.to_csv(CORE_DIR / "performance_metrics_final.csv", index=False)

    plt.close("all")
    print("图表与表格已生成: \n-", FIG_DIR.resolve(), "\n-", CORE_DIR.resolve())


if __name__ == "__main__":
    main()

