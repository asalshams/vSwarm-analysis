import os
from typing import Dict, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


RUN_DIRS: Dict[str, str] = {
    "B1": "results_fibonacci_python_b1",
    "B2": "results_fibonacci_python_b2",
    "B3": "results_fibonacci_python_b3",
    "B4": "results_fibonacci_python_b4",
    "B5": "results_fibonacci_python_b5",
}

# High-level config notes pulled from the user's spreadsheet screenshot
# These are attached to each run for annotation and legends
RUN_CONFIG: Dict[str, Dict[str, str]] = {
    "B1": {
        "pods": "1",
        "container_cpu": "1000m",
        "container_mem": "1Gi",
        "target_conc": "100",
    },
    "B2": {
        "pods": "2",
        "container_cpu": "500m",
        "container_mem": "500Mi",
        "target_conc": "50",
    },
    "B3": {
        "pods": "4",
        "container_cpu": "250m",
        "container_mem": "250Mi",
        "target_conc": "25",
    },
    "B4": {
        "pods": "8",
        "container_cpu": "125m",
        "container_mem": "125Mi",
        "target_conc": "12",
    },
    "B5": {
        "pods": "10",
        "container_cpu": "100m",
        "container_mem": "100Mi",
        "target_conc": "10",
    },
}


def read_detailed_csvs(base_dir: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for run, subdir in RUN_DIRS.items():
        csv_path = os.path.join(base_dir, subdir, f"fibonacci_python_{run.lower()}_detailed_analysis.csv")
        if not os.path.exists(csv_path):
            # Fallback: some folders may have different naming; scan for the detailed file
            candidates = [
                os.path.join(base_dir, subdir, f)
                for f in os.listdir(os.path.join(base_dir, subdir))
                if f.endswith("_detailed_analysis.csv")
            ]
            if not candidates:
                print(f"WARN: Missing detailed CSV for {run} in {subdir}")
                continue
            csv_path = candidates[0]

        df = pd.read_csv(csv_path)
        df["run"] = run
        # Attach readable label that includes the key config
        cfg = RUN_CONFIG.get(run, {})
        df["run_label"] = (
            f"{run} | pods={cfg.get('pods','?')} | CPU={cfg.get('container_cpu','?')} | RAM={cfg.get('container_mem','?')}"
        )
        rows.append(df)

    if not rows:
        raise FileNotFoundError("No detailed analysis CSVs found for B1–B5")
    return pd.concat(rows, ignore_index=True)


def aggregate_by_target(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "min",
        "max",
        "average",
        "median",
        "p95",
        "p99",
        "std",
        "throughput_rps",
        "total_requests",
    ]

    agg_map = {col: ["mean", "std", "min", "max"] for col in metric_cols}
    grouped = df.groupby(["run", "run_label", "target_rps"], as_index=False).agg(agg_map)
    # Flatten columns while preserving group keys
    flat_cols: List[str] = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            base, stat = col
            if stat == "":
                flat_cols.append(base)
            else:
                flat_cols.append(f"{base}_{stat}")
        else:
            flat_cols.append(col)
    grouped.columns = flat_cols
    return grouped


def ensure_output_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, "charts_fibonacci_python_b_comparison")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_aggregated_csv(grouped: pd.DataFrame, out_dir: str) -> None:
    out_csv = os.path.join(out_dir, "fibonacci_python_b_aggregated.csv")
    grouped.to_csv(out_csv, index=False)
    print(f"Wrote aggregated CSV → {out_csv}")


def plot_efficiency(df_agg: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    for run_label, sub in df_agg.groupby("run_label"):
        plt.plot(
            sub["target_rps"],
            sub["throughput_rps_mean"],
            marker="o",
            label=run_label,
        )
        # Optional: show +/- std as a band
        y = sub["throughput_rps_mean"].values
        yerr = sub["throughput_rps_std"].fillna(0).values
        plt.fill_between(sub["target_rps"], y - yerr, y + yerr, alpha=0.12)

    # 45-degree line (ideal: achieved == target)
    xlim = plt.xlim()
    plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle="--", color="black", alpha=0.4, label="ideal y=x")

    plt.title("Achieved throughput vs Target RPS (Fibonacci Python B1–B5)")
    plt.xlabel("Target RPS")
    plt.ylabel("Achieved throughput (RPS, mean)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "efficiency_target_vs_throughput.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved → {out_path}")


def plot_latency_vs_target(df_agg: pd.DataFrame, out_dir: str) -> None:
    metrics = [
        ("median_mean", "Median"),
        ("p95_mean", "p95"),
        ("p99_mean", "p99"),
        ("average_mean", "Average"),
    ]
    for col, label in metrics:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        for run_label, sub in df_agg.groupby("run_label"):
            plt.plot(sub["target_rps"], sub[f"{col}"], marker="o", label=run_label)
            y = sub[f"{col}"].values
            yerr = sub[f"{col.replace('_mean', '_std')}"].fillna(0).values
            plt.fill_between(sub["target_rps"], y - yerr, y + yerr, alpha=0.12)

        plt.title(f"{label} latency vs Target RPS (Fibonacci Python B1–B5)")
        plt.xlabel("Target RPS")
        plt.ylabel(f"{label} latency (same units as source CSV)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"latency_{col}_vs_target.png")
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved → {out_path}")


def plot_latency_vs_throughput(df_agg: pd.DataFrame, out_dir: str) -> None:
    metrics = [
        ("median_mean", "Median"),
        ("p95_mean", "p95"),
        ("p99_mean", "p99"),
        ("average_mean", "Average"),
    ]
    for col, label in metrics:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        for run_label, sub in df_agg.groupby("run_label"):
            plt.plot(
                sub["throughput_rps_mean"],
                sub[f"{col}"],
                marker="o",
                label=run_label,
            )
            y = sub[f"{col}"].values
            yerr = sub[f"{col.replace('_mean', '_std')}"].fillna(0).values
            plt.fill_between(sub["throughput_rps_mean"], y - yerr, y + yerr, alpha=0.12)

        plt.title(f"{label} latency vs Achieved Throughput (Fibonacci Python B1–B5)")
        plt.xlabel("Achieved throughput (RPS, mean)")
        plt.ylabel(f"{label} latency (same units as source CSV)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"latency_{col}_vs_throughput.png")
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved → {out_path}")


def plot_heatmap(df_agg: pd.DataFrame, out_dir: str, value_col: str, title: str, filename: str) -> None:
    pivot = df_agg.pivot_table(
        index="run",
        columns="target_rps",
        values=value_col,
        aggfunc="mean",
    )
    plt.figure(figsize=(14, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd")
    plt.title(title)
    plt.xlabel("Target RPS")
    plt.ylabel("Run (B1–B5)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved → {out_path}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = ensure_output_dir(base_dir)

    raw = read_detailed_csvs(base_dir)
    agg = aggregate_by_target(raw)
    save_aggregated_csv(agg, out_dir)

    # Plots
    plot_efficiency(agg, out_dir)
    plot_latency_vs_target(agg, out_dir)
    plot_latency_vs_throughput(agg, out_dir)
    # Heatmaps at-a-glance
    plot_heatmap(agg, out_dir, value_col="p95_mean", title="p95 latency (mean) heatmap", filename="heatmap_p95_mean.png")
    plot_heatmap(agg, out_dir, value_col="p99_mean", title="p99 latency (mean) heatmap", filename="heatmap_p99_mean.png")
    plot_heatmap(agg, out_dir, value_col="median_mean", title="Median latency (mean) heatmap", filename="heatmap_median_mean.png")
    plot_heatmap(agg, out_dir, value_col="throughput_rps_mean", title="Throughput (mean) heatmap", filename="heatmap_throughput_mean.png")

    print(f"All charts saved to: {out_dir}")


if __name__ == "__main__":
    main()


