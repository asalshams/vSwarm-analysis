import os
from typing import Dict, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


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


def read_timeseries_data(base_dir: str) -> Dict[str, pd.DataFrame]:
    """Read pod monitoring data for timeseries visualization."""
    timeseries_data = {}
    
    for run, subdir in RUN_DIRS.items():
        pod_csv_path = os.path.join(base_dir, subdir, "fibonacci_python_{run.lower()}_pod_monitoring.csv")
        if not os.path.exists(pod_csv_path):
            # Try alternative naming patterns
            candidates = [
                os.path.join(base_dir, subdir, f)
                for f in os.listdir(os.path.join(base_dir, subdir))
                if f.endswith("_pod_monitoring.csv")
            ]
            if candidates:
                pod_csv_path = candidates[0]
            else:
                print(f"WARN: Missing pod monitoring CSV for {run} in {subdir}")
                continue
        
        try:
            df = pd.read_csv(pod_csv_path)
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            timeseries_data[run] = df
        except Exception as e:
            print(f"WARN: Failed to read pod monitoring data for {run}: {e}")
            continue
    
    return timeseries_data


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


def plot_timeseries_comparison(timeseries_data: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """Create timeseries comparison charts showing resource usage over time for all runs."""
    if not timeseries_data:
        print("WARN: No timeseries data available for comparison")
        return
    
    # Plot CPU usage comparison
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    for run, data in timeseries_data.items():
        if 'cpu_usage_millicores' in data.columns:
            # Normalize time to 0-100% for comparison
            time_normalized = np.linspace(0, 100, len(data))
            plt.plot(time_normalized, data['cpu_usage_millicores'], 
                    label=f"{run} | pods={RUN_CONFIG.get(run, {}).get('pods', '?')}", 
                    linewidth=2, alpha=0.8)
    
    plt.title("CPU Usage Comparison Over Time (Fibonacci Python B1–B5)")
    plt.xlabel("Test Progress (%)")
    plt.ylabel("CPU Usage (millicores)")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "timeseries_cpu_comparison.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved → {out_path}")
    
    # Plot Memory usage comparison
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    for run, data in timeseries_data.items():
        if 'memory_usage_mib' in data.columns:
            time_normalized = np.linspace(0, 100, len(data))
            plt.plot(time_normalized, data['memory_usage_mib'], 
                    label=f"{run} | pods={RUN_CONFIG.get(run, {}).get('pods', '?')}", 
                    linewidth=2, alpha=0.8)
    
    plt.title("Memory Usage Comparison Over Time (Fibonacci Python B1–B5)")
    plt.xlabel("Test Progress (%)")
    plt.ylabel("Memory Usage (MiB)")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "timeseries_memory_comparison.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved → {out_path}")
    
    # Plot Pod count comparison
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    for run, data in timeseries_data.items():
        if 'pod_count' in data.columns:
            time_normalized = np.linspace(0, 100, len(data))
            plt.plot(time_normalized, data['pod_count'], 
                    label=f"{run} | pods={RUN_CONFIG.get(run, {}).get('pods', '?')}", 
                    linewidth=2, alpha=0.8, marker='o', markersize=4)
    
    plt.title("Pod Count Comparison Over Time (Fibonacci Python B1–B5)")
    plt.xlabel("Test Progress (%)")
    plt.ylabel("Pod Count")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "timeseries_pod_count_comparison.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved → {out_path}")


def plot_resource_efficiency(df_agg: pd.DataFrame, timeseries_data: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """Create charts showing resource efficiency vs performance."""
    if not timeseries_data:
        print("WARN: No timeseries data available for resource efficiency analysis")
        return
    
    # Calculate average resource usage per run
    resource_metrics = {}
    for run, data in timeseries_data.items():
        if 'cpu_usage_millicores' in data.columns and 'memory_usage_mib' in data.columns:
            avg_cpu = data['cpu_usage_millicores'].mean()
            avg_memory = data['memory_usage_mib'].mean()
            resource_metrics[run] = {
                'avg_cpu': avg_cpu,
                'avg_memory': avg_memory
            }
    
    # Plot CPU efficiency vs throughput
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    for run in df_agg['run'].unique():
        run_data = df_agg[df_agg['run'] == run]
        if run in resource_metrics:
            avg_cpu = resource_metrics[run]['avg_cpu']
            throughput = run_data['throughput_rps_mean'].mean()
            plt.scatter(avg_cpu, throughput, s=100, alpha=0.7, 
                       label=f"{run} | pods={RUN_CONFIG.get(run, {}).get('pods', '?')}")
    
    plt.xlabel("Average CPU Usage (millicores)")
    plt.ylabel("Average Throughput (RPS)")
    plt.title("CPU Efficiency vs Throughput (Fibonacci Python B1–B5)")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "resource_efficiency_cpu_vs_throughput.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved → {out_path}")
    
    # Plot Memory efficiency vs throughput
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    for run in df_agg['run'].unique():
        run_data = df_agg[df_agg['run'] == run]
        if run in resource_metrics:
            avg_memory = resource_metrics[run]['avg_memory']
            throughput = run_data['throughput_rps_mean'].mean()
            plt.scatter(avg_memory, throughput, s=100, alpha=0.7, 
                       label=f"{run} | pods={RUN_CONFIG.get(run, {}).get('pods', '?')}")
    
    plt.xlabel("Average Memory Usage (MiB)")
    plt.ylabel("Average Throughput (RPS)")
    plt.title("Memory Efficiency vs Throughput (Fibonacci Python B1–B5)")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "resource_efficiency_memory_vs_throughput.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved → {out_path}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = ensure_output_dir(base_dir)

    raw = read_detailed_csvs(base_dir)
    agg = aggregate_by_target(raw)
    save_aggregated_csv(agg, out_dir)

    # Read timeseries data for additional visualizations
    timeseries_data = read_timeseries_data(base_dir)

    # Plots
    plot_efficiency(agg, out_dir)
    plot_latency_vs_target(agg, out_dir)
    plot_latency_vs_throughput(agg, out_dir)
    # Heatmaps at-a-glance
    plot_heatmap(agg, out_dir, value_col="p95_mean", title="p95 latency (mean) heatmap", filename="heatmap_p95_mean.png")
    plot_heatmap(agg, out_dir, value_col="p99_mean", title="p99 latency (mean) heatmap", filename="heatmap_p99_mean.png")
    plot_heatmap(agg, out_dir, value_col="median_mean", title="Median latency (mean) heatmap", filename="heatmap_median_mean.png")
    plot_heatmap(agg, out_dir, value_col="throughput_rps_mean", title="Throughput (mean) heatmap", filename="heatmap_throughput_mean.png")
    
    # New timeseries visualizations
    plot_timeseries_comparison(timeseries_data, out_dir)
    plot_resource_efficiency(agg, timeseries_data, out_dir)

    print(f"All charts saved to: {out_dir}")


if __name__ == "__main__":
    main()


