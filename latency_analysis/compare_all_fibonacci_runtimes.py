import os
from typing import Dict, List, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Runtime configurations
RUNTIMES = {
    "python": {
        "color": "#3776ab",
        "marker": "o",
        "label_prefix": "Python"
    },
    "nodejs": {
        "color": "#f7df1e", 
        "marker": "s",
        "label_prefix": "Node.js"
    },
    "go": {
        "color": "#00add8",
        "marker": "^",
        "label_prefix": "Go"
    }
}

# Same capacity plan across all runtimes
RUN_CONFIG = {
    "B1": {"pods": "1", "container_cpu": "1000m", "container_mem": "1Gi", "target_conc": "100"},
    "B2": {"pods": "2", "container_cpu": "500m", "container_mem": "500Mi", "target_conc": "50"},
    "B3": {"pods": "4", "container_cpu": "250m", "container_mem": "250Mi", "target_conc": "25"},
    "B4": {"pods": "8", "container_cpu": "125m", "container_mem": "125Mi", "target_conc": "12"},
    "B5": {"pods": "10", "container_cpu": "100m", "container_mem": "100Mi", "target_conc": "10"},
}

def load_aggregated_data(base_dir: str) -> Dict[str, pd.DataFrame]:
    """Load aggregated CSV data for all three runtimes."""
    data = {}
    for runtime in ["python", "nodejs", "go"]:
        csv_path = os.path.join(
            base_dir, 
            f"charts_fibonacci_{runtime}_b_comparison",
            f"fibonacci_{runtime}_b_aggregated.csv"
        )
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['runtime'] = runtime
            data[runtime] = df
        else:
            print(f"Warning: {csv_path} not found")
    
    return data

def ensure_output_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, "charts_all_runtimes_comparison")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def plot_efficiency_comparison(data: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """Plot efficiency comparison across all runtimes."""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    for runtime, df in data.items():
        config = RUNTIMES[runtime]
        for run in ["B1", "B2", "B3", "B4", "B5"]:
            run_data = df[df['run'] == run]
            if not run_data.empty:
                label = f"{config['label_prefix']} {run}"
                plt.plot(
                    run_data['target_rps'], 
                    run_data['throughput_rps_mean'],
                    marker=config['marker'],
                    color=config['color'],
                    label=label,
                    linewidth=2,
                    markersize=6
                )
    
    # Ideal line
    xlim = plt.xlim()
    plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], 
             linestyle="--", color="black", alpha=0.4, label="ideal y=x")
    
    plt.title("Throughput Efficiency Comparison: Python vs Node.js vs Go", fontsize=14, fontweight='bold')
    plt.xlabel("Target RPS", fontsize=12)
    plt.ylabel("Achieved Throughput (RPS)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "efficiency_comparison_all_runtimes.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")

def plot_latency_comparison(data: Dict[str, pd.DataFrame], out_dir: str, metric: str = "p95_mean") -> None:
    """Plot latency comparison for a specific metric."""
    metric_labels = {
        "median_mean": "Median",
        "p95_mean": "p95", 
        "p99_mean": "p99",
        "average_mean": "Average"
    }
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    for runtime, df in data.items():
        config = RUNTIMES[runtime]
        for run in ["B1", "B2", "B3", "B4", "B5"]:
            run_data = df[df['run'] == run]
            if not run_data.empty:
                label = f"{config['label_prefix']} {run}"
                plt.plot(
                    run_data['target_rps'],
                    run_data[metric],
                    marker=config['marker'],
                    color=config['color'],
                    label=label,
                    linewidth=2,
                    markersize=6
                )
    
    plt.title(f"{metric_labels[metric]} Latency vs Target RPS: Runtime Comparison", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Target RPS", fontsize=12)
    plt.ylabel(f"{metric_labels[metric]} Latency", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f"latency_{metric}_comparison_all_runtimes.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")

def plot_throughput_scaling(data: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """Plot max throughput scaling with pod count."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    pods_map = {"B1": 1, "B2": 2, "B3": 4, "B4": 8, "B5": 10}
    
    for runtime, df in data.items():
        config = RUNTIMES[runtime]
        max_throughputs = []
        pod_counts = []
        
        for run in ["B1", "B2", "B3", "B4", "B5"]:
            run_data = df[df['run'] == run]
            if not run_data.empty:
                max_tp = run_data['throughput_rps_mean'].max()
                max_throughputs.append(max_tp)
                pod_counts.append(pods_map[run])
        
        plt.plot(
            pod_counts,
            max_throughputs,
            marker=config['marker'],
            color=config['color'],
            label=config['label_prefix'],
            linewidth=3,
            markersize=8
        )
    
    plt.title("Max Throughput Scaling with Pod Count", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Pods", fontsize=12)
    plt.ylabel("Max Achieved Throughput (RPS)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "throughput_scaling_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")

def generate_summary_analysis(data: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """Generate comprehensive summary analysis."""
    summary_rows = []
    
    for runtime, df in data.items():
        for run in ["B1", "B2", "B3", "B4", "B5"]:
            run_data = df[df['run'] == run]
            if not run_data.empty:
                max_throughput = run_data['throughput_rps_mean'].max()
                max_target = run_data.loc[run_data['throughput_rps_mean'].idxmax(), 'target_rps']
                
                # Find latency at ~320 RPS achieved
                target_320 = run_data.iloc[(run_data['throughput_rps_mean'] - 320).abs().argsort()[:1]]
                if not target_320.empty:
                    p95_at_320 = target_320.iloc[0]['p95_mean']
                    median_at_320 = target_320.iloc[0]['median_mean']
                else:
                    p95_at_320 = median_at_320 = None
                
                summary_rows.append({
                    'runtime': RUNTIMES[runtime]['label_prefix'],
                    'run': run,
                    'pods': RUN_CONFIG[run]['pods'],
                    'cpu': RUN_CONFIG[run]['container_cpu'],
                    'max_throughput': max_throughput,
                    'max_throughput_target': max_target,
                    'p95_at_320rps': p95_at_320,
                    'median_at_320rps': median_at_320
                })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, "runtime_comparison_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary → {summary_csv}")
    
    # Print key insights
    print("\n" + "="*80)
    print("RUNTIME COMPARISON SUMMARY")
    print("="*80)
    
    # Max throughput by runtime
    print("\n1. MAX THROUGHPUT BY RUNTIME:")
    for runtime in ["Python", "Node.js", "Go"]:
        runtime_data = summary_df[summary_df['runtime'] == runtime]
        max_tp = runtime_data['max_throughput'].max()
        best_run = runtime_data.loc[runtime_data['max_throughput'].idxmax(), 'run']
        print(f"   {runtime:8}: {max_tp:.1f} RPS (achieved in {best_run})")
    
    # Scaling efficiency
    print("\n2. SCALING EFFICIENCY (B1 to B5 max throughput):")
    for runtime in ["Python", "Node.js", "Go"]:
        runtime_data = summary_df[summary_df['runtime'] == runtime]
        b1_tp = runtime_data[runtime_data['run'] == 'B1']['max_throughput'].iloc[0]
        b5_tp = runtime_data[runtime_data['run'] == 'B5']['max_throughput'].iloc[0]
        scaling_ratio = b5_tp / b1_tp
        print(f"   {runtime:8}: B1={b1_tp:.1f}, B5={b5_tp:.1f}, scaling={scaling_ratio:.2f}x")
    
    # Latency comparison at ~320 RPS
    print("\n3. LATENCY AT ~320 RPS ACHIEVED:")
    for runtime in ["Python", "Node.js", "Go"]:
        runtime_data = summary_df[summary_df['runtime'] == runtime]
        avg_p95 = runtime_data['p95_at_320rps'].mean()
        avg_median = runtime_data['median_at_320rps'].mean()
        print(f"   {runtime:8}: p95={avg_p95:.0f}, median={avg_median:.0f}")
    
    print("\n" + "="*80)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = ensure_output_dir(base_dir)
    
    # Load all runtime data
    data = load_aggregated_data(base_dir)
    
    if not data:
        print("No aggregated data found. Please run the individual runtime comparison scripts first.")
        return
    
    print(f"Loaded data for runtimes: {list(data.keys())}")
    
    # Generate comparison plots
    plot_efficiency_comparison(data, out_dir)
    plot_latency_comparison(data, out_dir, "p95_mean")
    plot_latency_comparison(data, out_dir, "p99_mean")
    plot_latency_comparison(data, out_dir, "median_mean")
    plot_throughput_scaling(data, out_dir)
    
    # Generate summary analysis
    generate_summary_analysis(data, out_dir)
    
    print(f"\nAll comparison charts saved to: {out_dir}")

if __name__ == "__main__":
    main()


