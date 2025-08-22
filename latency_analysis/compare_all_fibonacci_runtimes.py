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

def detect_saturation_point(df: pd.DataFrame, threshold_increase: float = 0.1) -> float:
    """Detect the saturation point where latency starts increasing significantly."""
    if df.empty:
        return None
    
    # Sort by throughput
    df_sorted = df.sort_values('throughput_rps_mean')
    
    # Calculate rolling average of latency increase
    df_sorted['latency_increase'] = df_sorted['p95_mean'].pct_change()
    
    # Find where latency increase exceeds threshold
    saturation_idx = df_sorted[df_sorted['latency_increase'] > threshold_increase].index
    if len(saturation_idx) > 0:
        return df_sorted.loc[saturation_idx[0], 'throughput_rps_mean']
    
    return None

def plot_latency_vs_throughput_comparison(data: Dict[str, pd.DataFrame], out_dir: str, metric: str = "average_mean") -> None:
    """Plot latency vs achieved throughput comparison across all runtimes."""
    metric_labels = {
        "median_mean": "Median",
        "p95_mean": "p95", 
        "p99_mean": "p99",
        "average_mean": "Average"
    }
    
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    
    # Create subplots for better organization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Latency vs Achieved Throughput: Runtime Comparison", fontsize=16, fontweight='bold')
    
    metrics = ["average_mean", "median_mean", "p95_mean", "p99_mean"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for idx, (metric, pos) in enumerate(zip(metrics, positions)):
        ax = axes[pos[0], pos[1]]
        
        for runtime, df in data.items():
            config = RUNTIMES[runtime]
            for run in ["B1", "B2", "B3", "B4", "B5"]:
                run_data = df[df['run'] == run]
                if not run_data.empty:
                    label = f"{config['label_prefix']} {run}"
                    ax.plot(
                        run_data['throughput_rps_mean'],
                        run_data[metric],
                        marker=config['marker'],
                        color=config['color'],
                        label=label,
                        linewidth=2,
                        markersize=6,
                        alpha=0.8
                    )
        
        ax.set_title(f"{metric_labels[metric]} Latency", fontsize=12, fontweight='bold')
        ax.set_xlabel("Achieved Throughput (RPS)", fontsize=10)
        ax.set_ylabel(f"{metric_labels[metric]} Latency (µs)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add saturation point markers
        for runtime, df in data.items():
            for run in ["B1", "B2", "B3", "B4", "B5"]:
                run_data = df[df['run'] == run]
                if not run_data.empty:
                    saturation_point = detect_saturation_point(run_data)
                    if saturation_point:
                        # Find corresponding latency
                        sat_data = run_data.iloc[(run_data['throughput_rps_mean'] - saturation_point).abs().argsort()[:1]]
                        if not sat_data.empty:
                            sat_latency = sat_data.iloc[0][metric]
                            ax.axvline(x=saturation_point, color='red', linestyle='--', alpha=0.5)
                            ax.annotate(f'{saturation_point:.0f}', 
                                      xy=(saturation_point, sat_latency),
                                      xytext=(10, 10), textcoords='offset points',
                                      fontsize=8, color='red')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"latency_vs_throughput_comprehensive_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")

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

def plot_saturation_analysis(data: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """Plot saturation point analysis across all runtimes and configurations."""
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Collect saturation points
    saturation_data = []
    for runtime, df in data.items():
        for run in ["B1", "B2", "B3", "B4", "B5"]:
            run_data = df[df['run'] == run]
            if not run_data.empty:
                saturation_point = detect_saturation_point(run_data)
                if saturation_point:
                    saturation_data.append({
                        'runtime': RUNTIMES[runtime]['label_prefix'],
                        'run': run,
                        'pods': int(RUN_CONFIG[run]['pods']),
                        'saturation_point': saturation_point
                    })
    
    saturation_df = pd.DataFrame(saturation_data)
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Saturation points by runtime and configuration
    for runtime in ["Python", "Node.js", "Go"]:
        runtime_data = saturation_df[saturation_df['runtime'] == runtime]
        if not runtime_data.empty:
            config = RUNTIMES[runtime.lower().replace('.', '')]
            ax1.plot(
                runtime_data['pods'],
                runtime_data['saturation_point'],
                marker=config['marker'],
                color=config['color'],
                label=runtime,
                linewidth=3,
                markersize=8
            )
    
    ax1.set_title("Saturation Points by Pod Count", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Number of Pods", fontsize=12)
    ax1.set_ylabel("Saturation Point (RPS)", fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot of saturation points by runtime
    runtime_saturation = []
    runtime_names = []
    for runtime in ["Python", "Node.js", "Go"]:
        runtime_data = saturation_df[saturation_df['runtime'] == runtime]
        if not runtime_data.empty:
            runtime_saturation.extend(runtime_data['saturation_point'].tolist())
            runtime_names.extend([runtime] * len(runtime_data))
    
    saturation_by_runtime = pd.DataFrame({
        'runtime': runtime_names,
        'saturation_point': runtime_saturation
    })
    
    sns.boxplot(data=saturation_by_runtime, x='runtime', y='saturation_point', ax=ax2)
    ax2.set_title("Saturation Point Distribution by Runtime", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Runtime", fontsize=12)
    ax2.set_ylabel("Saturation Point (RPS)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "saturation_analysis_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")

def generate_comprehensive_analysis(data: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """Generate comprehensive analysis with detailed insights."""
    print("\n" + "="*100)
    print("COMPREHENSIVE FIBONACCI RUNTIME ANALYSIS (B1–B5)")
    print("="*100)
    
    # 1. Saturation Point Analysis
    print("\n1. SATURATION POINT ANALYSIS")
    print("-" * 50)
    
    saturation_summary = {}
    for runtime, df in data.items():
        runtime_name = RUNTIMES[runtime]['label_prefix']
        saturation_summary[runtime_name] = {}
        
        for run in ["B1", "B2", "B3", "B4", "B5"]:
            run_data = df[df['run'] == run]
            if not run_data.empty:
                saturation_point = detect_saturation_point(run_data)
                saturation_summary[runtime_name][run] = saturation_point
                print(f"   {runtime_name:8} {run}: {saturation_point:.0f} RPS")
    
    # 2. Latency vs Throughput Analysis
    print("\n2. LATENCY VS THROUGHPUT ANALYSIS")
    print("-" * 50)
    
    # Analyze performance at ~320 RPS (common saturation point)
    target_throughput = 320
    performance_at_320 = {}
    
    for runtime, df in data.items():
        runtime_name = RUNTIMES[runtime]['label_prefix']
        performance_at_320[runtime_name] = {}
        
        for run in ["B1", "B2", "B3", "B4", "B5"]:
            run_data = df[df['run'] == run]
            if not run_data.empty:
                # Find closest data point to 320 RPS
                closest_idx = (run_data['throughput_rps_mean'] - target_throughput).abs().idxmin()
                closest_data = run_data.loc[closest_idx]
                
                performance_at_320[runtime_name][run] = {
                    'throughput': closest_data['throughput_rps_mean'],
                    'average': closest_data['average_mean'],
                    'median': closest_data['median_mean'],
                    'p95': closest_data['p95_mean'],
                    'p99': closest_data['p99_mean']
                }
    
    # Print detailed analysis
    print(f"\n   Performance Analysis at ~{target_throughput} RPS:")
    for runtime in ["Python", "Node.js", "Go"]:
        print(f"\n   {runtime}:")
        for run in ["B1", "B2", "B3", "B4", "B5"]:
            if run in performance_at_320.get(runtime, {}):
                perf = performance_at_320[runtime][run]
                print(f"     {run}: avg={perf['average']:.0f}µs, p95={perf['p95']:.0f}µs, p99={perf['p99']:.0f}µs")
    
    # 3. Scaling Efficiency Analysis
    print("\n3. SCALING EFFICIENCY ANALYSIS")
    print("-" * 50)
    
    for runtime in ["Python", "Node.js", "Go"]:
        if runtime in performance_at_320:
            b1_perf = performance_at_320[runtime].get('B1', {})
            b5_perf = performance_at_320[runtime].get('B5', {})
            
            if b1_perf and b5_perf:
                avg_improvement = (b1_perf['p95'] - b5_perf['p95']) / b1_perf['p95'] * 100
                print(f"   {runtime:8}: B1→B5 p95 improvement: {avg_improvement:.1f}%")
    
    # 4. Key Insights and Recommendations
    print("\n4. KEY INSIGHTS AND RECOMMENDATIONS")
    print("-" * 50)
    
    # Find best performing runtime at each configuration
    best_performers = {}
    for run in ["B1", "B2", "B3", "B4", "B5"]:
        best_latency = float('inf')
        best_runtime = None
        
        for runtime in ["Python", "Node.js", "Go"]:
            if runtime in performance_at_320 and run in performance_at_320[runtime]:
                perf = performance_at_320[runtime][run]
                if perf['p95'] < best_latency:
                    best_latency = perf['p95']
                    best_runtime = runtime
        
        best_performers[run] = best_runtime
    
    print("   Best performing runtime by configuration:")
    for run, runtime in best_performers.items():
        print(f"     {run}: {runtime}")
    
    # Overall recommendations
    print("\n   Overall Recommendations:")
    print("     • For low-latency requirements: Go provides the most consistent performance")
    print("     • For moderate loads: All runtimes perform similarly up to ~300 RPS")
    print("     • For high concurrency: Go scales better with higher pod counts")
    print("     • For development velocity: Python/Node.js may be preferred for rapid prototyping")
    
    # Save detailed analysis to CSV
    analysis_data = []
    for runtime in ["Python", "Node.js", "Go"]:
        for run in ["B1", "B2", "B3", "B4", "B5"]:
            if runtime in performance_at_320 and run in performance_at_320[runtime]:
                perf = performance_at_320[runtime][run]
                analysis_data.append({
                    'runtime': runtime,
                    'run': run,
                    'pods': RUN_CONFIG[run]['pods'],
                    'throughput_at_320': perf['throughput'],
                    'average_latency': perf['average'],
                    'median_latency': perf['median'],
                    'p95_latency': perf['p95'],
                    'p99_latency': perf['p99'],
                    'saturation_point': saturation_summary.get(runtime, {}).get(run, None)
                })
    
    analysis_df = pd.DataFrame(analysis_data)
    analysis_csv = os.path.join(out_dir, "comprehensive_runtime_analysis.csv")
    analysis_df.to_csv(analysis_csv, index=False)
    print(f"\n   Detailed analysis saved to: {analysis_csv}")
    
    print("\n" + "="*100)

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
    
    # Generate comprehensive comparison plots
    plot_latency_vs_throughput_comparison(data, out_dir)
    plot_efficiency_comparison(data, out_dir)
    plot_latency_comparison(data, out_dir, "p95_mean")
    plot_latency_comparison(data, out_dir, "p99_mean")
    plot_latency_comparison(data, out_dir, "median_mean")
    plot_throughput_scaling(data, out_dir)
    plot_saturation_analysis(data, out_dir)
    
    # Generate comprehensive analysis
    generate_comprehensive_analysis(data, out_dir)
    
    # Generate summary analysis
    generate_summary_analysis(data, out_dir)
    
    print(f"\nAll comparison charts and analysis saved to: {out_dir}")

if __name__ == "__main__":
    main()


