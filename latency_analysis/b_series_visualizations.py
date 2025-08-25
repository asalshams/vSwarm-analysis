#!/usr/bin/env python3
"""
B-Series Visualizations
======================

Creates visualizations specifically for B-series (B1-B5) test runs.
Focuses on pod scaling strategy analysis and resource distribution effects.

Features:
- Cross-runtime B-series comparisons
- Scaling efficiency analysis (B1 → B5)
- Maximum throughput comparisons by configuration
- Pod scaling strategy effectiveness
- Runtime-specific charts (like compare scripts)
- Runtime order: Python → Node.js → Go
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
from pathlib import Path
from typing import Dict, List, Optional
import re
from io import StringIO

# Set style for consistent, professional plots
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid")

# =============================================================================
# CONSISTENT COLOR SCHEME CONFIGURATION
# =============================================================================

RUNTIME_COLORS = {
    "python": {
        "primary": "#FFC107",    # Yellow
        "label": "Python"
    },
    "nodejs": {
        "primary": "#4CAF50",    # Green
        "label": "Node.js"
    },
    "go": {
        "primary": "#2196F3",    # Blue
        "label": "Go"
    }
}

# Runtime processing order
RUNTIME_ORDER = ["python", "nodejs", "go"]

# B-series configuration details
RUN_CONFIG = {
    "B1": {"pods": "1", "container_cpu": "1000m", "container_mem": "1Gi", "target_conc": "100"},
    "B2": {"pods": "2", "container_cpu": "500m", "container_mem": "500Mi", "target_conc": "50"},
    "B3": {"pods": "4", "container_cpu": "250m", "container_mem": "250Mi", "target_conc": "25"},
    "B4": {"pods": "8", "container_cpu": "125m", "container_mem": "125Mi", "target_conc": "12"},
    "B5": {"pods": "10", "container_cpu": "100m", "container_mem": "100Mi", "target_conc": "10"},
}

# =============================================================================
# B-SERIES VISUALIZATION GENERATOR CLASS
# =============================================================================

class BSeriesVisualizer:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "b_series_visualizations_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create runtime-specific directories
        self.runtime_dirs = {}
        for runtime in RUNTIME_ORDER:
            runtime_dir = os.path.join(self.output_dir, f"{runtime}_b_series")
            os.makedirs(runtime_dir, exist_ok=True)
            self.runtime_dirs[runtime] = runtime_dir
        
        print("=" * 60)
        print("B-SERIES VISUALIZATIONS")
        print("=" * 60)
        print(f"Base Directory: {base_dir}")
        print(f"Central charts: {self.output_dir}")
        print(f"Runtime charts: {', '.join([f'{runtime}_b_series' for runtime in RUNTIME_ORDER])}")
        print(f"Focus: Pod scaling strategy analysis (containerConcurrency=100)")
        print(f"Runtime Order: {' → '.join([RUNTIME_COLORS[r]['label'] for r in RUNTIME_ORDER])}")
        print("=" * 60)

    # =========================================================================
    # DATA LOADING UTILITIES
    # =========================================================================

    def load_achieved_rps_data(self, results_dir: str) -> Optional[pd.DataFrame]:
        """Load achieved RPS data from existing summary file."""
        summary_file = os.path.join(results_dir, 'achieved_rps_summary.csv')
        if not os.path.exists(summary_file):
            print(f"Warning: {summary_file} not found")
            print(f"Please run extract_achieved_rps.py first to generate summary files")
            return None
        
        try:
            # Read file and filter out summary section
            with open(summary_file, 'r') as f:
                lines = f.readlines()
            
            # Find where the summary section starts
            data_lines = []
            for line in lines:
                if line.startswith('===') or line.startswith('Metric,'):
                    break
                data_lines.append(line)
            
            # Read the filtered data
            df = pd.read_csv(StringIO(''.join(data_lines)))
            
            # Clean and convert data
            df['target_rps'] = pd.to_numeric(df['target_rps'], errors='coerce')
            df['max_achieved'] = pd.to_numeric(df['max_achieved'], errors='coerce')
            df['avg_achieved'] = pd.to_numeric(df['avg_achieved'], errors='coerce')
            df['efficiency_pct'] = pd.to_numeric(df['efficiency_pct'], errors='coerce')
            
            # Remove any rows with NaN values
            df = df.dropna(subset=['target_rps', 'max_achieved'])
            
            return df
        except Exception as e:
            print(f"Error loading achieved RPS data from {summary_file}: {e}")
            return None

    def load_detailed_analysis_data(self, results_dir: str) -> Optional[pd.DataFrame]:
        """Load detailed analysis data for latency metrics."""
        detail_files = glob.glob(os.path.join(results_dir, '*_detailed_analysis.csv'))
        if not detail_files:
            print(f"Warning: No detailed analysis CSV found in {results_dir}")
            return None
        
        try:
            df = pd.read_csv(detail_files[0])
            
            # Group by target_rps and calculate averages across iterations
            grouped = df.groupby('target_rps').agg({
                'average': 'mean',
                'median': 'mean', 
                'p95': 'mean',
                'p99': 'mean',
                'throughput_rps': 'mean',
                'total_requests': 'mean'
            }).reset_index()
            
            return grouped
        except Exception as e:
            print(f"Error loading detailed analysis data: {e}")
            return None

    def find_b_series_directories(self) -> Dict[str, List[str]]:
        """Find all B-series result directories organized by runtime."""
        b_series_dirs = {runtime: [] for runtime in RUNTIME_ORDER}
        
        for item in os.listdir(self.base_dir):
            if item.startswith('results_fibonacci_') and '_b' in item:
                # Extract runtime and test type
                parts = item.replace('results_fibonacci_', '').split('_')
                if len(parts) >= 2:
                    runtime = parts[0]
                    test_type = '_'.join(parts[1:])
                    
                    if runtime in RUNTIME_COLORS and test_type.startswith('b'):
                        b_series_dirs[runtime].append(os.path.join(self.base_dir, item))
        
        # Sort each runtime's directories
        for runtime in b_series_dirs:
            b_series_dirs[runtime].sort()
        
        return b_series_dirs

    def load_runtime_data(self, b_series_dirs: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """Load and aggregate data for all B-series runtimes."""
        all_runtime_data = {}
        
        for runtime in RUNTIME_ORDER:
            if runtime in b_series_dirs and b_series_dirs[runtime]:
                runtime_aggregated = []
                
                for results_dir in b_series_dirs[runtime]:
                    # Extract run name (B1, B2, etc.)
                    run_name = os.path.basename(results_dir).split('_')[-1].upper()
                    
                    # Load data
                    achieved_df = self.load_achieved_rps_data(results_dir)
                    detailed_df = self.load_detailed_analysis_data(results_dir)
                    
                    if achieved_df is not None and detailed_df is not None:
                        # Merge and aggregate
                        merged = pd.merge(detailed_df, achieved_df[['target_rps', 'max_achieved']], 
                                        on='target_rps', how='left')
                        merged['actual_rps'] = merged['max_achieved'].fillna(merged['throughput_rps'])
                        merged['run'] = run_name
                        runtime_aggregated.append(merged)
                
                if runtime_aggregated:
                    combined_df = pd.concat(runtime_aggregated, ignore_index=True)
                    all_runtime_data[runtime] = combined_df
                    print(f"  Loaded {runtime} B-series data: {len(combined_df)} points across {len(runtime_aggregated)} runs")
                else:
                    print(f"  No valid data for {runtime} B-series")
        
        return all_runtime_data

    def extract_maximum_throughput_data(self, all_runtime_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract maximum throughput achieved by each runtime/config combination."""
        max_data = []
        
        for runtime, df in all_runtime_data.items():
            for run in ["B1", "B2", "B3", "B4", "B5"]:
                run_data = df[df['run'] == run]
                if not run_data.empty:
                    max_throughput = run_data['actual_rps'].max()
                    config = RUN_CONFIG[run]
                    
                    max_data.append({
                        'runtime': runtime,
                        'run': run,
                        'max_throughput': max_throughput,
                        'pods': int(config['pods']),
                        'cpu': config['container_cpu'],
                        'memory': config['container_mem'],
                        'target_conc': int(config['target_conc'])
                    })
        
        return pd.DataFrame(max_data)

    # =========================================================================
    # VISUALIZATION GENERATORS
    # =========================================================================

    def create_maximum_throughput_comparison(self, max_data: pd.DataFrame):
        """Create maximum throughput comparison by configuration."""
        print(f"\nCreating maximum throughput comparison")
        
        if max_data.empty:
            print("  No data available")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set up grouped bar chart
        runs = ["B1", "B2", "B3", "B4", "B5"]
        x = np.arange(len(runs))
        width = 0.25
        
        # Plot bars for each runtime
        for i, runtime in enumerate(RUNTIME_ORDER):
            if runtime in max_data['runtime'].values:
                runtime_data = max_data[max_data['runtime'] == runtime]
                throughputs = []
                
                for run in runs:
                    run_throughput = runtime_data[runtime_data['run'] == run]['max_throughput']
                    throughputs.append(run_throughput.iloc[0] if not run_throughput.empty else 0)
                
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                bars = ax.bar(x + i*width, throughputs, width, 
                             label=label, color=color, alpha=0.8)
                
                # Add value labels
                for bar, value in zip(bars, throughputs):
                    if value > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                               f'{value:.0f}', ha='center', va='bottom', 
                               fontsize=9, fontweight='bold')
        
        # Customize chart
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Maximum Achieved Throughput (RPS)', fontsize=12, fontweight='bold')
        ax.set_title('B-Series Maximum Throughput by Configuration\nPod Scaling Strategy Analysis (containerConcurrency=100)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add configuration details
        config_labels = []
        for run in runs:
            config = RUN_CONFIG[run]
            config_labels.append(f"{run}\n{config['pods']} pods\n{config['container_cpu']}")
        
        ax.set_xticks(x + width)
        ax.set_xticklabels(config_labels, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = "b_series_max_throughput_comparison.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")

    def create_scaling_efficiency_comparison(self, max_data: pd.DataFrame):
        """Create scaling efficiency comparison."""
        print(f"\nCreating scaling efficiency comparison")
        
        if max_data.empty:
            print("  No data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Configuration setup
        runs = ["B1", "B2", "B3", "B4", "B5"]
        config_indices = list(range(1, 6))
        
        # Plot lines for each runtime
        for runtime in RUNTIME_ORDER:
            if runtime in max_data['runtime'].values:
                runtime_data = max_data[max_data['runtime'] == runtime]
                throughputs = []
                
                for run in runs:
                    run_throughput = runtime_data[runtime_data['run'] == run]['max_throughput']
                    throughputs.append(run_throughput.iloc[0] if not run_throughput.empty else 0)
                
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                ax.plot(config_indices, throughputs, 
                       marker='o', linewidth=3, markersize=8,
                       label=label, color=color, alpha=0.9)
        
        ax.set_xlabel('Configuration Index (B1 → B5)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Maximum Throughput (RPS)', fontsize=12, fontweight='bold')
        ax.set_title('B-Series Scaling Efficiency Analysis\nThroughput Changes Across Pod Configurations', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(config_indices)
        ax.set_xticklabels(runs)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add configuration strategy annotation
        config_text = "Configuration Strategy:\n"
        config_text += "B1: 1 pod × 1000m CPU = 1.0 vCPU total\n"
        config_text += "B2: 2 pods × 500m CPU = 1.0 vCPU total\n"
        config_text += "B3: 4 pods × 250m CPU = 1.0 vCPU total\n"
        config_text += "B4: 8 pods × 125m CPU = 1.0 vCPU total\n"
        config_text += "B5: 10 pods × 100m CPU = 1.0 vCPU total"
        
        ax.text(0.98, 0.02, config_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        
        plt.tight_layout()
        
        filename = "b_series_scaling_efficiency_comparison.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")

    def create_runtime_scaling_trends(self, max_data: pd.DataFrame):
        """Create individual runtime scaling trend analysis."""
        print(f"\nCreating runtime scaling trends")
        
        if max_data.empty:
            print("  No data available")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('B-Series Individual Runtime Scaling Trends\nConfiguration Impact on Each Runtime', 
                     fontsize=16, fontweight='bold')
        
        runs = ["B1", "B2", "B3", "B4", "B5"]
        config_indices = list(range(1, 6))
        
        for i, runtime in enumerate(RUNTIME_ORDER):
            if runtime in max_data['runtime'].values:
                ax = axes[i]
                runtime_data = max_data[max_data['runtime'] == runtime]
                throughputs = []
                
                for run in runs:
                    run_throughput = runtime_data[runtime_data['run'] == run]['max_throughput']
                    throughputs.append(run_throughput.iloc[0] if not run_throughput.empty else 0)
                
                bars = ax.bar(config_indices, throughputs, alpha=0.8)
                
                # Add value labels
                for bar, value in zip(bars, throughputs):
                    if value > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                               f'{value:.0f}', ha='center', va='bottom', 
                               fontsize=10, fontweight='bold')
                
                runtime_label = RUNTIME_COLORS[runtime]['label']
                ax.set_title(f'{runtime_label} Scaling Pattern', fontweight='bold', fontsize=14)
                ax.set_xlabel('Configuration')
                ax.set_ylabel('Max Throughput (RPS)')
                ax.set_xticks(config_indices)
                ax.set_xticklabels(runs)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Calculate and show trend
                if len(throughputs) >= 2:
                    b1_val = throughputs[0] if throughputs[0] > 0 else 1
                    b5_val = throughputs[-1] if throughputs[-1] > 0 else 1
                    change_pct = ((b5_val - b1_val) / b1_val) * 100
                    
                    trend_text = f"B1→B5: {change_pct:+.1f}%"
                    if change_pct > 5:
                        trend_color = "green"
                    elif change_pct < -5:
                        trend_color = "red"
                    else:
                        trend_color = "orange"
                    
                    ax.text(0.02, 0.98, trend_text, transform=ax.transAxes,
                           verticalalignment='top', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=trend_color, alpha=0.7))
        
        plt.tight_layout()
        
        filename = "b_series_runtime_scaling_trends.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")

    # =========================================================================
    # RUNTIME-SPECIFIC CHART GENERATION (LIKE COMPARE SCRIPTS)
    # =========================================================================

    def create_runtime_specific_charts(self, all_runtime_data: Dict[str, pd.DataFrame]):
        """Create runtime-specific charts for each runtime (like compare scripts)."""
        print(f"\nCreating runtime-specific charts for each runtime")
        
        for runtime in RUNTIME_ORDER:
            if runtime in all_runtime_data and runtime in self.runtime_dirs:
                print(f"\n  Generating charts for {RUNTIME_COLORS[runtime]['label']}...")
                self._create_runtime_charts(runtime, all_runtime_data[runtime])

    def _create_runtime_charts(self, runtime: str, runtime_data: pd.DataFrame):
        """Create all charts for a specific runtime."""
        runtime_dir = self.runtime_dirs[runtime]
        runtime_label = RUNTIME_COLORS[runtime]['label']
        
        # Group by run and target_rps for aggregation
        grouped = runtime_data.groupby(['run', 'target_rps']).agg({
            'average': 'mean',
            'median': 'mean',
            'p95': 'mean',
            'p99': 'mean',
            'throughput_rps': 'mean',
            'actual_rps': 'mean'
        }).reset_index()
        
        # 1. Latency vs Target RPS charts
        self._create_latency_vs_target_charts(runtime, grouped, runtime_dir, runtime_label)
        
        # 2. Latency vs Throughput charts  
        self._create_latency_vs_throughput_charts(runtime, grouped, runtime_dir, runtime_label)
        
        # 3. Timeseries comparison charts
        self._create_timeseries_charts(runtime, runtime_dir, runtime_label)
        
        # 4. Generate aggregated CSV data (like compare scripts)
        self._create_aggregated_csv(runtime, runtime_data, runtime_dir, runtime_label)

    def _create_latency_vs_target_charts(self, runtime: str, grouped_data: pd.DataFrame, runtime_dir: str, runtime_label: str):
        """Create latency vs target RPS charts for a specific runtime."""
        metrics = [
            ('average', 'Average'),
            ('median', 'Median'),
            ('p95', 'P95'),
            ('p99', 'P99')
        ]
        
        for metric_col, metric_label in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for run in ["B1", "B2", "B3", "B4", "B5"]:
                run_data = grouped_data[grouped_data['run'] == run]
                if not run_data.empty:
                    config = RUN_CONFIG[run]
                    label = f"{run} | pods={config['pods']} | CPU={config['container_cpu']}"
                    
                    # Convert microseconds to milliseconds
                    y_values = run_data[metric_col].values / 1000
                    ax.plot(run_data['target_rps'], y_values, marker='o', label=label, linewidth=2)
            
            ax.set_xlabel('Target RPS', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric_label} Latency (ms)', fontsize=12, fontweight='bold')
            ax.set_title(f'{runtime_label} B-Series: {metric_label} Latency vs Target RPS\nPod Scaling Strategy Analysis', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f"{runtime}_b_series_latency_{metric_col}_vs_target.png"
            output_path = os.path.join(runtime_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved: {filename}")

    def _create_latency_vs_throughput_charts(self, runtime: str, grouped_data: pd.DataFrame, runtime_dir: str, runtime_label: str):
        """Create latency vs throughput charts for a specific runtime."""
        metrics = [
            ('average', 'Average'),
            ('median', 'Median'),
            ('p95', 'P95'),
            ('p99', 'P99')
        ]
        
        for metric_col, metric_label in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for run in ["B1", "B2", "B3", "B4", "B5"]:
                run_data = grouped_data[grouped_data['run'] == run]
                if not run_data.empty:
                    config = RUN_CONFIG[run]
                    label = f"{run} | pods={config['pods']} | CPU={config['container_cpu']}"
                    
                    # Convert microseconds to milliseconds
                    y_values = run_data[metric_col].values / 1000
                    # Use actual_rps if available, otherwise throughput_rps
                    x_values = run_data['actual_rps'].fillna(run_data['throughput_rps']).values
                    ax.plot(x_values, y_values, marker='o', label=label, linewidth=2)
            
            ax.set_xlabel('Achieved Throughput (RPS)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric_label} Latency (ms)', fontsize=12, fontweight='bold')
            ax.set_title(f'{runtime_label} B-Series: {metric_label} Latency vs Achieved Throughput\nPod Scaling Strategy Analysis', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f"{runtime}_b_series_latency_{metric_col}_vs_throughput.png"
            output_path = os.path.join(runtime_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved: {filename}")

    def _create_timeseries_charts(self, runtime: str, runtime_dir: str, runtime_label: str):
        """Create timeseries comparison charts for a specific runtime."""
        # Find all B-series directories for this runtime
        runtime_pattern = f"results_fibonacci_{runtime}_b"
        runtime_dirs = [d for d in os.listdir(self.base_dir) if d.startswith(runtime_pattern)]
        
        if not runtime_dirs:
            print(f"    No B-series directories found for {runtime}")
            return
        
        # Read pod monitoring data for each run
        timeseries_data = {}
        for run_dir in runtime_dirs:
            run_name = run_dir.split('_')[-1].upper()  # Extract B1, B2, etc.
            pod_csv_path = os.path.join(self.base_dir, run_dir, f"fibonacci_{runtime}_{run_name.lower()}_pod_monitoring.csv")
            
            if not os.path.exists(pod_csv_path):
                # Try alternative naming patterns
                candidates = [
                    os.path.join(self.base_dir, run_dir, f)
                    for f in os.listdir(os.path.join(self.base_dir, run_dir))
                    if f.endswith("_pod_monitoring.csv")
                ]
                if candidates:
                    pod_csv_path = candidates[0]
                else:
                    continue
            
            try:
                df = pd.read_csv(pod_csv_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                timeseries_data[run_name] = df
            except Exception as e:
                print(f"    Failed to read pod monitoring data for {run_name}: {e}")
                continue
        
        if not timeseries_data:
            print(f"    No timeseries data available for {runtime}")
            return
        
        # Create CPU usage comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for run_name, data in timeseries_data.items():
            if 'cpu_usage_millicores' in data.columns:
                config = RUN_CONFIG.get(run_name, {})
                label = f"{run_name} | pods={config.get('pods', '?')}"
                
                # Normalize time to 0-100% for comparison
                time_normalized = np.linspace(0, 100, len(data))
                ax.plot(time_normalized, data['cpu_usage_millicores'], 
                       label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('CPU Usage (millicores)', fontsize=12, fontweight='bold')
        ax.set_title(f'{runtime_label} B-Series: CPU Usage Comparison Over Time\nPod Scaling Strategy Analysis', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{runtime}_b_series_timeseries_cpu_comparison.png"
        output_path = os.path.join(runtime_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filename}")
        
        # Create Memory usage comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for run_name, data in timeseries_data.items():
            if 'memory_usage_mib' in data.columns:
                config = RUN_CONFIG.get(run_name, {})
                label = f"{run_name} | pods={config.get('pods', '?')}"
                
                time_normalized = np.linspace(0, 100, len(data))
                ax.plot(time_normalized, data['memory_usage_mib'], 
                       label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Usage (MiB)', fontsize=12, fontweight='bold')
        ax.set_title(f'{runtime_label} B-Series: Memory Usage Comparison Over Time\nPod Scaling Strategy Analysis', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{runtime}_b_series_timeseries_memory_comparison.png"
        output_path = os.path.join(runtime_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filename}")
        
        # Create Pod count comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for run_name, data in timeseries_data.items():
            if 'pod_count' in data.columns:
                config = RUN_CONFIG.get(run_name, {})
                label = f"{run_name} | pods={config.get('pods', '?')}"
                
                time_normalized = np.linspace(0, 100, len(data))
                ax.plot(time_normalized, data['pod_count'], 
                       label=label, linewidth=2, alpha=0.8, marker='o', markersize=4)
        
        ax.set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pod Count', fontsize=12, fontweight='bold')
        ax.set_title(f'{runtime_label} B-Series: Pod Count Comparison Over Time\nPod Scaling Strategy Analysis', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{runtime}_b_series_timeseries_pod_count_comparison.png"
        output_path = os.path.join(runtime_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filename}")

    def _create_aggregated_csv(self, runtime: str, runtime_data: pd.DataFrame, runtime_dir: str, runtime_label: str):
        """Create aggregated CSV data file for a specific runtime (like compare scripts)."""
        print(f"    Generating aggregated CSV data...")
        
        try:
            # Group by run and target_rps for aggregation
            grouped = runtime_data.groupby(['run', 'target_rps']).agg({
                'average': 'mean',
                'median': 'mean',
                'p95': 'mean',
                'p99': 'mean',
                'throughput_rps': 'mean',
                'actual_rps': 'mean',
                'total_requests': 'mean'
            }).reset_index()
            
            # Add configuration details
            grouped['pods'] = grouped['run'].map(lambda x: RUN_CONFIG[x]['pods'])
            grouped['container_cpu'] = grouped['run'].map(lambda x: RUN_CONFIG[x]['container_cpu'])
            grouped['container_mem'] = grouped['run'].map(lambda x: RUN_CONFIG[x]['container_mem'])
            grouped['target_conc'] = grouped['run'].map(lambda x: RUN_CONFIG[x]['target_conc'])
            
            # Convert latency from microseconds to milliseconds
            latency_columns = ['average', 'median', 'p95', 'p99']
            for col in latency_columns:
                if col in grouped.columns:
                    grouped[f'{col}_ms'] = grouped[col] / 1000
            
            # Reorder columns for better readability
            column_order = [
                'run', 'pods', 'container_cpu', 'container_mem', 'target_conc',
                'target_rps', 'throughput_rps', 'actual_rps', 'total_requests',
                'average', 'average_ms', 'median', 'median_ms', 'p95', 'p95_ms', 'p99', 'p99_ms'
            ]
            
            # Filter to only include columns that exist
            existing_columns = [col for col in column_order if col in grouped.columns]
            grouped = grouped[existing_columns]
            
            # Sort by run (B1, B2, B3, B4, B5) then by target_rps
            grouped['run_order'] = grouped['run'].map({'B1': 1, 'B2': 2, 'B3': 3, 'B4': 4, 'B5': 5})
            grouped = grouped.sort_values(['run_order', 'target_rps']).drop('run_order', axis=1)
            
            # Save aggregated CSV
            filename = f"fibonacci_{runtime}_b_aggregated.csv"
            output_path = os.path.join(runtime_dir, filename)
            grouped.to_csv(output_path, index=False)
            
            print(f"      Saved: {filename}")
            
            # Also save a summary version with key metrics - FIXED to use individual iteration maximums
            summary_data = []
            for run in ["B1", "B2", "B3", "B4", "B5"]:
                # Find the results directory for this run
                run_dir = None
                for item in os.listdir(self.base_dir):
                    if item.startswith(f'results_fibonacci_{runtime}_') and item.endswith(f'_{run.lower()}'):
                        run_dir = os.path.join(self.base_dir, item)
                        break
                
                if run_dir and os.path.exists(run_dir):
                    # Read the achieved_rps_summary.csv directly to get individual iteration maximums
                    summary_file = os.path.join(run_dir, 'achieved_rps_summary.csv')
                    if os.path.exists(summary_file):
                        try:
                            with open(summary_file, 'r') as f:
                                lines = f.readlines()
                            
                            # Extract the "Overall Max" value from the summary section
                            overall_max = None
                            target_at_max = None
                            
                            for line in lines:
                                if line.startswith('Overall Max'):
                                    parts = line.strip().split(',')
                                    if len(parts) >= 2:
                                        overall_max = float(parts[1])
                                        target_at_max = float(parts[2]) if len(parts) > 2 else None
                                        break
                            
                            if overall_max is not None:
                                # Get the corresponding row from grouped data for latency metrics
                                run_data = grouped[grouped['run'] == run]
                                if not run_data.empty:
                                    # Find the row closest to the target RPS at max
                                    if target_at_max is not None:
                                        closest_idx = (run_data['target_rps'] - target_at_max).abs().idxmin()
                                    else:
                                        closest_idx = run_data['actual_rps'].idxmax()
                                    
                                    max_throughput_row = run_data.loc[closest_idx]
                                    
                                    summary_data.append({
                                        'Configuration': run,
                                        'Pods': max_throughput_row['pods'],
                                        'CPU_per_Pod': max_throughput_row['container_cpu'],
                                        'Memory_per_Pod': max_throughput_row['container_mem'],
                                        'Max_Throughput_RPS': f"{overall_max:.1f}",
                                        'Target_RPS_at_Max': f"{target_at_max:.0f}" if target_at_max else "N/A",
                                        'Avg_Latency_ms': f"{max_throughput_row.get('average_ms', max_throughput_row['average']/1000):.2f}",
                                        'P95_Latency_ms': f"{max_throughput_row.get('p95_ms', max_throughput_row['p95']/1000):.2f}",
                                        'P99_Latency_ms': f"{max_throughput_row.get('p99_ms', max_throughput_row['p99']/1000):.2f}"
                                    })
                                else:
                                    print(f"      Warning: No grouped data found for {run}")
                            else:
                                print(f"      Warning: Could not extract Overall Max from {summary_file}")
                                
                        except Exception as e:
                            print(f"      Error reading {summary_file}: {e}")
                    else:
                        print(f"      Warning: {summary_file} not found")
                else:
                    print(f"      Warning: Results directory for {run} not found")
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_filename = f"fibonacci_{runtime}_b_summary.csv"
                summary_path = os.path.join(runtime_dir, summary_filename)
                summary_df.to_csv(summary_path, index=False)
                print(f"      Saved: {summary_filename}")
            else:
                print(f"      Warning: No summary data generated for {runtime}")
                
        except Exception as e:
            print(f"      Error generating aggregated CSV: {e}")

    # =========================================================================
    # MAIN EXECUTION FUNCTION
    # =========================================================================

    def generate_all_b_series_visualizations(self):
        """Generate all B-series visualizations."""
        print("\nStarting B-series visualization generation...")
        
        # Find B-series directories
        b_series_dirs = self.find_b_series_directories()
        
        if not any(b_series_dirs.values()):
            print("No B-series result directories found")
            return
        
        print(f"\nFound B-series directories:")
        for runtime in RUNTIME_ORDER:
            if b_series_dirs[runtime]:
                runtime_label = RUNTIME_COLORS[runtime]['label']
                print(f"  {runtime_label}: {len(b_series_dirs[runtime])} runs")
                for dir_path in b_series_dirs[runtime]:
                    print(f"    - {os.path.basename(dir_path)}")
        
        # Load runtime data
        all_runtime_data = self.load_runtime_data(b_series_dirs)
        
        if len(all_runtime_data) < 1:
            print("No valid B-series data found")
            return
        
        # Extract maximum throughput data
        max_data = self.extract_maximum_throughput_data(all_runtime_data)
        
        if max_data.empty:
            print("No maximum throughput data available")
            return
        
        print(f"\n{'='*60}")
        print(f"GENERATING B-SERIES COMPARISONS")
        print(f"{'='*60}")
        
        # Generate centralized B-series comparison charts
        self.create_maximum_throughput_comparison(max_data)
        self.create_scaling_efficiency_comparison(max_data)
        self.create_runtime_scaling_trends(max_data)
        
        # Generate runtime-specific charts (like compare scripts)
        self.create_runtime_specific_charts(all_runtime_data)
        

        
        print(f"\nALL B-SERIES VISUALIZATIONS COMPLETED!")
        print(f"Output directory: {self.output_dir}")
        print(f"Focus: Pod scaling strategy analysis")



# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate B-series visualizations')
    parser.add_argument('--base-dir', default='.', help='Base directory containing results')
    
    args = parser.parse_args()
    
    # Create and run visualizer
    visualizer = BSeriesVisualizer(args.base_dir)
    visualizer.generate_all_b_series_visualizations()

if __name__ == "__main__":
    main()
