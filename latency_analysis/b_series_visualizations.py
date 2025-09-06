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

# Y-axis scaling factors and limits for enhanced detail
Y_AXIS_SCALING = {
    "throughput": 0.7,      # Scale down throughput charts
    "latency": 500,         # Fixed 500ms max for latency charts
    "cpu": 0.6,             # Scale down CPU charts
    "memory": 0.7,          # Scale down memory charts
    "pod_count": 0.9        # Scale down pod count charts
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
        print("B-SERIES VISUALIZATIONS (ENHANCED DETAIL VIEW)")
        print("=" * 60)
        print(f"Base Directory: {base_dir}")
        print(f"Central charts: {self.output_dir}")
        print(f"Runtime charts: {', '.join([f'{runtime}_b_series' for runtime in RUNTIME_ORDER])}")
        print(f"Focus: Pod scaling strategy analysis (containerConcurrency=100)")
        print(f"Runtime Order: {' → '.join([RUNTIME_COLORS[r]['label'] for r in RUNTIME_ORDER])}")
        print(f"Y-axis scaling: Enhanced detail visibility with lower values")
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
                        # Merge and aggregate - Use avg_achieved (typical performance across iterations)
                        merged = pd.merge(detailed_df, achieved_df[['target_rps', 'avg_achieved']], 
                                        on='target_rps', how='left')
                        merged['actual_rps'] = merged['avg_achieved'].fillna(merged['throughput_rps'])
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
        
        # Need to go back to original directories to read "Avg of Max" from summary files
        b_series_dirs = self.find_b_series_directories()
        
        for runtime in ["python", "nodejs", "go"]:
            if runtime in b_series_dirs and b_series_dirs[runtime]:
                for results_dir in b_series_dirs[runtime]:
                    # Extract run name (B1, B2, etc.)
                    run_name = os.path.basename(results_dir).split('_')[-1].upper()
                    
                    # Read the "Avg of Max" from achieved_rps_summary.csv
                    summary_file = os.path.join(results_dir, 'achieved_rps_summary.csv')
                    if os.path.exists(summary_file):
                        try:
                            with open(summary_file, 'r') as f:
                                lines = f.readlines()
                            
                            # Look for "Avg of Max" line
                            avg_of_max = None
                            for line in lines:
                                if line.startswith('Avg of Max'):
                                    parts = line.strip().split(',')
                                    if len(parts) >= 2:
                                        avg_of_max = float(parts[1])
                                        break
                            
                            if avg_of_max is not None:
                                config = RUN_CONFIG[run_name]
                                max_data.append({
                                    'runtime': runtime,
                                    'run': run_name,
                                    'max_throughput': avg_of_max,
                                    'pods': int(config['pods']),
                                    'cpu': config['container_cpu'],
                                    'memory': config['container_mem'],
                                    'target_conc': int(config['target_conc'])
                                })
                            else:
                                print(f"Warning: Could not find 'Avg of Max' in {summary_file}")
                                
                        except Exception as e:
                            print(f"Error reading {summary_file}: {e}")
                    else:
                        print(f"Warning: {summary_file} not found")
        
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
        
        # Set reasonable y-axis limit with some headroom
        max_throughput = max_data['max_throughput'].max()
        y_limit = max_throughput * 1.1  # 10% headroom
        ax.set_ylim(0, y_limit)
        
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
        
        # Apply lower y-axis scaling for enhanced detail
        max_throughput = max_data['max_throughput'].max()
        y_limit = max_throughput * Y_AXIS_SCALING['throughput']
        ax.set_ylim(0, y_limit)
        
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
                
                # Apply lower y-axis scaling for enhanced detail
                if throughputs:
                    max_throughput = max(throughputs)
                    y_limit = max_throughput * Y_AXIS_SCALING['throughput']
                    ax.set_ylim(0, y_limit)
                
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
            ax.set_title(f'{runtime_label} B-Series: {metric_label} Latency vs Target RPS\nPod Scaling Strategy Analysis - Enhanced Detail View', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            
            # Apply fixed 500ms y-axis limit for enhanced detail
            ax.set_ylim(0, Y_AXIS_SCALING['latency'])
            
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
            ax.set_title(f'{runtime_label} B-Series: {metric_label} Latency vs Achieved Throughput\nPod Scaling Strategy Analysis - Enhanced Detail View', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            
            # Apply fixed 500ms y-axis limit for enhanced detail
            ax.set_ylim(0, Y_AXIS_SCALING['latency'])
            
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
        ax.set_title(f'{runtime_label} B-Series: CPU Usage Comparison Over Time\nPod Scaling Strategy Analysis - Enhanced Detail View', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Apply lower y-axis scaling for enhanced detail
        if timeseries_data:
            max_cpu = max([data['cpu_usage_millicores'].max() for data in timeseries_data.values() 
                          if 'cpu_usage_millicores' in data.columns])
            y_limit = max_cpu * Y_AXIS_SCALING['cpu']
            ax.set_ylim(0, y_limit)
        
        plt.tight_layout()
        
        filename = f"{runtime}_b_series_timeseries_cpu_comparison.png"
        output_path = os.path.join(runtime_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filename}")
        
        # Create CPU usage per-pod comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for run_name, data in timeseries_data.items():
            if 'cpu_usage_millicores' in data.columns:
                config = RUN_CONFIG.get(run_name, {})
                num_pods = int(config.get('pods', 1))
                
                # Calculate per-pod CPU usage
                cpu_per_pod = data['cpu_usage_millicores'] / num_pods
                
                label = f"{run_name} | pods={config.get('pods', '?')} | per pod"
                
                # Normalize time to 0-100% for comparison
                time_normalized = np.linspace(0, 100, len(data))
                ax.plot(time_normalized, cpu_per_pod, 
                       label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('CPU Usage per Pod (millicores)', fontsize=12, fontweight='bold')
        ax.set_title(f'{runtime_label} B-Series: CPU Usage per Pod Comparison Over Time\nPod Scaling Strategy Analysis - Per-Pod Resource Utilization', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Apply lower y-axis scaling for enhanced detail
        if timeseries_data:
            max_cpu_per_pod = max([(data['cpu_usage_millicores'] / int(RUN_CONFIG.get(run_name, {}).get('pods', 1))).max() 
                                 for run_name, data in timeseries_data.items() 
                                 if 'cpu_usage_millicores' in data.columns])
            y_limit = max_cpu_per_pod * Y_AXIS_SCALING['cpu']
            ax.set_ylim(0, y_limit)
        
        plt.tight_layout()
        
        filename = f"{runtime}_b_series_timeseries_cpu_per_pod_comparison.png"
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
        ax.set_title(f'{runtime_label} B-Series: Memory Usage Comparison Over Time\nPod Scaling Strategy Analysis - Enhanced Detail View', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Apply lower y-axis scaling for enhanced detail
        if timeseries_data:
            max_memory = max([data['memory_usage_mib'].max() for data in timeseries_data.values() 
                             if 'memory_usage_mib' in data.columns])
            y_limit = max_memory * Y_AXIS_SCALING['memory']
            ax.set_ylim(0, y_limit)
        
        plt.tight_layout()
        
        filename = f"{runtime}_b_series_timeseries_memory_comparison.png"
        output_path = os.path.join(runtime_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filename}")
        
        # Create Memory usage per-pod comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for run_name, data in timeseries_data.items():
            if 'memory_usage_mib' in data.columns:
                config = RUN_CONFIG.get(run_name, {})
                num_pods = int(config.get('pods', 1))
                
                # Calculate per-pod memory usage
                memory_per_pod = data['memory_usage_mib'] / num_pods
                
                label = f"{run_name} | pods={config.get('pods', '?')} | per pod"
                
                time_normalized = np.linspace(0, 100, len(data))
                ax.plot(time_normalized, memory_per_pod, 
                       label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Usage per Pod (MiB)', fontsize=12, fontweight='bold')
        ax.set_title(f'{runtime_label} B-Series: Memory Usage per Pod Comparison Over Time\nPod Scaling Strategy Analysis - Per-Pod Resource Utilization', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Apply lower y-axis scaling for enhanced detail
        if timeseries_data:
            max_memory_per_pod = max([(data['memory_usage_mib'] / int(RUN_CONFIG.get(run_name, {}).get('pods', 1))).max() 
                                    for run_name, data in timeseries_data.items() 
                                    if 'memory_usage_mib' in data.columns])
            y_limit = max_memory_per_pod * Y_AXIS_SCALING['memory']
            ax.set_ylim(0, y_limit)
        
        plt.tight_layout()
        
        filename = f"{runtime}_b_series_timeseries_memory_per_pod_comparison.png"
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
        ax.set_title(f'{runtime_label} B-Series: Pod Count Comparison Over Time\nPod Scaling Strategy Analysis - Enhanced Detail View', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Apply lower y-axis scaling for enhanced detail
        if timeseries_data:
            max_pods = max([data['pod_count'].max() for data in timeseries_data.values() 
                           if 'pod_count' in data.columns])
            y_limit = max_pods * Y_AXIS_SCALING['pod_count']
            ax.set_ylim(0, y_limit)
        
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
    # COMPREHENSIVE RESOURCE SCALING ANALYSIS
    # =========================================================================

    def create_comprehensive_resource_scaling_analysis(self, all_runtime_data=None):
        """Create comprehensive resource scaling analysis across all runtimes and configurations."""
        print(f"\nCreating comprehensive resource scaling analysis")
        
        # Load resource data for all runtimes and configurations
        resource_data = self._load_all_resource_data()
        
        if not resource_data:
            print("  No resource data available")
            return
        
        # Create configuration-aggregated analysis
        if all_runtime_data:
            self._create_configuration_aggregated_analysis(all_runtime_data, resource_data)
        
        # Create timeseries-style resource analysis
        self._create_resource_timeseries_comparison(resource_data)
        
        # Create CPU scaling analysis
        self._create_cpu_scaling_analysis(resource_data)
        
        # Create Memory scaling analysis
        self._create_memory_scaling_analysis(resource_data)
        
        # Create Resource efficiency analysis
        self._create_resource_efficiency_analysis(resource_data)
        
        # Create Resource utilization heatmap
        self._create_resource_utilization_heatmap(resource_data)
        
        # Create per-pod resource scaling analysis
        self._create_per_pod_resource_scaling_analysis(resource_data)

    def _load_all_resource_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load resource monitoring data for all runtimes and configurations."""
        resource_data = {}
        
        for runtime in RUNTIME_ORDER:
            resource_data[runtime] = {}
            
            # Find all B-series directories for this runtime
            runtime_pattern = f"results_fibonacci_{runtime}_b"
            runtime_dirs = [d for d in os.listdir(self.base_dir) if d.startswith(runtime_pattern)]
            
            for run_dir in sorted(runtime_dirs):
                run_name = run_dir.split('_')[-1].upper()  # Extract B1, B2, etc.
                
                # Find pod monitoring file
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
                        df = df.sort_values('timestamp')
                    
                    # Calculate resource metrics per pod
                    if 'pod_count' in df.columns and 'cpu_usage_millicores' in df.columns:
                        df['cpu_per_pod'] = df['cpu_usage_millicores'] / df['pod_count']
                    if 'pod_count' in df.columns and 'memory_usage_mib' in df.columns:
                        df['memory_per_pod'] = df['memory_usage_mib'] / df['pod_count']
                    
                    resource_data[runtime][run_name] = df
                    print(f"  Loaded {runtime} {run_name}: {len(df)} data points")
                    
                except Exception as e:
                    print(f"  Failed to load {runtime} {run_name}: {e}")
                    continue
        
        return resource_data

    def _create_configuration_aggregated_analysis(self, all_runtime_data: Dict[str, pd.DataFrame], resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create configuration-aggregated analysis showing each config across all runtimes."""
        print(f"    Creating configuration-aggregated analysis")
        
        # Create charts for each configuration (B1, B2, B3, B4, B5)
        for config in ["B1", "B2", "B3", "B4", "B5"]:
            # Create config-specific directory
            config_dir = os.path.join(self.output_dir, f"b{config[1]}")
            os.makedirs(config_dir, exist_ok=True)
            self._create_config_specific_analysis(config, all_runtime_data, resource_data, config_dir)

    def _create_config_specific_analysis(self, config: str, all_runtime_data: Dict[str, pd.DataFrame], resource_data: Dict[str, Dict[str, pd.DataFrame]], config_dir: str):
        """Create analysis for a specific configuration across all runtimes."""
        print(f"      Creating {config} configuration analysis")
        
        # Create throughput comparison for this config
        self._create_config_throughput_comparison(config, all_runtime_data, config_dir)
        
        # Create latency comparison for this config
        self._create_config_latency_comparison(config, all_runtime_data, config_dir)
        
        # Create resource comparison for this config (aggregated)
        self._create_config_resource_comparison(config, resource_data, config_dir)
        
        # Create per-pod resource comparison for this config
        self._create_config_per_pod_resource_comparison(config, resource_data, config_dir)

    def _create_config_throughput_comparison(self, config: str, all_runtime_data: Dict[str, pd.DataFrame], config_dir: str):
        """Create throughput comparison for a specific configuration across all runtimes."""
        print(f"        Creating {config} throughput comparison")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for runtime in RUNTIME_ORDER:
            if runtime in all_runtime_data:
                data = all_runtime_data[runtime]
                config_data = data[data['run'] == config]
                
                if not config_data.empty:
                    color = RUNTIME_COLORS[runtime]['primary']
                    label = RUNTIME_COLORS[runtime]['label']
                    
                    # Use actual_rps if available, otherwise throughput_rps
                    x_values = config_data['actual_rps'].fillna(config_data['throughput_rps']).values
                    y_values = config_data['target_rps'].values
                    
                    ax.plot(y_values, x_values, 'o-', 
                           color=color, linewidth=3, markersize=8, 
                           label=f"{label} Achieved", alpha=0.8)
        
        # Add 100% efficiency line
        if not all_runtime_data:
            return
        max_target = max([data[data['run'] == config]['target_rps'].max() 
                         for data in all_runtime_data.values() 
                         if not data[data['run'] == config].empty])
        ax.plot([0, max_target], [0, max_target], 'k--', alpha=0.5, label='100% Efficiency')
        
        config_info = RUN_CONFIG[config]
        ax.set_xlabel('Target RPS', fontsize=12, fontweight='bold')
        ax.set_ylabel('Achieved RPS', fontsize=12, fontweight='bold')
        ax.set_title(f'{config} Configuration - Throughput Comparison Across Runtimes\n'
                    f'{config_info["pods"]} pods × {config_info["container_cpu"]} CPU × {config_info["container_mem"]} Memory', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"throughput_comparison.png"
        output_path = os.path.join(config_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"          Saved: {filename}")

    def _create_config_latency_comparison(self, config: str, all_runtime_data: Dict[str, pd.DataFrame], config_dir: str):
        """Create latency comparison for a specific configuration across all runtimes."""
        print(f"        Creating {config} latency comparison")
        
        metrics = [
            ('average', 'Average'),
            ('p95', 'P95'),
            ('p99', 'P99')
        ]
        
        for metric_col, metric_label in metrics:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for runtime in RUNTIME_ORDER:
                if runtime in all_runtime_data:
                    data = all_runtime_data[runtime]
                    config_data = data[data['run'] == config]
                    
                    if not config_data.empty:
                        color = RUNTIME_COLORS[runtime]['primary']
                        label = RUNTIME_COLORS[runtime]['label']
                        
                        # Use actual_rps if available, otherwise throughput_rps
                        x_values = config_data['actual_rps'].fillna(config_data['throughput_rps']).values
                        # Convert microseconds to milliseconds
                        y_values = config_data[metric_col].values / 1000
                        
                        ax.plot(x_values, y_values, 'o-', 
                               color=color, linewidth=3, markersize=8, 
                               label=label, alpha=0.8)
            
            config_info = RUN_CONFIG[config]
            ax.set_xlabel('Achieved Throughput (RPS)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric_label} Latency (ms)', fontsize=12, fontweight='bold')
            ax.set_title(f'{config} Configuration - {metric_label} Latency Comparison Across Runtimes\n'
                        f'{config_info["pods"]} pods × {config_info["container_cpu"]} CPU × {config_info["container_mem"]} Memory', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plt.tight_layout()
            
            filename = f"latency_{metric_col}_comparison.png"
            output_path = os.path.join(config_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"          Saved: {filename}")

    def _create_config_resource_comparison(self, config: str, resource_data: Dict[str, Dict[str, pd.DataFrame]], config_dir: str):
        """Create per-pod resource comparison for a specific configuration across all runtimes."""
        print(f"        Creating {config} per-pod resource comparison")
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle(f'{config} Configuration - Per-Pod Resource Usage Comparison Across Runtimes\n'
                    f'{RUN_CONFIG[config]["pods"]} pods × {RUN_CONFIG[config]["container_cpu"]} CPU × {RUN_CONFIG[config]["container_mem"]} Memory', 
                    fontsize=16, fontweight='bold')
        
        # Get the number of pods for this configuration
        num_pods = int(RUN_CONFIG[config]["pods"])
        
        # CPU Usage per Pod
        ax1 = axes[0]
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data and config in resource_data[runtime]:
                data = resource_data[runtime][config]
                if 'cpu_usage_millicores' in data.columns and 'timestamp' in data.columns:
                    color = RUNTIME_COLORS[runtime]['primary']
                    label = RUNTIME_COLORS[runtime]['label']
                    
                    # Calculate per-pod CPU usage
                    cpu_per_pod = data['cpu_usage_millicores'] / num_pods
                    
                    # Normalize time to 0-100% for comparison
                    time_normalized = np.linspace(0, 100, len(data))
                    ax1.plot(time_normalized, cpu_per_pod, 
                           color=color, linewidth=2, label=label, alpha=0.8)
        
        # Add theoretical CPU limit per pod
        theoretical_cpu_per_pod = int(RUN_CONFIG[config]["container_cpu"].replace('m', ''))
        ax1.axhline(y=theoretical_cpu_per_pod, color='red', linestyle='--', alpha=0.7, 
                   label=f'Theoretical Limit ({theoretical_cpu_per_pod}m)')
        
        ax1.set_ylabel('CPU Usage per Pod (millicores)', fontsize=12, fontweight='bold')
        ax1.set_title('CPU Usage per Pod Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        # Memory Usage per Pod
        ax2 = axes[1]
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data and config in resource_data[runtime]:
                data = resource_data[runtime][config]
                if 'memory_usage_mib' in data.columns and 'timestamp' in data.columns:
                    color = RUNTIME_COLORS[runtime]['primary']
                    label = RUNTIME_COLORS[runtime]['label']
                    
                    # Calculate per-pod memory usage
                    memory_per_pod = data['memory_usage_mib'] / num_pods
                    
                    time_normalized = np.linspace(0, 100, len(data))
                    ax2.plot(time_normalized, memory_per_pod, 
                           color=color, linewidth=2, label=label, alpha=0.8)
        
        # Add theoretical memory limit per pod (convert to MiB)
        memory_limit_str = RUN_CONFIG[config]["container_mem"]
        if memory_limit_str.endswith('Gi'):
            theoretical_memory_per_pod = int(memory_limit_str.replace('Gi', '')) * 1024
        elif memory_limit_str.endswith('Mi'):
            theoretical_memory_per_pod = int(memory_limit_str.replace('Mi', ''))
        else:
            theoretical_memory_per_pod = 0
        
        if theoretical_memory_per_pod > 0:
            ax2.axhline(y=theoretical_memory_per_pod, color='red', linestyle='--', alpha=0.7, 
                       label=f'Theoretical Limit ({memory_limit_str})')
        
        ax2.set_ylabel('Memory Usage per Pod (MiB)', fontsize=12, fontweight='bold')
        ax2.set_title('Memory Usage per Pod Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        # Pod Count
        ax3 = axes[2]
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data and config in resource_data[runtime]:
                data = resource_data[runtime][config]
                if 'pod_count' in data.columns and 'timestamp' in data.columns:
                    color = RUNTIME_COLORS[runtime]['primary']
                    label = RUNTIME_COLORS[runtime]['label']
                    
                    time_normalized = np.linspace(0, 100, len(data))
                    ax3.plot(time_normalized, data['pod_count'], 
                           color=color, linewidth=2, label=label, alpha=0.8, 
                           marker='o', markersize=3)
        
        ax3.set_ylabel('Pod Count', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Pod Count Over Time', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"resource_comparison.png"
        output_path = os.path.join(config_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"          Saved: {filename}")

    def _create_config_per_pod_resource_comparison(self, config: str, resource_data: Dict[str, Dict[str, pd.DataFrame]], config_dir: str):
        """Create per-pod resource comparison for a specific configuration across all runtimes."""
        print(f"        Creating {config} per-pod resource comparison")
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle(f'{config} Configuration - Per-Pod Resource Usage Comparison Across Runtimes\n'
                    f'{RUN_CONFIG[config]["pods"]} pods × {RUN_CONFIG[config]["container_cpu"]} CPU × {RUN_CONFIG[config]["container_mem"]} Memory', 
                    fontsize=16, fontweight='bold')
        
        # Get the number of pods for this configuration
        num_pods = int(RUN_CONFIG[config]["pods"])
        
        # CPU Usage per Pod
        ax1 = axes[0]
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data and config in resource_data[runtime]:
                data = resource_data[runtime][config]
                if 'cpu_usage_millicores' in data.columns and 'timestamp' in data.columns:
                    color = RUNTIME_COLORS[runtime]['primary']
                    label = RUNTIME_COLORS[runtime]['label']
                    
                    # Calculate per-pod CPU usage
                    cpu_per_pod = data['cpu_usage_millicores'] / num_pods
                    
                    # Normalize time to 0-100% for comparison
                    time_normalized = np.linspace(0, 100, len(data))
                    ax1.plot(time_normalized, cpu_per_pod, 
                           color=color, linewidth=2, label=label, alpha=0.8)
        
        # Add theoretical CPU limit per pod
        theoretical_cpu_per_pod = int(RUN_CONFIG[config]["container_cpu"].replace('m', ''))
        ax1.axhline(y=theoretical_cpu_per_pod, color='red', linestyle='--', alpha=0.7, 
                   label=f'Theoretical Limit ({theoretical_cpu_per_pod}m)')
        
        ax1.set_ylabel('CPU Usage per Pod (millicores)', fontsize=12, fontweight='bold')
        ax1.set_title('CPU Usage per Pod Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        # Memory Usage per Pod
        ax2 = axes[1]
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data and config in resource_data[runtime]:
                data = resource_data[runtime][config]
                if 'memory_usage_mib' in data.columns and 'timestamp' in data.columns:
                    color = RUNTIME_COLORS[runtime]['primary']
                    label = RUNTIME_COLORS[runtime]['label']
                    
                    # Calculate per-pod memory usage
                    memory_per_pod = data['memory_usage_mib'] / num_pods
                    
                    time_normalized = np.linspace(0, 100, len(data))
                    ax2.plot(time_normalized, memory_per_pod, 
                           color=color, linewidth=2, label=label, alpha=0.8)
        
        # Add theoretical memory limit per pod (convert to MiB)
        memory_limit_str = RUN_CONFIG[config]["container_mem"]
        if memory_limit_str.endswith('Gi'):
            theoretical_memory_per_pod = int(memory_limit_str.replace('Gi', '')) * 1024
        elif memory_limit_str.endswith('Mi'):
            theoretical_memory_per_pod = int(memory_limit_str.replace('Mi', ''))
        else:
            theoretical_memory_per_pod = 0
        
        if theoretical_memory_per_pod > 0:
            ax2.axhline(y=theoretical_memory_per_pod, color='red', linestyle='--', alpha=0.7, 
                       label=f'Theoretical Limit ({memory_limit_str})')
        
        ax2.set_ylabel('Memory Usage per Pod (MiB)', fontsize=12, fontweight='bold')
        ax2.set_title('Memory Usage per Pod Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        # Resource Efficiency per Pod
        ax3 = axes[2]
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data and config in resource_data[runtime]:
                data = resource_data[runtime][config]
                if ('cpu_usage_millicores' in data.columns and 'memory_usage_mib' in data.columns 
                    and 'timestamp' in data.columns):
                    color = RUNTIME_COLORS[runtime]['primary']
                    label = RUNTIME_COLORS[runtime]['label']
                    
                    # Calculate resource efficiency (CPU + Memory combined)
                    cpu_per_pod = data['cpu_usage_millicores'] / num_pods
                    memory_per_pod = data['memory_usage_mib'] / num_pods
                    
                    # Calculate efficiency as percentage of theoretical limits
                    cpu_efficiency = (cpu_per_pod / theoretical_cpu_per_pod) * 100
                    memory_efficiency = (memory_per_pod / theoretical_memory_per_pod) * 100 if theoretical_memory_per_pod > 0 else 0
                    
                    # Combined efficiency (average of CPU and Memory)
                    combined_efficiency = (cpu_efficiency + memory_efficiency) / 2
                    
                    time_normalized = np.linspace(0, 100, len(data))
                    ax3.plot(time_normalized, combined_efficiency, 
                           color=color, linewidth=2, label=label, alpha=0.8)
        
        ax3.set_ylabel('Resource Efficiency per Pod (%)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Resource Efficiency per Pod Over Time', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% Efficiency')
        ax3.set_ylim(0, 120)  # Allow some headroom above 100%
        
        plt.tight_layout()
        
        filename = f"per_pod_resource_comparison.png"
        output_path = os.path.join(config_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"          Saved: {filename}")

    def _create_per_pod_resource_scaling_analysis(self, resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create per-pod resource scaling analysis across all configurations."""
        print(f"    Creating per-pod resource scaling analysis")
        
        # Create per-pod CPU scaling analysis
        self._create_per_pod_cpu_scaling_analysis(resource_data)
        
        # Create per-pod Memory scaling analysis
        self._create_per_pod_memory_scaling_analysis(resource_data)

    def _create_per_pod_cpu_scaling_analysis(self, resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create per-pod CPU scaling analysis across all configurations."""
        print(f"      Creating per-pod CPU scaling analysis")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data:
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                configs = []
                avg_cpu_per_pod = []
                theoretical_limits = []
                
                for run in ["B1", "B2", "B3", "B4", "B5"]:
                    if run in resource_data[runtime]:
                        data = resource_data[runtime][run]
                        if 'cpu_usage_millicores' in data.columns and 'pod_count' in data.columns:
                            # Calculate average CPU usage per pod
                            num_pods = int(RUN_CONFIG[run]["pods"])
                            avg_cpu_total = data['cpu_usage_millicores'].mean()
                            avg_cpu_per_pod_val = avg_cpu_total / num_pods
                            
                            configs.append(run)
                            avg_cpu_per_pod.append(avg_cpu_per_pod_val)
                            theoretical_limits.append(int(RUN_CONFIG[run]["container_cpu"].replace('m', '')))
                
                if configs:
                    config_indices = [1, 2, 3, 4, 5]  # B1=1, B2=2, etc.
                    ax.plot(config_indices, avg_cpu_per_pod, 'o-', 
                           color=color, linewidth=3, markersize=8, 
                           label=f"{label} Actual", alpha=0.8)
                    ax.plot(config_indices, theoretical_limits, 's--', 
                           color=color, linewidth=2, markersize=6, 
                           label=f"{label} Theoretical", alpha=0.6)
        
        ax.set_xlabel('Configuration (B1 → B5)', fontsize=12, fontweight='bold')
        ax.set_ylabel('CPU Usage per Pod (millicores)', fontsize=12, fontweight='bold')
        ax.set_title('B-Series Per-Pod CPU Scaling Analysis\nCPU Usage per Pod Across Configurations', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['B1', 'B2', 'B3', 'B4', 'B5'])
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Add configuration details
        config_text = "Configuration Details:\n"
        config_text += "B1: 1 pod × 1000m CPU\n"
        config_text += "B2: 2 pods × 500m CPU\n"
        config_text += "B3: 4 pods × 250m CPU\n"
        config_text += "B4: 8 pods × 125m CPU\n"
        config_text += "B5: 10 pods × 100m CPU"
        
        ax.text(0.02, 0.98, config_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        filename = "b_series_per_pod_cpu_scaling_analysis.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"        Saved: {filename}")

    def _create_per_pod_memory_scaling_analysis(self, resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create per-pod Memory scaling analysis across all configurations."""
        print(f"      Creating per-pod Memory scaling analysis")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data:
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                configs = []
                avg_memory_per_pod = []
                theoretical_limits = []
                
                for run in ["B1", "B2", "B3", "B4", "B5"]:
                    if run in resource_data[runtime]:
                        data = resource_data[runtime][run]
                        if 'memory_usage_mib' in data.columns and 'pod_count' in data.columns:
                            # Calculate average memory usage per pod
                            num_pods = int(RUN_CONFIG[run]["pods"])
                            avg_memory_total = data['memory_usage_mib'].mean()
                            avg_memory_per_pod_val = avg_memory_total / num_pods
                            
                            configs.append(run)
                            avg_memory_per_pod.append(avg_memory_per_pod_val)
                            
                            # Convert theoretical limit to MiB
                            memory_limit_str = RUN_CONFIG[run]["container_mem"]
                            if memory_limit_str.endswith('Gi'):
                                theoretical_limit = int(memory_limit_str.replace('Gi', '')) * 1024
                            elif memory_limit_str.endswith('Mi'):
                                theoretical_limit = int(memory_limit_str.replace('Mi', ''))
                            else:
                                theoretical_limit = 0
                            theoretical_limits.append(theoretical_limit)
                
                if configs:
                    config_indices = [1, 2, 3, 4, 5]  # B1=1, B2=2, etc.
                    ax.plot(config_indices, avg_memory_per_pod, 'o-', 
                           color=color, linewidth=3, markersize=8, 
                           label=f"{label} Actual", alpha=0.8)
                    ax.plot(config_indices, theoretical_limits, 's--', 
                           color=color, linewidth=2, markersize=6, 
                           label=f"{label} Theoretical", alpha=0.6)
        
        ax.set_xlabel('Configuration (B1 → B5)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Usage per Pod (MiB)', fontsize=12, fontweight='bold')
        ax.set_title('B-Series Per-Pod Memory Scaling Analysis\nMemory Usage per Pod Across Configurations', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['B1', 'B2', 'B3', 'B4', 'B5'])
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Add configuration details
        config_text = "Configuration Details:\n"
        config_text += "B1: 1 pod × 1Gi Memory\n"
        config_text += "B2: 2 pods × 500Mi Memory\n"
        config_text += "B3: 4 pods × 250Mi Memory\n"
        config_text += "B4: 8 pods × 125Mi Memory\n"
        config_text += "B5: 10 pods × 100Mi Memory"
        
        ax.text(0.02, 0.98, config_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        
        filename = "b_series_per_pod_memory_scaling_analysis.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"        Saved: {filename}")

    def _create_resource_timeseries_comparison(self, resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create timeseries-style resource comparison across all runtimes and configurations."""
        print(f"    Creating resource timeseries comparison")
        
        # Create CPU timeseries comparison
        self._create_cpu_timeseries_comparison(resource_data)
        
        # Create Memory timeseries comparison
        self._create_memory_timeseries_comparison(resource_data)
        
        # Create Pod count timeseries comparison
        self._create_pod_count_timeseries_comparison(resource_data)

    def _create_cpu_timeseries_comparison(self, resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create CPU timeseries comparison across all runtimes and configurations."""
        print(f"      Creating CPU timeseries comparison")
        
        # Create aggregated version
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle('B-Series CPU Usage Timeseries Comparison (Aggregated)\nResource Scaling Analysis Across All Runtimes and Configurations', 
                     fontsize=16, fontweight='bold')
        
        for i, runtime in enumerate(RUNTIME_ORDER):
            if runtime in resource_data:
                ax = axes[i]
                color = RUNTIME_COLORS[runtime]['primary']
                runtime_label = RUNTIME_COLORS[runtime]['label']
                
                for run in ["B1", "B2", "B3", "B4", "B5"]:
                    if run in resource_data[runtime]:
                        data = resource_data[runtime][run]
                        if 'cpu_usage_millicores' in data.columns and 'timestamp' in data.columns:
                            config = RUN_CONFIG[run]
                            label = f"{run} | {config['pods']} pods | {config['container_cpu']}"
                            
                            # Normalize time to 0-100% for comparison
                            time_normalized = np.linspace(0, 100, len(data))
                            ax.plot(time_normalized, data['cpu_usage_millicores'], 
                                   label=label, linewidth=2, alpha=0.8)
                
                ax.set_ylabel('CPU Usage (millicores)', fontsize=12, fontweight='bold')
                ax.set_title(f'{runtime_label} CPU Usage Over Time', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        axes[-1].set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = "b_series_cpu_timeseries_comparison.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"        Saved: {filename}")
        
        # Create per-pod version
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle('B-Series CPU Usage per Pod Timeseries Comparison\nResource Scaling Analysis Across All Runtimes and Configurations', 
                     fontsize=16, fontweight='bold')
        
        for i, runtime in enumerate(RUNTIME_ORDER):
            if runtime in resource_data:
                ax = axes[i]
                color = RUNTIME_COLORS[runtime]['primary']
                runtime_label = RUNTIME_COLORS[runtime]['label']
                
                for run in ["B1", "B2", "B3", "B4", "B5"]:
                    if run in resource_data[runtime]:
                        data = resource_data[runtime][run]
                        if 'cpu_usage_millicores' in data.columns and 'timestamp' in data.columns:
                            config = RUN_CONFIG[run]
                            num_pods = int(config['pods'])
                            
                            # Calculate per-pod CPU usage
                            cpu_per_pod = data['cpu_usage_millicores'] / num_pods
                            
                            label = f"{run} | {config['pods']} pods | {config['container_cpu']} per pod"
                            
                            # Normalize time to 0-100% for comparison
                            time_normalized = np.linspace(0, 100, len(data))
                            ax.plot(time_normalized, cpu_per_pod, 
                                   label=label, linewidth=2, alpha=0.8)
                
                ax.set_ylabel('CPU Usage per Pod (millicores)', fontsize=12, fontweight='bold')
                ax.set_title(f'{runtime_label} CPU Usage per Pod Over Time', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        axes[-1].set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = "b_series_cpu_per_pod_timeseries_comparison.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"        Saved: {filename}")

    def _create_memory_timeseries_comparison(self, resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create Memory timeseries comparison across all runtimes and configurations."""
        print(f"      Creating Memory timeseries comparison")
        
        # Create aggregated version
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle('B-Series Memory Usage Timeseries Comparison (Aggregated)\nResource Scaling Analysis Across All Runtimes and Configurations', 
                     fontsize=16, fontweight='bold')
        
        for i, runtime in enumerate(RUNTIME_ORDER):
            if runtime in resource_data:
                ax = axes[i]
                color = RUNTIME_COLORS[runtime]['primary']
                runtime_label = RUNTIME_COLORS[runtime]['label']
                
                for run in ["B1", "B2", "B3", "B4", "B5"]:
                    if run in resource_data[runtime]:
                        data = resource_data[runtime][run]
                        if 'memory_usage_mib' in data.columns and 'timestamp' in data.columns:
                            config = RUN_CONFIG[run]
                            label = f"{run} | {config['pods']} pods | {config['container_mem']}"
                            
                            # Normalize time to 0-100% for comparison
                            time_normalized = np.linspace(0, 100, len(data))
                            ax.plot(time_normalized, data['memory_usage_mib'], 
                                   label=label, linewidth=2, alpha=0.8)
                
                ax.set_ylabel('Memory Usage (MiB)', fontsize=12, fontweight='bold')
                ax.set_title(f'{runtime_label} Memory Usage Over Time', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        axes[-1].set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = "b_series_memory_timeseries_comparison.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"        Saved: {filename}")
        
        # Create per-pod version
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle('B-Series Memory Usage per Pod Timeseries Comparison\nResource Scaling Analysis Across All Runtimes and Configurations', 
                     fontsize=16, fontweight='bold')
        
        for i, runtime in enumerate(RUNTIME_ORDER):
            if runtime in resource_data:
                ax = axes[i]
                color = RUNTIME_COLORS[runtime]['primary']
                runtime_label = RUNTIME_COLORS[runtime]['label']
                
                for run in ["B1", "B2", "B3", "B4", "B5"]:
                    if run in resource_data[runtime]:
                        data = resource_data[runtime][run]
                        if 'memory_usage_mib' in data.columns and 'timestamp' in data.columns:
                            config = RUN_CONFIG[run]
                            num_pods = int(config['pods'])
                            
                            # Calculate per-pod memory usage
                            memory_per_pod = data['memory_usage_mib'] / num_pods
                            
                            label = f"{run} | {config['pods']} pods | {config['container_mem']} per pod"
                            
                            # Normalize time to 0-100% for comparison
                            time_normalized = np.linspace(0, 100, len(data))
                            ax.plot(time_normalized, memory_per_pod, 
                                   label=label, linewidth=2, alpha=0.8)
                
                ax.set_ylabel('Memory Usage per Pod (MiB)', fontsize=12, fontweight='bold')
                ax.set_title(f'{runtime_label} Memory Usage per Pod Over Time', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        axes[-1].set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = "b_series_memory_per_pod_timeseries_comparison.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"        Saved: {filename}")

    def _create_pod_count_timeseries_comparison(self, resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create Pod count timeseries comparison across all runtimes and configurations."""
        print(f"      Creating Pod count timeseries comparison")
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle('B-Series Pod Count Timeseries Comparison\nResource Scaling Analysis Across All Runtimes and Configurations', 
                     fontsize=16, fontweight='bold')
        
        for i, runtime in enumerate(RUNTIME_ORDER):
            if runtime in resource_data:
                ax = axes[i]
                color = RUNTIME_COLORS[runtime]['primary']
                runtime_label = RUNTIME_COLORS[runtime]['label']
                
                for run in ["B1", "B2", "B3", "B4", "B5"]:
                    if run in resource_data[runtime]:
                        data = resource_data[runtime][run]
                        if 'pod_count' in data.columns and 'timestamp' in data.columns:
                            config = RUN_CONFIG[run]
                            label = f"{run} | {config['pods']} pods | {config['container_cpu']}"
                            
                            # Normalize time to 0-100% for comparison
                            time_normalized = np.linspace(0, 100, len(data))
                            ax.plot(time_normalized, data['pod_count'], 
                                   label=label, linewidth=2, alpha=0.8, marker='o', markersize=3)
                
                ax.set_ylabel('Pod Count', fontsize=12, fontweight='bold')
                ax.set_title(f'{runtime_label} Pod Count Over Time', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        axes[-1].set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = "b_series_pod_count_timeseries_comparison.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"        Saved: {filename}")

    def _create_cpu_scaling_analysis(self, resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create CPU scaling analysis across all configurations."""
        print(f"    Creating CPU scaling analysis")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: CPU usage vs configuration
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data:
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                configs = []
                avg_cpu_totals = []
                avg_cpu_per_pod = []
                
                for run in ["B1", "B2", "B3", "B4", "B5"]:
                    if run in resource_data[runtime]:
                        data = resource_data[runtime][run]
                        if 'cpu_usage_millicores' in data.columns and 'pod_count' in data.columns:
                            # Calculate average CPU usage
                            avg_cpu_total = data['cpu_usage_millicores'].mean()
                            avg_cpu_per_pod_val = data['cpu_per_pod'].mean() if 'cpu_per_pod' in data.columns else avg_cpu_total / data['pod_count'].mean()
                            
                            configs.append(run)
                            avg_cpu_totals.append(avg_cpu_total)
                            avg_cpu_per_pod.append(avg_cpu_per_pod_val)
                
                if configs:
                    config_indices = [1, 2, 3, 4, 5]  # B1=1, B2=2, etc.
                    ax1.plot(config_indices, avg_cpu_totals, 'o-', 
                           color=color, linewidth=3, markersize=8, 
                           label=f"{label} Total CPU", alpha=0.8)
                    ax2.plot(config_indices, avg_cpu_per_pod, 's--', 
                           color=color, linewidth=2, markersize=6, 
                           label=f"{label} CPU per Pod", alpha=0.8)
        
        # Customize plot 1 (Total CPU)
        ax1.set_xlabel('Configuration (B1 → B5)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total CPU Usage (millicores)', fontsize=12, fontweight='bold')
        ax1.set_title('B-Series CPU Scaling Analysis\nTotal CPU Usage Across Configurations', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks([1, 2, 3, 4, 5])
        ax1.set_xticklabels(['B1', 'B2', 'B3', 'B4', 'B5'])
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add configuration details
        config_text = "Configuration Details:\n"
        config_text += "B1: 1 pod × 1000m CPU\n"
        config_text += "B2: 2 pods × 500m CPU\n"
        config_text += "B3: 4 pods × 250m CPU\n"
        config_text += "B4: 8 pods × 125m CPU\n"
        config_text += "B5: 10 pods × 100m CPU"
        
        ax1.text(0.02, 0.98, config_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Customize plot 2 (CPU per Pod)
        ax2.set_xlabel('Configuration (B1 → B5)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('CPU Usage per Pod (millicores)', fontsize=12, fontweight='bold')
        ax2.set_title('B-Series CPU Efficiency Analysis\nCPU Usage per Pod Across Configurations', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks([1, 2, 3, 4, 5])
        ax2.set_xticklabels(['B1', 'B2', 'B3', 'B4', 'B5'])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add theoretical limits
        theoretical_limits = [1000, 500, 250, 125, 100]  # CPU limits per pod
        ax2.plot([1, 2, 3, 4, 5], theoretical_limits, 'k--', alpha=0.5, 
                linewidth=2, label='Theoretical CPU Limit')
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        filename = "b_series_cpu_scaling_analysis.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      Saved: {filename}")

    def _create_memory_scaling_analysis(self, resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create Memory scaling analysis across all configurations."""
        print(f"    Creating Memory scaling analysis")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Memory usage vs configuration
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data:
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                configs = []
                avg_memory_totals = []
                avg_memory_per_pod = []
                
                for run in ["B1", "B2", "B3", "B4", "B5"]:
                    if run in resource_data[runtime]:
                        data = resource_data[runtime][run]
                        if 'memory_usage_mib' in data.columns and 'pod_count' in data.columns:
                            # Calculate average memory usage
                            avg_memory_total = data['memory_usage_mib'].mean()
                            avg_memory_per_pod_val = data['memory_per_pod'].mean() if 'memory_per_pod' in data.columns else avg_memory_total / data['pod_count'].mean()
                            
                            configs.append(run)
                            avg_memory_totals.append(avg_memory_total)
                            avg_memory_per_pod.append(avg_memory_per_pod_val)
                
                if configs:
                    config_indices = [1, 2, 3, 4, 5]  # B1=1, B2=2, etc.
                    ax1.plot(config_indices, avg_memory_totals, 'o-', 
                           color=color, linewidth=3, markersize=8, 
                           label=f"{label} Total Memory", alpha=0.8)
                    ax2.plot(config_indices, avg_memory_per_pod, 's--', 
                           color=color, linewidth=2, markersize=6, 
                           label=f"{label} Memory per Pod", alpha=0.8)
        
        # Customize plot 1 (Total Memory)
        ax1.set_xlabel('Configuration (B1 → B5)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Memory Usage (MiB)', fontsize=12, fontweight='bold')
        ax1.set_title('B-Series Memory Scaling Analysis\nTotal Memory Usage Across Configurations', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks([1, 2, 3, 4, 5])
        ax1.set_xticklabels(['B1', 'B2', 'B3', 'B4', 'B5'])
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add configuration details
        config_text = "Configuration Details:\n"
        config_text += "B1: 1 pod × 1Gi Memory\n"
        config_text += "B2: 2 pods × 500Mi Memory\n"
        config_text += "B3: 4 pods × 250Mi Memory\n"
        config_text += "B4: 8 pods × 125Mi Memory\n"
        config_text += "B5: 10 pods × 100Mi Memory"
        
        ax1.text(0.02, 0.98, config_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # Customize plot 2 (Memory per Pod)
        ax2.set_xlabel('Configuration (B1 → B5)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Memory Usage per Pod (MiB)', fontsize=12, fontweight='bold')
        ax2.set_title('B-Series Memory Efficiency Analysis\nMemory Usage per Pod Across Configurations', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks([1, 2, 3, 4, 5])
        ax2.set_xticklabels(['B1', 'B2', 'B3', 'B4', 'B5'])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add theoretical limits (convert Gi to MiB)
        theoretical_limits = [1024, 512, 256, 128, 102]  # Memory limits per pod in MiB
        ax2.plot([1, 2, 3, 4, 5], theoretical_limits, 'k--', alpha=0.5, 
                linewidth=2, label='Theoretical Memory Limit')
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        filename = "b_series_memory_scaling_analysis.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      Saved: {filename}")

    def _create_resource_efficiency_analysis(self, resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create resource efficiency analysis comparing actual vs theoretical usage."""
        print(f"    Creating resource efficiency analysis")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # CPU and Memory theoretical limits
        cpu_limits = [1000, 500, 250, 125, 100]  # millicores per pod
        memory_limits = [1024, 512, 256, 128, 102]  # MiB per pod
        
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data:
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                cpu_efficiency = []
                memory_efficiency = []
                
                for i, run in enumerate(["B1", "B2", "B3", "B4", "B5"]):
                    if run in resource_data[runtime]:
                        data = resource_data[runtime][run]
                        
                        # Calculate CPU efficiency
                        if 'cpu_per_pod' in data.columns:
                            avg_cpu_per_pod = data['cpu_per_pod'].mean()
                            cpu_eff = (avg_cpu_per_pod / cpu_limits[i]) * 100
                            cpu_efficiency.append(cpu_eff)
                        
                        # Calculate Memory efficiency
                        if 'memory_per_pod' in data.columns:
                            avg_memory_per_pod = data['memory_per_pod'].mean()
                            memory_eff = (avg_memory_per_pod / memory_limits[i]) * 100
                            memory_efficiency.append(memory_eff)
                
                if cpu_efficiency:
                    config_indices = [1, 2, 3, 4, 5]
                    ax1.plot(config_indices, cpu_efficiency, 'o-', 
                           color=color, linewidth=3, markersize=8, 
                           label=f"{label} CPU Efficiency", alpha=0.8)
                
                if memory_efficiency:
                    ax2.plot(config_indices, memory_efficiency, 's--', 
                           color=color, linewidth=2, markersize=6, 
                           label=f"{label} Memory Efficiency", alpha=0.8)
        
        # Customize CPU efficiency plot
        ax1.set_xlabel('Configuration (B1 → B5)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('CPU Efficiency (%)', fontsize=12, fontweight='bold')
        ax1.set_title('B-Series CPU Resource Efficiency\nActual vs Theoretical CPU Usage per Pod', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks([1, 2, 3, 4, 5])
        ax1.set_xticklabels(['B1', 'B2', 'B3', 'B4', 'B5'])
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Add efficiency zones
        ax1.axhspan(0, 25, alpha=0.1, color='red', label='Low Efficiency')
        ax1.axhspan(25, 75, alpha=0.1, color='yellow', label='Medium Efficiency')
        ax1.axhspan(75, 100, alpha=0.1, color='green', label='High Efficiency')
        
        # Customize Memory efficiency plot
        ax2.set_xlabel('Configuration (B1 → B5)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Memory Efficiency (%)', fontsize=12, fontweight='bold')
        ax2.set_title('B-Series Memory Resource Efficiency\nActual vs Theoretical Memory Usage per Pod', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks([1, 2, 3, 4, 5])
        ax2.set_xticklabels(['B1', 'B2', 'B3', 'B4', 'B5'])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add efficiency zones
        ax2.axhspan(0, 25, alpha=0.1, color='red', label='Low Efficiency')
        ax2.axhspan(25, 75, alpha=0.1, color='yellow', label='Medium Efficiency')
        ax2.axhspan(75, 100, alpha=0.1, color='green', label='High Efficiency')
        
        plt.tight_layout()
        
        filename = "b_series_resource_efficiency_analysis.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      Saved: {filename}")

    def _create_resource_utilization_heatmap(self, resource_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Create resource utilization heatmap across runtimes and configurations."""
        print(f"    Creating resource utilization heatmap")
        
        # Prepare data for heatmap
        runtimes = []
        configs = ['B1', 'B2', 'B3', 'B4', 'B5']
        cpu_matrix = []
        memory_matrix = []
        
        for runtime in RUNTIME_ORDER:
            if runtime in resource_data:
                runtimes.append(RUNTIME_COLORS[runtime]['label'])
                cpu_row = []
                memory_row = []
                
                for run in configs:
                    if run in resource_data[runtime]:
                        data = resource_data[runtime][run]
                        
                        # Calculate average resource usage per pod
                        if 'cpu_per_pod' in data.columns:
                            avg_cpu = data['cpu_per_pod'].mean()
                        else:
                            avg_cpu = 0
                        
                        if 'memory_per_pod' in data.columns:
                            avg_memory = data['memory_per_pod'].mean()
                        else:
                            avg_memory = 0
                        
                        cpu_row.append(avg_cpu)
                        memory_row.append(avg_memory)
                    else:
                        cpu_row.append(0)
                        memory_row.append(0)
                
                cpu_matrix.append(cpu_row)
                memory_matrix.append(memory_row)
        
        if not cpu_matrix or not memory_matrix:
            print("      No data available for heatmap")
            return
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # CPU heatmap
        im1 = ax1.imshow(cpu_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs)
        ax1.set_yticks(range(len(runtimes)))
        ax1.set_yticklabels(runtimes)
        ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Runtime', fontsize=12, fontweight='bold')
        ax1.set_title('CPU Usage per Pod (millicores)\nResource Utilization Heatmap', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add text annotations for CPU
        for i in range(len(runtimes)):
            for j in range(len(configs)):
                text = ax1.text(j, i, f'{cpu_matrix[i][j]:.0f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar for CPU
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('CPU (millicores)', fontsize=10)
        
        # Memory heatmap
        im2 = ax2.imshow(memory_matrix, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs)
        ax2.set_yticks(range(len(runtimes)))
        ax2.set_yticklabels(runtimes)
        ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Runtime', fontsize=12, fontweight='bold')
        ax2.set_title('Memory Usage per Pod (MiB)\nResource Utilization Heatmap', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add text annotations for Memory
        for i in range(len(runtimes)):
            for j in range(len(configs)):
                text = ax2.text(j, i, f'{memory_matrix[i][j]:.0f}',
                               ha="center", va="center", color="white", fontweight='bold')
        
        # Add colorbar for Memory
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Memory (MiB)', fontsize=10)
        
        plt.tight_layout()
        
        filename = "b_series_resource_utilization_heatmap.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      Saved: {filename}")

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
        
        # Generate comprehensive resource scaling analysis
        self.create_comprehensive_resource_scaling_analysis(all_runtime_data)
        
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
