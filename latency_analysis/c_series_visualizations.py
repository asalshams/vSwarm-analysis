#!/usr/bin/env python3
"""
C-Series Visualizations
======================

Creates visualizations specifically for C-series (C1-C5) test runs.
Focuses on multi-pod scaling analysis with varying containerConcurrency.

Features:
- Cross-runtime C-series comparisons
- Multi-pod scaling efficiency analysis
- containerConcurrency impact analysis
- Latency behavior comparisons across scaling configurations
- Pod scaling performance analysis
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

# C-series configuration details - Multi-pod scaling with varying containerConcurrency
RUN_CONFIG = {
    "C1": {"pods": "1", "container_cpu": "2000m", "container_mem": "2Gi", "target_conc": "200"},
    "C2": {"pods": "2", "container_cpu": "1000m", "container_mem": "1Gi", "target_conc": "100"},
    "C3": {"pods": "4", "container_cpu": "500m", "container_mem": "500Mi", "target_conc": "50"},
    "C4": {"pods": "8", "container_cpu": "250m", "container_mem": "250Mi", "target_conc": "25"},
    "C5": {"pods": "10", "container_cpu": "200m", "container_mem": "200Mi", "target_conc": "20"},
}

# =============================================================================
# C-SERIES VISUALIZATION GENERATOR CLASS
# =============================================================================

class CSeriesVisualizer:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "c_series_visualizations_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create runtime-specific directories
        self.runtime_dirs = {}
        for runtime in RUNTIME_ORDER:
            runtime_dir = os.path.join(self.output_dir, f"{runtime}_c_series")
            os.makedirs(runtime_dir, exist_ok=True)
            self.runtime_dirs[runtime] = runtime_dir
        
        print("=" * 60)
        print("C-SERIES VISUALIZATIONS")
        print("=" * 60)
        print(f"Base Directory: {base_dir}")
        print(f"Central charts: {self.output_dir}")
        print(f"Runtime charts: {', '.join([f'{runtime}_c_series' for runtime in RUNTIME_ORDER])}")
        print(f"Focus: Multi-pod scaling with varying containerConcurrency")
        print(f"Runtime Order: {' → '.join([RUNTIME_COLORS[r]['label'] for r in RUNTIME_ORDER])}")
        print("=" * 60)

    # =========================================================================
    # CHART MANAGEMENT UTILITIES 
    # =========================================================================
    
    def clear_existing_charts(self):
        """Remove any existing C-series charts from the output directory."""
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                if file.endswith('.png') and 'c_series' in file.lower():
                    os.remove(os.path.join(self.output_dir, file))
                    print(f"  Removed existing chart: {file}")
    
    def _save_chart(self, filename: str, fig):
        """Save a chart to the centralized output directory."""
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")

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

    def load_statistics_data(self, results_dir: str) -> Optional[pd.DataFrame]:
        """Load calculated throughput data from statistics file."""
        stats_files = glob.glob(os.path.join(results_dir, '*_statistics.csv'))
        if not stats_files:
            print(f"Warning: No statistics files found in {results_dir}")
            return None
        
        try:
            # Use the first statistics file found
            stats_file = stats_files[0]
            df = pd.read_csv(stats_file)
            
            # Clean and convert data
            df['rps'] = pd.to_numeric(df['rps'], errors='coerce')
            df['throughput_rps'] = pd.to_numeric(df['throughput_rps'], errors='coerce')
            df['average'] = pd.to_numeric(df['average'], errors='coerce')
            df['p95'] = pd.to_numeric(df['p95'], errors='coerce')
            df['p99'] = pd.to_numeric(df['p99'], errors='coerce')
            
            # Remove any rows with NaN values
            df = df.dropna(subset=['rps', 'throughput_rps'])
            
            return df
        except Exception as e:
            print(f"Error loading statistics data from {stats_file}: {e}")
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

    def find_c_series_directories(self) -> Dict[str, List[str]]:
        """Find all C-series result directories organized by runtime."""
        c_series_dirs = {runtime: [] for runtime in RUNTIME_ORDER}
        
        for item in os.listdir(self.base_dir):
            if item.startswith('results_fibonacci_') and '_c' in item:
                # Extract runtime and test type
                parts = item.replace('results_fibonacci_', '').split('_')
                if len(parts) >= 2:
                    runtime = parts[0]
                    test_type = '_'.join(parts[1:])
                    
                    if runtime in RUNTIME_COLORS and test_type.startswith('c'):
                        c_series_dirs[runtime].append(os.path.join(self.base_dir, item))
        
        # Sort each runtime's directories
        for runtime in c_series_dirs:
            c_series_dirs[runtime].sort()
        
        return c_series_dirs

    def load_runtime_data(self, c_series_dirs: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """Load and aggregate data for all C-series runtimes."""
        all_runtime_data = {}
        
        for runtime in RUNTIME_ORDER:
            if runtime in c_series_dirs and c_series_dirs[runtime]:
                runtime_aggregated = []
                
                for results_dir in c_series_dirs[runtime]:
                    # Extract run name (C1, C2, etc.)
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
                    print(f"  Loaded {runtime} C-series data: {len(combined_df)} points across {len(runtime_aggregated)} runs")
                else:
                    print(f"  No valid data for {runtime} C-series")
        
        return all_runtime_data

    # =========================================================================
    # VISUALIZATION FUNCTIONS
    # =========================================================================

    def create_cross_runtime_throughput_comparison(self, runtime_data: Dict[str, pd.DataFrame]):
        """Create cross-runtime throughput comparison chart."""
        print(f"\nCreating cross-runtime throughput comparison")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for runtime in RUNTIME_ORDER:
            if runtime in runtime_data:
                data = runtime_data[runtime].sort_values('target_rps')
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                ax.plot(data['target_rps'], data['actual_rps'], 'o-', 
                       color=color, linewidth=3, markersize=6, label=label)
        
        # Add 100% efficiency line
        max_target = max([data['target_rps'].max() for data in runtime_data.values()])
        ax.plot([0, max_target], [0, max_target], 'k--', alpha=0.5, label='100% Efficiency')
        
        ax.set_xlabel('Target RPS', fontsize=12, fontweight='bold')
        ax.set_ylabel('Achieved RPS', fontsize=12, fontweight='bold')
        ax.set_title('C-Series Cross-Runtime Throughput Comparison\nMulti-Pod Scaling with Varying containerConcurrency', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = "c_series_cross_runtime_throughput_comparison.png"
        self._save_chart(filename, fig)

    def create_efficiency_comparison(self, runtime_data: Dict[str, pd.DataFrame]):
        """Create efficiency comparison chart."""
        print(f"\nCreating efficiency comparison")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for runtime in RUNTIME_ORDER:
            if runtime in runtime_data:
                data = runtime_data[runtime]
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                # Calculate efficiency
                efficiency = (data['actual_rps'] / data['target_rps']) * 100
                ax.plot(data['target_rps'], efficiency, 'o-', 
                       color=color, linewidth=3, markersize=6, label=label)
        
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='100% Efficiency')
        ax.set_xlabel('Target RPS', fontsize=12, fontweight='bold')
        ax.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
        ax.set_title('C-Series Efficiency Comparison\nMulti-Pod Scaling Performance Analysis', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = "c_series_efficiency_comparison.png"
        self._save_chart(filename, fig)

    def create_latency_comparisons(self, runtime_data: Dict[str, pd.DataFrame]):
        """Create latency comparison charts."""
        print(f"\nCreating latency comparisons")
        
        metrics = [
            ('average', 'Average'),
            ('p95', 'P95'),
            ('p99', 'P99')
        ]
        
        for metric_col, metric_label in metrics:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for runtime in RUNTIME_ORDER:
                if runtime in runtime_data:
                    data = runtime_data[runtime].sort_values('actual_rps')
                    color = RUNTIME_COLORS[runtime]['primary']
                    label = RUNTIME_COLORS[runtime]['label']
                    
                    # Convert microseconds to milliseconds 
                    latency_ms = data[metric_col] / 1000.0
                    ax.plot(data['actual_rps'], latency_ms, 'o-', 
                           color=color, linewidth=3, markersize=6, label=label)
            
            ax.set_xlabel('Achieved Throughput (RPS)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric_label} Latency (ms)', fontsize=12, fontweight='bold')
            ax.set_title(f'C-Series {metric_label} Latency Comparison\nCross-Runtime Performance Under Multi-Pod Scaling', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plt.tight_layout()
            
            filename = f"c_series_{metric_col}_latency_comparison.png"
            self._save_chart(filename, fig)
            
            print(f"  Saved {metric_label} comparison")

    # =========================================================================
    # COMPREHENSIVE C-SERIES VISUALIZATION FUNCTIONS
    # =========================================================================

    def extract_maximum_throughput_data(self, all_runtime_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract maximum throughput achieved by each runtime/config combination."""
        max_data = []
        
        # Need to go back to original directories to read "Avg of Max" from summary files
        c_series_dirs = self.find_c_series_directories()
        
        for runtime in ["go", "nodejs", "python"]:
            if runtime in c_series_dirs and c_series_dirs[runtime]:
                for results_dir in c_series_dirs[runtime]:
                    # Extract run name (C1, C2, etc.)
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

    def create_maximum_throughput_comparison(self, max_data: pd.DataFrame):
        """Create maximum throughput comparison by configuration."""
        print(f"\nCreating maximum throughput comparison")
        
        if max_data.empty:
            print("  No data available")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set up grouped bar chart
        runs = ["C1", "C2", "C3", "C4", "C5"]
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
        ax.set_title('C-Series Maximum Throughput by Configuration\nMulti-Pod Scaling Analysis', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add configuration details
        config_labels = []
        for run in runs:
            config = RUN_CONFIG[run]
            config_labels.append(f"{run}\n{config['pods']}×{config['container_cpu']}\nConc={config['target_conc']}")
        
        ax.set_xticks(x + width)
        ax.set_xticklabels(config_labels, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = "c_series_max_throughput_comparison.png"
        self._save_chart(filename, fig)

    def create_scaling_efficiency_comparison(self, max_data: pd.DataFrame):
        """Create scaling efficiency comparison."""
        print(f"\nCreating scaling efficiency comparison")
        
        if max_data.empty:
            print("  No data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Configuration setup
        runs = ["C1", "C2", "C3", "C4", "C5"]
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
        
        ax.set_xlabel('Configuration Index (C1 → C5)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Maximum Throughput (RPS)', fontsize=12, fontweight='bold')
        ax.set_title('C-Series Scaling Efficiency Analysis\nThroughput Changes Across Multi-Pod Configurations', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(config_indices)
        ax.set_xticklabels(runs)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add configuration strategy annotation
        config_text = "Multi-Pod Scaling Strategy:\n"
        config_text += "C1: 1 pod × 2000m CPU = 2.0 vCPU total (Conc=200)\n"
        config_text += "C2: 2 pods × 1000m CPU = 2.0 vCPU total (Conc=100)\n"
        config_text += "C3: 4 pods × 500m CPU = 2.0 vCPU total (Conc=50)\n"
        config_text += "C4: 8 pods × 250m CPU = 2.0 vCPU total (Conc=25)\n"
        config_text += "C5: 10 pods × 200m CPU = 2.0 vCPU total (Conc=20)"
        
        ax.text(0.98, 0.02, config_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
        
        plt.tight_layout()
        
        filename = "c_series_scaling_efficiency_comparison.png"
        self._save_chart(filename, fig)

    def create_pod_scaling_analysis(self, max_data: pd.DataFrame):
        """Create pod scaling analysis chart showing relationship between pod count and performance."""
        print(f"\nCreating pod scaling analysis")
        
        if max_data.empty:
            print("  No data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot throughput vs pod count for each runtime
        for runtime in RUNTIME_ORDER:
            if runtime in max_data['runtime'].values:
                runtime_data = max_data[max_data['runtime'] == runtime]
                
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                ax.plot(runtime_data['pods'], runtime_data['max_throughput'], 
                       marker='o', linewidth=3, markersize=8,
                       label=label, color=color, alpha=0.9)
                
                # Add annotations for containerConcurrency
                for _, row in runtime_data.iterrows():
                    ax.annotate(f"Conc={row['target_conc']}", 
                               (row['pods'], row['max_throughput']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Number of Pods', fontsize=12, fontweight='bold')
        ax.set_ylabel('Maximum Throughput (RPS)', fontsize=12, fontweight='bold')
        ax.set_title('C-Series Pod Scaling Analysis\nThroughput vs Pod Count with containerConcurrency', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 4, 8, 10])
        
        plt.tight_layout()
        
        filename = "c_series_pod_scaling_analysis.png"
        self._save_chart(filename, fig)

    def create_concurrency_impact_analysis(self, max_data: pd.DataFrame):
        """Create analysis showing impact of containerConcurrency on performance."""
        print(f"\nCreating containerConcurrency impact analysis")
        
        if max_data.empty:
            print("  No data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot throughput vs containerConcurrency for each runtime
        for runtime in RUNTIME_ORDER:
            if runtime in max_data['runtime'].values:
                runtime_data = max_data[max_data['runtime'] == runtime]
                
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                # Sort by containerConcurrency (descending)
                runtime_data = runtime_data.sort_values('target_conc', ascending=False)
                
                ax.plot(runtime_data['target_conc'], runtime_data['max_throughput'], 
                       marker='o', linewidth=3, markersize=8,
                       label=label, color=color, alpha=0.9)
                
                # Add annotations for pod count
                for _, row in runtime_data.iterrows():
                    ax.annotate(f"{row['pods']} pods", 
                               (row['target_conc'], row['max_throughput']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
        
        ax.set_xlabel('containerConcurrency', fontsize=12, fontweight='bold')
        ax.set_ylabel('Maximum Throughput (RPS)', fontsize=12, fontweight='bold')
        ax.set_title('C-Series containerConcurrency Impact Analysis\nThroughput vs Concurrency with Pod Counts', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([20, 25, 50, 100, 200])
        
        plt.tight_layout()
        
        filename = "c_series_concurrency_impact_analysis.png"
        self._save_chart(filename, fig)

    def create_runtime_scaling_trends(self, max_data: pd.DataFrame):
        """Create individual runtime scaling trend analysis."""
        print(f"\nCreating runtime scaling trends")
        
        if max_data.empty:
            print("  No data available")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('C-Series Individual Runtime Scaling Trends\nMulti-Pod Configuration Impact on Each Runtime', 
                     fontsize=16, fontweight='bold')
        
        runs = ["C1", "C2", "C3", "C4", "C5"]
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
                    c1_val = throughputs[0] if throughputs[0] > 0 else 1
                    c5_val = throughputs[-1] if throughputs[-1] > 0 else 1
                    change_pct = ((c5_val - c1_val) / c1_val) * 100
                    
                    trend_text = f"C1→C5: {change_pct:+.1f}%"
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
        
        filename = "c_series_runtime_scaling_trends.png"
        self._save_chart(filename, fig)

    # =========================================================================
    # RUNTIME-SPECIFIC CHART GENERATION (LIKE A-SERIES AND B-SERIES)
    # =========================================================================

    def create_runtime_specific_charts(self, all_runtime_data: Dict[str, pd.DataFrame]):
        """Create runtime-specific charts for each runtime (like A-series and B-series)."""
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
        
        # 4. Generate aggregated CSV data (like A-series and B-series)
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
            
            for run in ["C1", "C2", "C3", "C4", "C5"]:
                run_data = grouped_data[grouped_data['run'] == run]
                if not run_data.empty:
                    config = RUN_CONFIG[run]
                    label = f"{run} | {config['pods']}×{config['container_cpu']} | Conc={config['target_conc']}"
                    
                    # Convert microseconds to milliseconds
                    y_values = run_data[metric_col].values / 1000
                    ax.plot(run_data['target_rps'], y_values, marker='o', label=label, linewidth=2)
            
            ax.set_xlabel('Target RPS', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric_label} Latency (ms)', fontsize=12, fontweight='bold')
            ax.set_title(f'{runtime_label} C-Series: {metric_label} Latency vs Target RPS\nMulti-Pod Scaling Analysis', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f"{runtime}_c_series_latency_{metric_col}_vs_target.png"
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
            
            for run in ["C1", "C2", "C3", "C4", "C5"]:
                run_data = grouped_data[grouped_data['run'] == run]
                if not run_data.empty:
                    config = RUN_CONFIG[run]
                    label = f"{run} | {config['pods']}×{config['container_cpu']} | Conc={config['target_conc']}"
                    
                    # Convert microseconds to milliseconds
                    y_values = run_data[metric_col].values / 1000
                    # Use actual_rps if available, otherwise throughput_rps
                    x_values = run_data['actual_rps'].fillna(run_data['throughput_rps']).values
                    ax.plot(x_values, y_values, marker='o', label=label, linewidth=2)
            
            ax.set_xlabel('Achieved Throughput (RPS)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric_label} Latency (ms)', fontsize=12, fontweight='bold')
            ax.set_title(f'{runtime_label} C-Series: {metric_label} Latency vs Achieved Throughput\nMulti-Pod Scaling Analysis', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f"{runtime}_c_series_latency_{metric_col}_vs_throughput.png"
            output_path = os.path.join(runtime_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved: {filename}")

    def _create_timeseries_charts(self, runtime: str, runtime_dir: str, runtime_label: str):
        """Create timeseries comparison charts for a specific runtime."""
        # Find all C-series directories for this runtime
        runtime_pattern = f"results_fibonacci_{runtime}_c"
        runtime_dirs = [d for d in os.listdir(self.base_dir) if d.startswith(runtime_pattern)]
        
        if not runtime_dirs:
            print(f"    No C-series directories found for {runtime}")
            return
        
        # Read pod monitoring data for each run
        timeseries_data = {}
        for run_dir in runtime_dirs:
            run_name = run_dir.split('_')[-1].upper()  # Extract C1, C2, etc.
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
                label = f"{run_name} | {config.get('pods', '?')}×{config.get('container_cpu', '?')}"
                
                # Normalize time to 0-100% for comparison
                time_normalized = np.linspace(0, 100, len(data))
                ax.plot(time_normalized, data['cpu_usage_millicores'], 
                       label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('CPU Usage (millicores)', fontsize=12, fontweight='bold')
        ax.set_title(f'{runtime_label} C-Series: CPU Usage Comparison Over Time\nMulti-Pod Scaling Analysis', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{runtime}_c_series_timeseries_cpu_comparison.png"
        output_path = os.path.join(runtime_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filename}")
        
        # Create Memory usage comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for run_name, data in timeseries_data.items():
            if 'memory_usage_mib' in data.columns:
                config = RUN_CONFIG.get(run_name, {})
                label = f"{run_name} | {config.get('pods', '?')}×{config.get('container_mem', '?')}"
                
                time_normalized = np.linspace(0, 100, len(data))
                ax.plot(time_normalized, data['memory_usage_mib'], 
                       label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Usage (MiB)', fontsize=12, fontweight='bold')
        ax.set_title(f'{runtime_label} C-Series: Memory Usage Comparison Over Time\nMulti-Pod Scaling Analysis', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{runtime}_c_series_timeseries_memory_comparison.png"
        output_path = os.path.join(runtime_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filename}")
        
        # Create Pod count comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for run_name, data in timeseries_data.items():
            if 'pod_count' in data.columns:
                config = RUN_CONFIG.get(run_name, {})
                label = f"{run_name} | {config.get('pods', '?')} pods"
                
                time_normalized = np.linspace(0, 100, len(data))
                ax.plot(time_normalized, data['pod_count'], 
                       label=label, linewidth=2, alpha=0.8, marker='o', markersize=4)
        
        ax.set_xlabel('Test Progress (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pod Count', fontsize=12, fontweight='bold')
        ax.set_title(f'{runtime_label} C-Series: Pod Count Comparison Over Time\nMulti-Pod Scaling Analysis', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{runtime}_c_series_timeseries_pod_count_comparison.png"
        output_path = os.path.join(runtime_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filename}")

    def _create_aggregated_csv(self, runtime: str, runtime_data: pd.DataFrame, runtime_dir: str, runtime_label: str):
        """Create aggregated CSV data file for a specific runtime."""
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
            
            # Sort by run (C1, C2, C3, C4, C5) then by target_rps
            grouped['run_order'] = grouped['run'].map({'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5})
            grouped = grouped.sort_values(['run_order', 'target_rps']).drop('run_order', axis=1)
            
            # Save aggregated CSV
            filename = f"fibonacci_{runtime}_c_aggregated.csv"
            output_path = os.path.join(runtime_dir, filename)
            grouped.to_csv(output_path, index=False)
            
            print(f"      Saved: {filename}")
            
            # Also save a summary version with key metrics - similar to B-series
            summary_data = []
            for run in ["C1", "C2", "C3", "C4", "C5"]:
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
                                        'containerConcurrency': max_throughput_row['target_conc'],
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
                summary_filename = f"fibonacci_{runtime}_c_summary.csv"
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

    def generate_all_c_series_visualizations(self):
        """Generate all C-series visualizations."""
        print("\nStarting C-series visualization generation...")
        
        # Clear existing charts first
        self.clear_existing_charts()
        
        # Find C-series directories
        c_series_dirs = self.find_c_series_directories()
        
        if not any(c_series_dirs.values()):
            print("No C-series result directories found")
            return
        
        print(f"\nFound C-series directories:")
        for runtime in RUNTIME_ORDER:
            if c_series_dirs[runtime]:
                runtime_label = RUNTIME_COLORS[runtime]['label']
                print(f"  {runtime_label}: {len(c_series_dirs[runtime])} runs")
                for dir_path in c_series_dirs[runtime]:
                    print(f"    - {os.path.basename(dir_path)}")
        
        # Load runtime data
        all_runtime_data = self.load_runtime_data(c_series_dirs)
        
        if len(all_runtime_data) < 1:
            print("No valid C-series data found")
            return
        
        # Extract maximum throughput data
        max_data = self.extract_maximum_throughput_data(all_runtime_data)
        
        if max_data.empty:
            print("No maximum throughput data available")
            return
        
        print(f"\n{'='*60}")
        print(f"GENERATING C-SERIES COMPARISONS")
        print(f"{'='*60}")
        
        # Generate centralized C-series comparison charts (like A-series and B-series)
        self.create_maximum_throughput_comparison(max_data)
        self.create_scaling_efficiency_comparison(max_data)
        self.create_runtime_scaling_trends(max_data)
        
        # Generate runtime-specific charts (like A-series and B-series)
        self.create_runtime_specific_charts(all_runtime_data)
        
        print(f"\nALL C-SERIES VISUALIZATIONS COMPLETED!")
        print(f"Output directory: {self.output_dir}")
        print(f"Focus: Multi-pod scaling with varying containerConcurrency")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate C-series visualizations')
    parser.add_argument('--base-dir', default='.', help='Base directory containing results')
    
    args = parser.parse_args()
    
    # Create and run visualizer
    visualizer = CSeriesVisualizer(args.base_dir)
    visualizer.generate_all_c_series_visualizations()

if __name__ == "__main__":
    main()
