#!/usr/bin/env python3
"""
A-Series Visualizations
======================

Creates visualizations specifically for A-series (A1) test runs.
Focuses on cross-runtime comparisons and single-threading performance analysis.

Features:
- Cross-runtime A1 comparisons
- Dual throughput analysis comparisons
- Efficiency analysis under containerConcurrency=1
- Latency behavior comparisons
- Single-threading performance limits analysis
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
from typing import Dict, Optional
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

# =============================================================================
# A-SERIES VISUALIZATION GENERATOR CLASS
# =============================================================================

class ASeriesVisualizer:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "a_series_visualizations_results")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("A-SERIES VISUALIZATIONS")
        print("=" * 60)
        print(f"Base Directory: {base_dir}")
        print(f"Charts will be saved in: {self.output_dir}")
        print(f"Focus: Single-threading performance (containerConcurrency=1)")
        print(f"Runtime Order: {' → '.join([RUNTIME_COLORS[r]['label'] for r in RUNTIME_ORDER])}")
        print("=" * 60)

    # =========================================================================
    # CHART MANAGEMENT UTILITIES 
    # =========================================================================
    
    def clear_existing_charts(self):
        """Remove any existing A-series charts from the output directory."""
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                if file.endswith('.png') and 'a1_' in file:
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

    def _find_a1_directories(self) -> Dict[str, str]:
        """Find all A1 test directories."""
        a1_dirs = {}
        
        for runtime in RUNTIME_ORDER:
            pattern = f"results_fibonacci_{runtime}_a1"
            matching_dirs = glob.glob(pattern)
            if matching_dirs:
                a1_dirs[runtime] = matching_dirs[0]
        
        return a1_dirs

    def load_runtime_data(self, a1_dirs: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load data for all A1 runtimes."""
        runtime_data = {}
        
        for runtime in RUNTIME_ORDER:
            if runtime in a1_dirs:
                achieved_df = self.load_achieved_rps_data(a1_dirs[runtime])
                stats_df = self.load_statistics_data(a1_dirs[runtime])
                
                if achieved_df is not None and stats_df is not None:
                    # Merge data
                    merged = pd.merge(achieved_df, stats_df, left_on='target_rps', right_on='rps', how='inner')
                    merged['runtime'] = runtime
                    runtime_data[runtime] = merged
                    print(f"  Loaded {runtime} A1 data: {len(merged)} points")
                else:
                    print(f"  Failed to load {runtime} A1 data")
            else:
                print(f"  No A1 directory found for {runtime}")
        
        return runtime_data

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
                
                ax.plot(data['target_rps'], data['max_achieved'], 'o-', 
                       color=color, linewidth=3, markersize=6, label=label)
        
        # Add 100% efficiency line
        max_target = max([data['target_rps'].max() for data in runtime_data.values()])
        ax.plot([0, max_target], [0, max_target], 'k--', alpha=0.5, label='100% Efficiency')
        
        ax.set_xlabel('Target RPS', fontsize=12, fontweight='bold')
        ax.set_ylabel('Achieved RPS', fontsize=12, fontweight='bold')
        ax.set_title('A-Series Cross-Runtime Throughput Comparison\nSingle-Threading Performance Limits', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = "a1_cross_runtime_throughput_comparison.png"
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
                efficiency = (data['max_achieved'] / data['target_rps']) * 100
                ax.plot(data['target_rps'], efficiency, 'o-', 
                       color=color, linewidth=3, markersize=6, label=label)
        
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='100% Efficiency')
        ax.set_xlabel('Target RPS', fontsize=12, fontweight='bold')
        ax.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
        ax.set_title('A-Series Efficiency Comparison\nSingle-Threading Performance Limits', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = "a1_efficiency_comparison.png"
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
                    data = runtime_data[runtime].sort_values('max_achieved')
                    color = RUNTIME_COLORS[runtime]['primary']
                    label = RUNTIME_COLORS[runtime]['label']
                    
                    # Convert microseconds to milliseconds 
                    latency_ms = data[metric_col] / 1000.0
                    ax.plot(data['max_achieved'], latency_ms, 'o-', 
                           color=color, linewidth=3, markersize=6, label=label)
            
            ax.set_xlabel('Achieved Throughput (RPS)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric_label} Latency (ms)', fontsize=12, fontweight='bold')
            ax.set_title(f'A-Series {metric_label} Latency Comparison\nCross-Runtime Performance Under Single-Threading', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plt.tight_layout()
            
            filename = f"a1_{metric_col}_latency_comparison.png"
            self._save_chart(filename, fig)
            
            print(f"  Saved {metric_label} comparison")



    def create_dual_throughput_analysis(self, runtime_data: Dict[str, pd.DataFrame]):
        """Create dual throughput analysis chart comparing achieved vs calculated RPS."""
        print(f"\nCreating dual throughput analysis")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for runtime in RUNTIME_ORDER:
            if runtime in runtime_data:
                data = runtime_data[runtime].sort_values('target_rps')
                color = RUNTIME_COLORS[runtime]['primary']
                label = RUNTIME_COLORS[runtime]['label']
                
                # Plot achieved RPS (solid line with circles)
                ax.plot(data['target_rps'], data['max_achieved'], 'o-', 
                       color=color, linewidth=3, markersize=6, label=f"{label} Achieved")
                
                # Plot calculated throughput (dashed line with squares, lighter color)
                ax.plot(data['target_rps'], data['throughput_rps'], 's--', 
                       color=color, linewidth=2, markersize=4, alpha=0.7, label=f"{label} Calculated")
        
        # Add 100% efficiency line (dashed grey)
        max_target = max([data['target_rps'].max() for data in runtime_data.values()])
        ax.plot([0, max_target], [0, max_target], 'k--', alpha=0.5, label='100% Efficiency')
        
        ax.set_xlabel('Target RPS', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (RPS)', fontsize=12, fontweight='bold')
        ax.set_title('A1 Dual Throughput Analysis - All Runtimes\nAchieved vs Calculated RPS Under containerConcurrency=1', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = "a1_dual_throughput_analysis.png"
        self._save_chart(filename, fig)
        
        print(f"  Saved dual throughput analysis")



    # =========================================================================
    # MAIN EXECUTION FUNCTION
    # =========================================================================

    def generate_all_a_series_visualizations(self):
        """Generate all A-series visualizations."""
        print("\nStarting A-series visualization generation...")
        
        # Clear existing charts first
        self.clear_existing_charts()
        
        # Find A1 directories
        a1_dirs = self._find_a1_directories()
        if not a1_dirs:
            print("No A1 directories found!")
            return
        
        print("\nFound A1 directories:")
        for runtime, path in a1_dirs.items():
            print(f"  {RUNTIME_COLORS[runtime]['label']}: {path}")
        
        # Load data for all runtimes
        runtime_data = self.load_runtime_data(a1_dirs)
        if not runtime_data:
            print("No runtime data loaded!")
            return
        
        print(f"\n{'='*60}")
        print("GENERATING A-SERIES COMPARISONS")
        print(f"{'='*60}")
        
        # Generate all visualizations
        self.create_cross_runtime_throughput_comparison(runtime_data)
        self.create_efficiency_comparison(runtime_data)
        self.create_latency_comparisons(runtime_data)
        self.create_dual_throughput_analysis(runtime_data)

        
        print(f"\nALL A-SERIES VISUALIZATIONS COMPLETED!")
        print(f"Charts saved in: {self.output_dir}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate A-series visualizations')
    parser.add_argument('--base-dir', default='.', help='Base directory containing results')
    
    args = parser.parse_args()
    
    # Create and run visualizer
    visualizer = ASeriesVisualizer(args.base_dir)
    visualizer.generate_all_a_series_visualizations()

if __name__ == "__main__":
    main()
