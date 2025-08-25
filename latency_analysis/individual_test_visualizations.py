#!/usr/bin/env python3
"""
Individual Test Run Visualizations
=================================

Creates individual latency curve visualizations for each test run.
Generates separate files for each runtime and test configuration.

Features:
- Consistent color scheme: Python=Yellow, Node.js=Green, Go=Blue
- Individual latency curves (avg, p95, p99) for each test run
- Uses achieved RPS (not calculated throughput)
- Dual throughput analysis for each run
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
from typing import Optional
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
        "shades": {
            "avg": "#FFC107",     # Base yellow
            "p95": "#FF9800",     # Darker yellow/orange
            "p99": "#FF6F00"      # Darkest yellow/orange
        },
        "label": "Python"
    },
    "nodejs": {
        "primary": "#4CAF50",    # Green
        "shades": {
            "avg": "#4CAF50",     # Base green
            "p95": "#388E3C",     # Darker green
            "p99": "#2E7D32"      # Darkest green
        },
        "label": "Node.js"
    },
    "go": {
        "primary": "#2196F3",    # Blue
        "shades": {
            "avg": "#2196F3",     # Base blue
            "p95": "#1976D2",     # Darker blue
            "p99": "#0D47A1"      # Darkest blue
        },
        "label": "Go"
    }
}

# Runtime processing order
RUNTIME_ORDER = ["python", "nodejs", "go"]

# =============================================================================
# INDIVIDUAL TEST VISUALIZATION GENERATOR CLASS
# =============================================================================

class IndividualTestVisualizer:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        
        print("=" * 60)
        print("INDIVIDUAL TEST RUN VISUALIZATIONS")
        print("=" * 60)
        print(f"Base Directory: {base_dir}")
        print("Charts will be saved directly in each test results directory")
        print(f"Runtime Order: {' → '.join([RUNTIME_COLORS[r]['label'] for r in RUNTIME_ORDER])}")
        print("=" * 60)

    # =========================================================================
    # CHART MANAGEMENT UTILITIES 
    # =========================================================================
    
    def clear_existing_charts(self, results_dir: str):
        """Remove any existing charts directories in the test results directory."""
        charts_dir = os.path.join(results_dir, "charts")
        charts_with_rps_dir = os.path.join(results_dir, "charts_with_achieved_rps")
        
        # Remove charts directory if it exists
        if os.path.exists(charts_dir):
            import shutil
            shutil.rmtree(charts_dir)
            print(f"  Removed existing charts directory")
        
        # Remove charts_with_achieved_rps directory if it exists
        if os.path.exists(charts_with_rps_dir):
            import shutil
            shutil.rmtree(charts_with_rps_dir)
            print(f"  Removed existing charts_with_achieved_rps directory")
        
        # Also clean up any existing PNG files from previous runs
        chart_keywords = ['latency_curves', 'dual_throughput', 'resource_timeseries']
        for file in os.listdir(results_dir):
            if file.endswith('.png') and any(keyword in file for keyword in chart_keywords):
                os.remove(os.path.join(results_dir, file))
                print(f"  Removed existing chart: {file}")

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
            print(f"Warning: No statistics CSV found in {results_dir}")
            return None
        
        try:
            df = pd.read_csv(stats_files[0])
            return df
        except Exception as e:
            print(f"Error loading statistics data: {e}")
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

    def find_all_test_directories(self):
        """Find all test result directories."""
        test_dirs = []
        
        for item in os.listdir(self.base_dir):
            if item.startswith('results_fibonacci_'):
                # Extract runtime and test type
                parts = item.replace('results_fibonacci_', '').split('_')
                if len(parts) >= 2:
                    runtime = parts[0]
                    test_type = '_'.join(parts[1:])
                    
                    if runtime in RUNTIME_COLORS:
                        test_dirs.append({
                            'directory': item,
                            'runtime': runtime,
                            'test_type': test_type,
                            'full_path': os.path.join(self.base_dir, item)
                        })
        
        # Sort by runtime order, then by test type
        def sort_key(item):
            runtime_idx = RUNTIME_ORDER.index(item['runtime']) if item['runtime'] in RUNTIME_ORDER else 999
            return (runtime_idx, item['test_type'])
        
        test_dirs.sort(key=sort_key)
        return test_dirs

    # =========================================================================
    # VISUALIZATION GENERATORS
    # =========================================================================

    def create_latency_curves(self, test_info: dict):
        """Create latency vs throughput curves for a single test run."""
        results_dir = test_info['full_path']
        runtime = test_info['runtime']
        test_type = test_info['test_type']
        
        print(f"\nCreating latency curves for {runtime} {test_type}")
        
        # Load data
        achieved_df = self.load_achieved_rps_data(results_dir)
        detailed_df = self.load_detailed_analysis_data(results_dir)
        
        if achieved_df is None or detailed_df is None:
            print(f"  Missing data for {results_dir}")
            return
        
        # Merge achieved RPS with detailed analysis
        merged_df = pd.merge(detailed_df, achieved_df[['target_rps', 'max_achieved']], 
                           on='target_rps', how='left')
        
        # Use achieved RPS where available, fallback to calculated
        merged_df['actual_rps'] = merged_df['max_achieved'].fillna(merged_df['throughput_rps'])
        merged_df = merged_df.sort_values('actual_rps')
        
        # Get runtime colors
        colors = RUNTIME_COLORS[runtime]['shades']
        runtime_label = RUNTIME_COLORS[runtime]['label']
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the three latency curves
        metrics = [
            ('average', 'avg', 'Average Latency'),
            ('p95', 'p95', 'P95 Latency'),
            ('p99', 'p99', 'P99 Latency')
        ]
        
        for metric_col, color_key, label in metrics:
            # Convert microseconds to milliseconds (latency data is in microseconds from invoker)
            latency_ms = merged_df[metric_col] / 1000.0
            
            ax.plot(merged_df['actual_rps'], latency_ms, 
                   marker='o', linewidth=3, markersize=6, 
                   color=colors[color_key], label=label, alpha=0.9)
        
        # Customize the chart
        ax.set_xlabel('Achieved Throughput (RPS)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'{runtime_label} {test_type.upper()} - Latency vs Throughput', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11)
        ax.set_yscale('log')
        
        # Add performance summary
        max_rps = merged_df['actual_rps'].max()
        min_avg_latency = (merged_df['average'] / 1000.0).min()  # Convert μs to ms
        summary_text = f'Peak: {max_rps:.0f} RPS\nMin Avg: {min_avg_latency:.1f}ms'
        ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        # Clear existing charts in this directory
        self.clear_existing_charts(results_dir)
        
        # Save the chart directly in the test results directory
        filename = f"{runtime}_{test_type}_latency_curves.png"
        output_path = os.path.join(results_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path}")

    def create_dual_throughput_analysis(self, test_info: dict):
        """Create dual throughput analysis for a single test run."""
        results_dir = test_info['full_path']
        runtime = test_info['runtime']
        test_type = test_info['test_type']
        
        print(f"\nCreating dual throughput analysis for {runtime} {test_type}")
        
        # Load data
        achieved_df = self.load_achieved_rps_data(results_dir)
        stats_df = self.load_statistics_data(results_dir)
        
        if achieved_df is None or stats_df is None:
            print(f"  Missing data for {results_dir}")
            return
        
        # Merge the dataframes
        merged = pd.merge(achieved_df, stats_df, left_on='target_rps', right_on='rps', how='inner')
        
        # Get runtime colors
        primary_color = RUNTIME_COLORS[runtime]['primary']
        runtime_label = RUNTIME_COLORS[runtime]['label']
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot calculated throughput (background)
        ax.plot(merged['target_rps'], merged['throughput_rps'], 's--', 
               color=primary_color, alpha=0.6, linewidth=2, markersize=6, 
               label=f'{runtime_label} Calculated')
        
        # Plot achieved throughput (foreground)
        ax.plot(merged['target_rps'], merged['max_achieved'], 'o-', 
               color=primary_color, linewidth=3, markersize=8, 
               label=f'{runtime_label} Achieved')
        
        # Plot ideal line
        ax.plot(merged['target_rps'], merged['target_rps'], '--', 
               color='gray', alpha=0.5, label='Target RPS')
        
        # Add annotations for key points
        max_achieved_idx = merged['max_achieved'].idxmax()
        max_achieved_rps = merged.loc[max_achieved_idx, 'max_achieved']
        max_achieved_target = merged.loc[max_achieved_idx, 'target_rps']
        
        ax.annotate(f'Max Achieved: {max_achieved_rps:.1f} RPS\n@ {max_achieved_target:.0f} target', 
                   xy=(max_achieved_target, max_achieved_rps),
                   xytext=(10, 10), textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color=primary_color))
        
        # Customize the chart
        ax.set_xlabel('Target RPS', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (RPS)', fontsize=12, fontweight='bold')
        ax.set_title(f'{runtime_label} {test_type.upper()} - Dual Throughput Analysis\nAchieved vs Calculated RPS', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add gap analysis text
        throughput_gap = merged['throughput_rps'] - merged['max_achieved']
        max_gap = throughput_gap.max()
        avg_gap = throughput_gap.mean()
        
        gap_text = f'Throughput Gap Analysis:\nMax Gap: {max_gap:.1f} RPS\nAvg Gap: {avg_gap:.1f} RPS'
        ax.text(0.02, 0.98, gap_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        # Save the chart directly in the test results directory
        filename = f"{runtime}_{test_type}_dual_throughput_analysis.png"
        output_path = os.path.join(results_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path}")

    def create_resource_timeseries_chart(self, test_info: dict):
        """Create resource usage timeseries chart if monitoring data exists."""
        results_dir = test_info['full_path']
        runtime = test_info['runtime']
        test_type = test_info['test_type']
        
        # Look for pod monitoring file
        monitoring_files = glob.glob(os.path.join(results_dir, '*_pod_monitoring.csv'))
        if not monitoring_files:
            print(f"  No pod monitoring data found for {runtime} {test_type}")
            return
        
        monitoring_file = monitoring_files[0]
        print(f"\nCreating resource timeseries chart for {runtime} {test_type}")
        
        try:
            # Load monitoring data
            resource_df = pd.read_csv(monitoring_file)
            
            if resource_df.empty:
                print(f"  Empty monitoring data")
                return
            
            # Convert timestamp to datetime
            resource_df['timestamp'] = pd.to_datetime(resource_df['timestamp'])
            resource_df = resource_df.sort_values('timestamp')
            
            # Create timeseries chart with proper styling like run_test.py
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
            
            runtime_label = RUNTIME_COLORS[runtime]['label']
            color = RUNTIME_COLORS[runtime]['primary']
            
            # Pod count over time
            ax1.plot(resource_df['timestamp'], resource_df['pod_count'], 
                    color=color, linewidth=2, marker='o', markersize=4)
            ax1.set_ylabel('Pod Count', fontsize=11, fontweight='bold')
            ax1.set_title(f'{runtime_label} {test_type.upper()} - Resource Usage Over Time', 
                         fontsize=14, fontweight='bold', pad=20)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
            ax1.axvline(x=resource_df['timestamp'].min(), color='black', linewidth=0.5, alpha=0.3)
            
            # CPU usage over time
            ax2.plot(resource_df['timestamp'], resource_df['cpu_usage_millicores'], 
                    color=color, linewidth=2, marker='o', markersize=4)
            ax2.set_ylabel('CPU (millicores)', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
            ax2.axvline(x=resource_df['timestamp'].min(), color='black', linewidth=0.5, alpha=0.3)
            
            # Memory usage over time
            ax3.plot(resource_df['timestamp'], resource_df['memory_usage_mib'], 
                    color=color, linewidth=2, marker='o', markersize=4)
            ax3.set_xlabel('Time', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Memory (MiB)', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
            ax3.axvline(x=resource_df['timestamp'].min(), color='black', linewidth=0.5, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            filename = f"{runtime}_{test_type}_resource_timeseries.png"
            output_path = os.path.join(results_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {output_path}")
            
        except Exception as e:
            print(f"  Error creating resource timeseries: {e}")

    def create_comprehensive_timeseries_chart(self, test_info: dict):
        """Create comprehensive timeseries chart with latency above and resources below."""
        results_dir = test_info['full_path']
        runtime = test_info['runtime']
        test_type = test_info['test_type']
        
        # Look for pod monitoring file
        monitoring_files = glob.glob(os.path.join(results_dir, '*_pod_monitoring.csv'))
        if not monitoring_files:
            print(f"  No pod monitoring data found for comprehensive timeseries")
            return
        
        monitoring_file = monitoring_files[0]
        print(f"\nCreating comprehensive timeseries chart for {runtime} {test_type}")
        
        try:
            # Load monitoring data
            resource_data = pd.read_csv(monitoring_file)
            
            if resource_data.empty:
                print(f"  Empty monitoring data")
                return
            
            # Convert timestamp to datetime
            resource_data['timestamp'] = pd.to_datetime(resource_data['timestamp'])
            resource_data = resource_data.sort_values('timestamp')
            
            # Load actual latency data files for comprehensive view (like run_test.py)
            latency_files = glob.glob(os.path.join(results_dir, '*_lat.csv'))
            if not latency_files:
                print(f"  No latency data files found for comprehensive timeseries")
                return
            
            # Build ordered list of (target_rps, iteration, data) like run_test.py
            ordered_entries = []
            for filename in latency_files:
                try:
                    # Expect: target_rps{R}_iter{II}_rps{A}_lat.csv
                    m = re.search(r'target_rps(\d+\.?\d*)_iter(\d+)_rps(\d+\.?\d*)_lat\.csv', filename)
                    if m:
                        target_rps = float(m.group(1))
                        iteration = int(m.group(2))
                        # Load the actual latency data
                        data = np.loadtxt(filename, delimiter=',')
                        ordered_entries.append((target_rps, iteration, data, filename))
                    else:
                        # Fallback: put at end if pattern doesn't match
                        data = np.loadtxt(filename, delimiter=',')
                        ordered_entries.append((float('inf'), 999, data, filename))
                except Exception:
                    ordered_entries.append((float('inf'), 999, data, filename))
            
            # Sort by target RPS then iteration
            ordered_entries.sort(key=lambda x: (x[0], x[1]))
            
            # Concatenate all latency datasets for the full test series
            concatenated = np.concatenate([e[2] for e in ordered_entries]) if ordered_entries else None
            total_requests = int(len(concatenated)) if concatenated is not None else 0
            
            # Create comprehensive timeseries chart like run_test.py
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            runtime_label = RUNTIME_COLORS[runtime]['label']
            
            # Create request axis for resource data alignment
            time_points = np.linspace(0, total_requests, len(resource_data))
            
            # Plot pod count and resources
            ax2.plot(time_points, resource_data['pod_count'], 'g-', linewidth=2, label='Pod Count')
            ax2.set_ylabel('Pod Count')
            ax2.set_xlabel('Request Number')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            if 'cpu_usage_millicores' in resource_data.columns:
                ax3 = ax2.twinx()
                ax3.plot(time_points, resource_data['cpu_usage_millicores'], 'orange', linewidth=2, label='CPU (millicores)', alpha=0.7)
                ax3.set_ylabel('CPU Usage (millicores)', color='orange')
                ax3.legend(loc='upper right')
            
            if 'memory_usage_mib' in resource_data.columns:
                ax3.plot(time_points, resource_data['memory_usage_mib'], 'purple', linewidth=2, label='Memory (MiB)', alpha=0.7)
                ax3.set_ylabel('Memory Usage (MiB)', color='purple')
            
            # Compute phase offsets using actual lengths per target RPS
            request_offset = 0
            phase_lengths = {}
            for target_rps, iteration, data, _ in ordered_entries:
                phase_lengths.setdefault(target_rps, 0)
                phase_lengths[target_rps] += len(data)
            
            for target_rps in sorted(phase_lengths.keys()):
                ax2.axvline(x=request_offset, color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax2.text(request_offset, ax2.get_ylim()[1] * 0.9, f'RPS {target_rps:.0f}', 
                         rotation=90, verticalalignment='top', fontsize=8)
                request_offset += phase_lengths[target_rps]
            
            # Plot full latency series with moving average
            if concatenated is not None and len(concatenated) > 0:
                latency_ms = concatenated / 1000.0
                ax1.plot(range(len(latency_ms)), latency_ms, alpha=0.3, linewidth=0.8, label='Latency')
                ax1.set_ylabel('Latency (ms)')
                ax1.set_title(f'{runtime_label} {test_type.upper()} - Comprehensive Test Timeline')
                ax1.grid(True, alpha=0.3)
                
                # Moving average over a window proportional to phase size
                window_size = max(20, min(1000, len(latency_ms) // 200))
                if window_size > 1:
                    moving_avg = pd.Series(latency_ms).rolling(window=window_size).mean()
                    ax1.plot(range(len(moving_avg)), moving_avg, 'r-', linewidth=2, 
                             label=f'Moving Average ({window_size} requests)')
                    ax1.legend()
            
            plt.tight_layout()
            
            # Save chart
            filename = f"{runtime}_{test_type}_comprehensive_timeseries.png"
            output_path = os.path.join(results_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {output_path}")
            
        except Exception as e:
            print(f"  Error creating comprehensive timeseries: {e}")

    # =========================================================================
    # MAIN EXECUTION FUNCTION
    # =========================================================================

    def generate_all_individual_visualizations(self):
        """Generate all individual test visualizations."""
        print("\nStarting individual test visualization generation...")
        
        # Find all test directories
        test_dirs = self.find_all_test_directories()
        
        if not test_dirs:
            print("No test result directories found")
            return
        
        print(f"\nFound {len(test_dirs)} test directories:")
        for test_info in test_dirs:
            runtime_label = RUNTIME_COLORS[test_info['runtime']]['label']
            print(f"  {runtime_label} {test_info['test_type']}: {test_info['directory']}")
        
        # Process each test directory
        for test_info in test_dirs:
            runtime_label = RUNTIME_COLORS[test_info['runtime']]['label']
            print(f"\n{'='*60}")
            print(f"PROCESSING {runtime_label.upper()} {test_info['test_type'].upper()}")
            print(f"{'='*60}")
            
            # Create latency curves (combined)
            self.create_latency_curves(test_info)
            
            # Create dual throughput analysis
            self.create_dual_throughput_analysis(test_info)
            
            # Create resource timeseries chart (if monitoring data exists)
            self.create_resource_timeseries_chart(test_info)
            
                    # Create comprehensive timeseries chart (if monitoring data exists)
        self.create_comprehensive_timeseries_chart(test_info)
        
        print(f"\nALL INDIVIDUAL VISUALIZATIONS COMPLETED!")
        print(f"Charts saved directly in each test results directory")



# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate individual test visualizations')
    parser.add_argument('--base-dir', default='.', help='Base directory containing results')
    
    args = parser.parse_args()
    
    # Create and run visualizer
    visualizer = IndividualTestVisualizer(args.base_dir)
    visualizer.generate_all_individual_visualizations()

if __name__ == "__main__":
    main()
