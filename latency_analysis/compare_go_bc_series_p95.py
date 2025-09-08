#!/usr/bin/env python3
"""
Script to compare Go B series and C series P95 latency vs throughput curves
on the same axes, with B series curves being lighter/dotted and C series being bold.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import re
from pathlib import Path

def load_series_data(series, language="go"):
    """Load aggregated data for a specific series and language."""
    
    # Find the aggregated CSV file
    if series.upper() == "B":
        csv_pattern = f"b_series_visualizations_results/{language}_b_series/fibonacci_{language}_b_aggregated.csv"
    elif series.upper() == "C":
        csv_pattern = f"c_series_visualizations_results/{language}_c_series/fibonacci_{language}_c_aggregated.csv"
    else:
        print(f"Error: Unknown series {series}")
        return None
    
    if not os.path.exists(csv_pattern):
        print(f"Error: {csv_pattern} not found")
        return None
    
    try:
        # Skip comment lines that start with #
        df = pd.read_csv(csv_pattern, comment='#')
        print(f"Loaded {csv_pattern}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading {csv_pattern}: {e}")
        return None

def create_go_bc_p95_comparison():
    """Create a comparison chart with Go B and C series P95 latency vs throughput curves."""
    
    # Load data for both series
    b_data = load_series_data("B", "go")
    c_data = load_series_data("C", "go")
    
    if b_data is None or c_data is None:
        print("Error: Could not load data for both series")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define colors for configurations
    colors = {
        'B1': '#1f77b4',  # Blue
        'B2': '#ff7f0e',  # Orange
        'B3': '#2ca02c',  # Green
        'B4': '#d62728',  # Red
        'B5': '#9467bd',  # Purple
        'C1': '#1f77b4',  # Blue (same as B1)
        'C2': '#ff7f0e',  # Orange (same as B2)
        'C3': '#2ca02c',  # Green (same as B3)
        'C4': '#d62728',  # Red (same as B4)
        'C5': '#9467bd',  # Purple (same as B5)
    }
    
    # Plot B series data (lighter/dotted)
    b_configs = sorted(b_data['run'].unique())
    for config in b_configs:
        config_data = b_data[b_data['run'] == config]
        
        # Filter out NaN values
        mask = ~(config_data['p95_ms'].isna() | config_data['throughput_rps'].isna())
        x = config_data.loc[mask, 'throughput_rps']
        y = config_data.loc[mask, 'p95_ms']
        
        if len(x) > 0:
            plt.plot(x, y, 
                    color=colors[config], 
                    linestyle='--',  # Dotted line
                    alpha=0.7,       # Lighter
                    linewidth=2,
                    label=f'{config} (B Series)',
                    marker='o',
                    markersize=4)
    
    # Plot C series data (bold/solid)
    c_configs = sorted(c_data['run'].unique())
    for config in c_configs:
        config_data = c_data[c_data['run'] == config]
        
        # Filter out NaN values
        mask = ~(config_data['p95_ms'].isna() | config_data['throughput_rps'].isna())
        x = config_data.loc[mask, 'throughput_rps']
        y = config_data.loc[mask, 'p95_ms']
        
        if len(x) > 0:
            plt.plot(x, y, 
                    color=colors[config], 
                    linestyle='-',   # Solid line
                    alpha=1.0,       # Bold
                    linewidth=3,
                    label=f'{config} (C Series)',
                    marker='s',
                    markersize=5)
    
    # Customize the plot
    plt.xlabel('Throughput (RPS)', fontsize=12, fontweight='bold')
    plt.ylabel('P95 Latency (ms)', fontsize=12, fontweight='bold')
    plt.title('Go Series Comparison: P95 Latency vs Throughput\nB Series (Single-threading) vs C Series (Multi-pod Scaling)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Set y-axis limit to 500ms as requested
    plt.ylim(0, 500)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the comparison chart
    output_path = "go_bc_series_p95_latency_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comparison chart saved as: {output_path}")
    
    # Show the plot
    plt.show()

def main():
    """Main function to run the comparison script."""
    print("Creating Go B vs C series P95 latency vs throughput comparison...")
    create_go_bc_p95_comparison()
    print("Comparison complete!")

if __name__ == "__main__":
    main()
