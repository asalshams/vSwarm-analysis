#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os

def create_enhanced_cost_analysis(csv_file, base_output_dir="enhanced_cost_analysis"):
    """Create comprehensive cost analysis with request-based and instance-based billing visualizations."""
    
    # Load data
    df = pd.read_csv(csv_file)
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"Creating enhanced cost analysis from {len(df)} data points...")
    
    # Determine series type
    first_config = df['config'].iloc[0]
    if first_config.startswith('A'):
        series_name = "A-Series"
        config_order = ['A1', 'A2', 'A3', 'A4', 'A5']
        strategy_desc = "Single Pod Scale-Up"
    elif first_config.startswith('B'):
        series_name = "B-Series"
        config_order = ['B1', 'B2', 'B3', 'B4', 'B5']
        strategy_desc = "Multi-Pod Scale-Out"
    elif first_config.startswith('C'):
        series_name = "C-Series"
        config_order = ['C1', 'C2', 'C3', 'C4', 'C5']
        strategy_desc = "Multi-Pod Scale-Out (Higher Resources)"
    else:
        series_name = "Unknown-Series"
        config_order = sorted(df['config'].unique())
        strategy_desc = "Unknown Strategy"
    
    print(f"Detected: {series_name} ({strategy_desc})")
    
    # Create series directory structure
    series_dir = os.path.join(base_output_dir, series_name.lower().replace('-', '_'))
    request_dir = os.path.join(series_dir, "request_based")
    instance_dir = os.path.join(series_dir, "instance_based")
    
    os.makedirs(request_dir, exist_ok=True)
    os.makedirs(instance_dir, exist_ok=True)
    
    # Filter to existing configs and all RPS levels (100-500)
    existing_configs = [c for c in config_order if c in df['config'].values]
    rps_levels = [100, 200, 300, 400, 500]
    runtimes = ['python', 'nodejs', 'go']
    
    # Runtime colors
    RUNTIME_COLORS = {
        "python": "#FFC107",    # Yellow
        "nodejs": "#4CAF50",    # Green
        "go": "#2196F3"         # Blue
    }
    
    print(f"Processing configurations: {existing_configs}")
    print(f"RPS levels: {rps_levels}")
    print(f"Runtimes: {runtimes}")
    
    # Create visualizations for both billing models
    for billing_type in ['request', 'instance']:
        output_dir = request_dir if billing_type == 'request' else instance_dir
        cost_column = f'{billing_type}_cost_per_1k'
        billing_label = "Request-Based" if billing_type == 'request' else "Instance-Based"
        
        print(f"\n=== Creating {billing_label} Billing Visualizations ===")
        
        # 1. Individual RPS Level Comparisons (separate chart for each RPS level)
        create_individual_rps_comparisons(df, output_dir, series_name, strategy_desc, existing_configs, 
                                         cost_column, billing_label, RUNTIME_COLORS, runtimes, rps_levels)
        
        # 2. Comprehensive Breakdown (all RPS levels, all configs)
        create_comprehensive_breakdown(df, output_dir, series_name, existing_configs, 
                                     cost_column, billing_label, RUNTIME_COLORS, runtimes, rps_levels)
        
        # 3. Cost Heatmaps
        create_cost_heatmaps(df, output_dir, series_name, existing_configs, 
                           cost_column, billing_label, runtimes, rps_levels)
        
        # 4. Individual Runtime Charts
        create_individual_runtime_charts(df, output_dir, series_name, existing_configs, 
                                       cost_column, billing_label, RUNTIME_COLORS, runtimes, rps_levels)
    
    print(f"\n✓ Enhanced cost analysis completed!")
    print(f"✓ {series_name} analysis saved in: {series_dir}/")
    print(f"✓ Request-based charts: {request_dir}/")
    print(f"✓ Instance-based charts: {instance_dir}/")

def create_individual_rps_comparisons(df, output_dir, series_name, strategy_desc, existing_configs, 
                                     cost_column, billing_label, RUNTIME_COLORS, runtimes, rps_levels):
    """Create individual cost comparison charts for each RPS level."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a subdirectory for individual RPS charts
    rps_charts_dir = os.path.join(output_dir, "individual_rps_charts")
    os.makedirs(rps_charts_dir, exist_ok=True)
    
    for rps_level in rps_levels:
        # Filter data for this RPS level
        rps_data = df[df['target_rps'] == rps_level]
        if rps_data.empty:
            print(f"No data found for {rps_level} RPS, skipping...")
            continue
            
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Get runtime costs for this RPS level
        runtime_costs = rps_data.groupby(['config', 'runtime']).agg({
            cost_column: 'mean',
            'num_pods': 'first',
            'vcpu_per_pod': 'first',
            'memory_per_pod': 'first'
        }).reset_index()
        
        runtime_costs = runtime_costs[runtime_costs['config'].isin(existing_configs)]
        
        # Create grouped bar chart
        x = np.arange(len(existing_configs))
        width = 0.25
        
        for i, runtime in enumerate(runtimes):
            runtime_data = runtime_costs[runtime_costs['runtime'] == runtime]
            costs = []
            
            for config in existing_configs:
                config_data = runtime_data[runtime_data['config'] == config]
                if not config_data.empty:
                    costs.append(config_data[cost_column].iloc[0])
                else:
                    costs.append(0)
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, costs, width, 
                         label=runtime.capitalize(), 
                         color=RUNTIME_COLORS[runtime],
                         alpha=0.8,
                         edgecolor='black',
                         linewidth=0.5)
            
            # Add value labels
            for j, (bar, cost) in enumerate(zip(bars, costs)):
                if cost > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02,
                           f'${cost:.5f}', ha='center', va='bottom', fontsize=9, rotation=45)
        
        # Formatting
        ax.set_xlabel(f'{series_name} Configuration ({strategy_desc})', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{billing_label} Cost per 1,000 requests ($)', fontsize=14, fontweight='bold')
        ax.set_title(f'{series_name} {billing_label} Cost Comparison at {rps_level} RPS\n' +
                    f'Configuration Performance by Runtime', fontsize=16, fontweight='bold', pad=20)
        
        # Create x-axis labels with resource info
        x_labels = []
        for config in existing_configs:
            if not runtime_costs[runtime_costs['config'] == config].empty:
                config_row = runtime_costs[runtime_costs['config'] == config].iloc[0]
                num_pods = int(config_row['num_pods']) if pd.notna(config_row['num_pods']) else 1
                vcpu = config_row['vcpu_per_pod'] if pd.notna(config_row['vcpu_per_pod']) else 0
                memory = config_row['memory_per_pod'] if pd.notna(config_row['memory_per_pod']) else 0
                
                if series_name == "A-Series":
                    label = f"{config}\n({vcpu}vCPU, {memory}GB)"
                elif series_name == "B-Series":
                    label = f"{config}\n({num_pods} pods, {vcpu}vCPU each)"
                else:  # C-Series
                    label = f"{config}\n({num_pods} pods, {vcpu}vCPU each)"
                x_labels.append(label)
            else:
                x_labels.append(config)
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.legend(title='Runtime', loc='upper left', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        output_path = os.path.join(rps_charts_dir, f'{rps_level}rps_{billing_label.lower().replace("-", "_")}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"{rps_level} RPS {billing_label} comparison saved to: {output_path}")
        plt.close()

def create_comprehensive_breakdown(df, output_dir, series_name, existing_configs, 
                                 cost_column, billing_label, RUNTIME_COLORS, runtimes, rps_levels):
    """Create comprehensive breakdown showing all RPS levels for each config."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create subplots - one for each configuration
    n_configs = len(existing_configs)
    fig, axes = plt.subplots(1, n_configs, figsize=(5*n_configs, 8))
    
    if n_configs == 1:
        axes = [axes]
    
    fig.suptitle(f'{series_name} {billing_label} Cost Breakdown\n' +
                f'All RPS Levels × All Runtimes × All Configurations', 
                fontsize=16, fontweight='bold', y=0.95)
    
    for config_idx, config in enumerate(existing_configs):
        ax = axes[config_idx]
        
        # Get data for this configuration
        config_data = df[df['config'] == config]
        
        # Get resource info
        config_info = config_data.iloc[0]
        num_pods = int(config_info['num_pods']) if pd.notna(config_info['num_pods']) else 1
        vcpu = config_info['vcpu_per_pod'] if pd.notna(config_info['vcpu_per_pod']) else 0
        memory = config_info['memory_per_pod'] if pd.notna(config_info['memory_per_pod']) else 0
        
        # Create grouped bar chart for this config
        x = np.arange(len(rps_levels))
        width = 0.25
        
        for runtime_idx, runtime in enumerate(runtimes):
            runtime_data = config_data[config_data['runtime'] == runtime]
            costs = []
            
            for rps in rps_levels:
                rps_data = runtime_data[runtime_data['target_rps'] == rps]
                if not rps_data.empty:
                    cost = rps_data[cost_column].mean()
                    costs.append(cost)
                else:
                    costs.append(0)
            
            offset = (runtime_idx - 1) * width
            bars = ax.bar(x + offset, costs, width, 
                         label=runtime.capitalize(), 
                         color=RUNTIME_COLORS[runtime],
                         alpha=0.8,
                         edgecolor='black',
                         linewidth=0.5)
            
            # Add value labels on significant bars
            for j, (bar, cost) in enumerate(zip(bars, costs)):
                if cost > 0 and cost > 0.001:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.05,
                           f'${cost:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
        
        # Formatting for this subplot
        ax.set_xlabel('Target RPS', fontsize=11, fontweight='bold')
        if config_idx == 0:
            ax.set_ylabel(f'{billing_label} Cost per 1,000 requests ($)', fontsize=11, fontweight='bold')
        
        # Configuration title with resource info
        if series_name == "A-Series":
            config_title = f'{config}\n({vcpu}vCPU, {memory}GB)'
        elif series_name == "B-Series":
            config_title = f'{config}\n({num_pods} pods, {vcpu}vCPU each)'
        else:  # C-Series
            config_title = f'{config}\n({num_pods} pods, {vcpu}vCPU each)'
        
        ax.set_title(config_title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{rps}' for rps in rps_levels], fontsize=10)
        
        # Only show legend on the last subplot
        if config_idx == len(existing_configs) - 1:
            ax.legend(title='Runtime', loc='upper left', fontsize=10)
        
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'comprehensive_{billing_label.lower().replace("-", "_")}_breakdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comprehensive {billing_label} breakdown saved to: {output_path}")
    plt.close()

def create_cost_heatmaps(df, output_dir, series_name, existing_configs, 
                        cost_column, billing_label, runtimes, rps_levels):
    """Create heatmaps showing costs across configurations and RPS levels."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f'{series_name} {billing_label} Cost Heatmap\n' +
                f'Configuration × RPS Level for Each Runtime', 
                fontsize=16, fontweight='bold')
    
    runtime_labels = ['Python', 'Node.js', 'Go']
    
    for runtime_idx, (runtime, runtime_label) in enumerate(zip(runtimes, runtime_labels)):
        ax = axes[runtime_idx]
        
        # Filter data for this runtime
        runtime_data = df[df['runtime'] == runtime]
        
        # Create pivot table for heatmap
        pivot_data = runtime_data.pivot_table(
            values=cost_column,
            index='config',
            columns='target_rps',
            aggfunc='mean'
        )
        
        # Reindex to ensure proper order
        pivot_data = pivot_data.reindex([c for c in existing_configs if c in pivot_data.index])
        
        # Create heatmap
        sns.heatmap(pivot_data, 
                    annot=True, 
                    fmt='.5f', 
                    cmap='YlOrRd',
                    cbar_kws={'label': f'{billing_label} Cost per 1,000 requests ($)'},
                    ax=ax)
        
        ax.set_title(f'{runtime_label} Runtime', fontsize=14, fontweight='bold')
        ax.set_xlabel('Target RPS', fontsize=12, fontweight='bold')
        
        if runtime_idx == 0:
            ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{billing_label.lower().replace("-", "_")}_heatmap_by_runtime.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"{billing_label} heatmap saved to: {output_path}")
    plt.close()

def create_individual_runtime_charts(df, output_dir, series_name, existing_configs, 
                                   cost_column, billing_label, RUNTIME_COLORS, runtimes, rps_levels):
    """Create individual charts for each runtime showing all configs across all RPS levels."""
    
    for runtime in runtimes:
        runtime_data = df[df['runtime'] == runtime]
        if runtime_data.empty:
            continue
            
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create line plot for each configuration
        for config in existing_configs:
            config_data = runtime_data[runtime_data['config'] == config]
            if config_data.empty:
                continue
                
            # Get costs for each RPS level
            rps_costs = []
            actual_rps = []
            
            for rps in rps_levels:
                rps_data = config_data[config_data['target_rps'] == rps]
                if not rps_data.empty:
                    cost = rps_data[cost_column].mean()
                    rps_costs.append(cost)
                    actual_rps.append(rps)
            
            if rps_costs:
                ax.plot(actual_rps, rps_costs, marker='o', linewidth=2, markersize=6,
                       label=config, alpha=0.8)
                
                # Add value labels on points
                for rps, cost in zip(actual_rps, rps_costs):
                    if cost > 0.001:  # Only label significant costs
                        ax.annotate(f'${cost:.4f}', (rps, cost), 
                                  textcoords="offset points", xytext=(0,10), 
                                  ha='center', fontsize=8, rotation=45)
        
        # Formatting
        ax.set_xlabel('Target RPS', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{billing_label} Cost per 1,000 requests ($)', fontsize=14, fontweight='bold')
        ax.set_title(f'{series_name} {billing_label} Cost Scaling - {runtime.capitalize()} Runtime\n' +
                    f'How Costs Scale with Load Across Different Configurations', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.legend(title='Configuration', fontsize=11, title_fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set x-axis to show all RPS levels
        ax.set_xticks(rps_levels)
        ax.set_xticklabels([f'{rps}' for rps in rps_levels])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{runtime}_{billing_label.lower().replace("-", "_")}_scaling.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"{runtime.capitalize()} {billing_label} scaling chart saved to: {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced cost analysis with request-based and instance-based billing visualizations')
    parser.add_argument('csv_file', help='Path to multi-pod cost analysis CSV file')
    parser.add_argument('--output-dir', '-o', default='enhanced_cost_analysis', 
                       help='Base output directory for analysis')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    try:
        create_enhanced_cost_analysis(args.csv_file, args.output_dir)
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
