import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import argparse

class GCRCostVisualizer:
    def __init__(self):
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_cost_comparison(self, results_df, series_type="B", save_path=None):
        """Create comprehensive cost comparison charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Google Cloud Run {series_type.upper()}-Series Cost Analysis', fontsize=16, fontweight='bold')
        
        # 1. Total Cost by Configuration
        ax1.bar(results_df['config'], results_df['total_cost'], 
                color=self.colors[:len(results_df)], alpha=0.8)
        ax1.set_title('Total Cost by Configuration')
        ax1.set_ylabel('Cost (USD)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(results_df['total_cost']):
            ax1.text(i, v + max(results_df['total_cost']) * 0.01, f'${v:.6f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Cost Breakdown (Compute vs Requests)
        width = 0.35
        x = np.arange(len(results_df))
        
        ax2.bar(x, results_df['compute_cost'], width, label='Compute Cost', 
                color='lightblue', alpha=0.8)
        ax2.bar(x, results_df['request_cost'], width, bottom=results_df['compute_cost'],
                label='Request Cost', color='lightcoral', alpha=0.8)
        
        ax2.set_title('Cost Breakdown: Compute vs Requests')
        ax2.set_ylabel('Cost (USD)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['config'], rotation=45)
        ax2.legend()
        
        # 3. Cost per Pod
        cost_per_pod = results_df['total_cost'] / results_df['pods']
        ax3.plot(results_df['pods'], cost_per_pod, 'o-', linewidth=2, markersize=8, color='green')
        ax3.set_title('Cost Efficiency: Cost per Pod')
        ax3.set_xlabel('Number of Pods')
        ax3.set_ylabel('Cost per Pod (USD)')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (pods, cost) in enumerate(zip(results_df['pods'], cost_per_pod)):
            ax3.annotate(f'${cost:.6f}', (pods, cost), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        
        # 4. Resource Utilization vs Cost
        # Create color array that matches the number of data points
        scatter_colors = [self.colors[i % len(self.colors)] for i in range(len(results_df))]
        ax4.scatter(results_df['avg_cpu'], results_df['total_cost'], 
                   s=results_df['pods']*20, alpha=0.7, c=scatter_colors)
        ax4.set_title('Resource Utilization vs Cost\n(Bubble size = Pod Count)')
        ax4.set_xlabel('Average CPU Usage (millicores)')
        ax4.set_ylabel('Total Cost (USD)')
        ax4.grid(True, alpha=0.3)
        
        # Add labels for each point
        for i, row in results_df.iterrows():
            ax4.annotate(row['config'], (row['avg_cpu'], row['total_cost']), 
                        textcoords="offset points", xytext=(5,5), ha='left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cost comparison chart saved to {save_path}")
        
        plt.show()
        
    def plot_simple_cost_comparison(self, results_df, series_type="B", save_path=None):
        """Create simplified cost comparison charts (Total Cost + Cost Breakdown only)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Google Cloud Run {series_type.upper()}-Series Simple Cost Analysis', fontsize=16, fontweight='bold')
        
        # 1. Total Cost by Configuration
        ax1.bar(results_df['config'], results_df['total_cost'], 
                color=self.colors[:len(results_df)], alpha=0.8)
        ax1.set_title('Total Cost by Configuration')
        ax1.set_ylabel('Cost (USD)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(results_df['total_cost']):
            ax1.text(i, v + max(results_df['total_cost']) * 0.01, f'${v:.6f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Cost Breakdown (Compute vs Requests)
        width = 0.35
        x = np.arange(len(results_df))
        
        ax2.bar(x, results_df['compute_cost'], width, label='Compute Cost', 
                color='lightblue', alpha=0.8)
        ax2.bar(x, results_df['request_cost'], width, bottom=results_df['compute_cost'],
                label='Request Cost', color='lightcoral', alpha=0.8)
        
        ax2.set_title('Cost Breakdown: Compute vs Requests')
        ax2.set_ylabel('Cost (USD)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['config'], rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Simple cost comparison chart saved to {save_path}")
        
        plt.show()
        
    def plot_resource_timeline(self, calculator, telemetry_files, config_names, save_path=None):
        """Plot resource usage over time for all configurations"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Resource Usage Timeline Comparison', fontsize=16, fontweight='bold')
        
        for i, (file_path, config_name) in enumerate(zip(telemetry_files, config_names)):
            df = calculator.parse_telemetry_file(file_path)
            if df is not None and not df.empty:
                # Convert timestamp to relative time (minutes from start)
                df['relative_time'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 60
                
                color = self.colors[i % len(self.colors)]
                
                # CPU usage timeline
                ax1.plot(df['relative_time'], df['cpu_usage_millicores'], 
                        label=config_name, linewidth=2, color=color)
                
                # Memory usage timeline
                ax2.plot(df['relative_time'], df['memory_usage_mib'], 
                        label=config_name, linewidth=2, color=color)
        
        ax1.set_title('CPU Usage Over Time')
        ax1.set_ylabel('CPU Usage (millicores)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Memory Usage Over Time')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Memory Usage (MiB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Resource timeline chart saved to {save_path}")
        
        plt.show()
        
    def plot_cost_efficiency_analysis(self, results_df, series_type="B", save_path=None):
        """Create detailed cost efficiency analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{series_type.upper()}-Series Cost Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cost per Request
        cost_per_request = results_df['total_cost'] / results_df['total_requests'] * 1000000  # Cost per million requests
        ax1.bar(results_df['config'], cost_per_request, color=self.colors[:len(results_df)], alpha=0.8)
        ax1.set_title('Cost per Million Requests')
        ax1.set_ylabel('Cost per Million Requests (USD)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. CPU Efficiency (Cost per CPU hour)
        cpu_hours = (results_df['avg_cpu'] / 1000) * (results_df['duration_seconds'] / 3600)
        cpu_efficiency = results_df['compute_cost'] / cpu_hours
        ax2.bar(results_df['config'], cpu_efficiency, color='lightgreen', alpha=0.8)
        ax2.set_title('CPU Cost Efficiency ($ per CPU-hour)')
        ax2.set_ylabel('Cost per CPU-hour (USD)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Scaling Efficiency
        ax3.plot(results_df['pods'], results_df['total_cost'], 'o-', linewidth=2, markersize=8)
        ax3.set_title('Cost Scaling with Pod Count')
        ax3.set_xlabel('Number of Pods')
        ax3.set_ylabel('Total Cost (USD)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Resource Distribution
        compute_pct = results_df['compute_percentage']
        request_pct = results_df['request_percentage']
        
        x = np.arange(len(results_df))
        width = 0.6
        
        ax4.bar(x, compute_pct, width, label='Compute %', color='lightblue', alpha=0.8)
        ax4.bar(x, request_pct, width, bottom=compute_pct, label='Request %', color='lightcoral', alpha=0.8)
        ax4.set_title('Cost Distribution: Compute vs Request Charges')
        ax4.set_ylabel('Percentage of Total Cost')
        ax4.set_xticks(x)
        ax4.set_xticklabels(results_df['config'], rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cost efficiency analysis saved to {save_path}")
        
        plt.show()
        
    def create_cost_summary_table(self, results_df, series_type="B", save_path=None):
        """Create a formatted summary table"""
        # Prepare data for table
        summary_data = {
            'Configuration': results_df['config'],
            'Pods': results_df['pods'].astype(int),
            'Avg CPU (m)': results_df['avg_cpu'].round(1),
            'Avg Memory (MiB)': results_df['avg_memory'].round(1),
            'Total Requests': results_df['total_requests'].astype(int),
            'Max Achieved RPS': results_df['max_achieved_rps'].round(1) if 'max_achieved_rps' in results_df.columns else 'N/A',
            'Compute Cost': results_df['compute_cost'].apply(lambda x: f"${x:.6f}"),
            'Request Cost': results_df['request_cost'].apply(lambda x: f"${x:.6f}"),
            'Total Cost': results_df['total_cost'].apply(lambda x: f"${x:.6f}"),
            'Cost/Million Req': (results_df['total_cost'] / results_df['total_requests'] * 1000000).apply(lambda x: f"${x:.2f}"),
            'Cost/Achieved RPS': results_df['cost_per_rps'].apply(lambda x: f"${x:.6f}" if pd.notna(x) else 'N/A')
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_df) + 1):
            for j in range(len(summary_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title(f'{series_type.upper()}-Series Cost Analysis Summary', fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary table saved to {save_path}")
        
        plt.show()
        
        return summary_df
        
    def generate_all_visualizations(self, calculator, data_directory, series_type="B", output_directory=None):
        """Generate all visualizations for series analysis"""
        # Set default output directory
        if output_directory is None:
            output_directory = f"{series_type.lower()}_series_gcr_costs"
        
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        # Process configurations for the specified series
        results = calculator.process_series_configurations(data_directory, series_type)
        
        if not results:
            print("No results to visualize")
            return
        
        results_df = pd.DataFrame(results)
        
        print("Generating visualizations...")
        
        # 1. Cost comparison charts
        self.plot_cost_comparison(
            results_df, 
            series_type,
            save_path=os.path.join(output_directory, "cost_comparison.png")
        )
        
        # 2. Cost efficiency analysis
        self.plot_cost_efficiency_analysis(
            results_df,
            series_type,
            save_path=os.path.join(output_directory, "cost_efficiency.png")
        )
        
        # 3. Summary table
        summary_df = self.create_cost_summary_table(
            results_df,
            series_type,
            save_path=os.path.join(output_directory, "summary_table.png")
        )
        
        # 4. Resource timeline (if telemetry files are available)
        telemetry_files = glob.glob(os.path.join(data_directory, f"*{series_type.lower()}*telemetry*.csv"))
        if telemetry_files:
            # Generate config names based on series type
            if series_type.upper() == 'A':
                config_names = [f"A{i}" for i in [1, 2, 3, 4, 5]][:len(telemetry_files)]
            elif series_type.upper() == 'B':
                config_names = [f"B{i}" for i in [1, 2, 3, 4, 5]][:len(telemetry_files)]
            elif series_type.upper() == 'C':
                config_names = [f"C{i}" for i in [1, 2, 3, 4, 5]][:len(telemetry_files)]
            else:
                config_names = [f"{series_type}{i}" for i in range(1, len(telemetry_files)+1)]
            
            self.plot_resource_timeline(
                calculator,
                telemetry_files[:5],  # Limit to first 5 files
                config_names,
                save_path=os.path.join(output_directory, "resource_timeline.png")
            )
        
        # Save results to CSV
        series_prefix = f"{series_type.lower()}_series"
        results_df.to_csv(os.path.join(output_directory, f"{series_prefix}_results.csv"), index=False)
        summary_df.to_csv(os.path.join(output_directory, f"{series_prefix}_summary.csv"), index=False)
        
        print(f"All visualizations saved to {output_directory}/")
        
        return results_df
        
    def generate_simple_cost_comparison(self, calculator, data_directory, series_type="B", output_directory=None):
        """Generate only the simple cost comparison chart"""
        # Set default output directory
        if output_directory is None:
            output_directory = f"{series_type.lower()}_series_gcr_costs"
        
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        # Process configurations for the specified series
        results = calculator.process_series_configurations(data_directory, series_type)
        
        if not results:
            print("No results to visualize")
            return
        
        results_df = pd.DataFrame(results)
        
        print("Generating simple cost comparison...")
        
        # Generate simple cost comparison chart
        self.plot_simple_cost_comparison(
            results_df, 
            series_type,
            save_path=os.path.join(output_directory, "simple_cost_comparison.png")
        )
        
        # Save results to CSV
        series_prefix = f"{series_type.lower()}_series"
        results_df.to_csv(os.path.join(output_directory, f"{series_prefix}_results.csv"), index=False)
        
        print(f"Simple cost comparison saved to {output_directory}/")
        
        return results_df

# Example usage combining both scripts
if __name__ == "__main__":
    # Import the cost calculator (assuming it's in the same directory)
    from gcr_cost_calculator import GCRCostCalculator
    
    parser = argparse.ArgumentParser(description='GCR Cost Visualizer for A, B, and C Series')
    parser.add_argument('--series', '-s', choices=['A', 'B', 'C'], default='B',
                       help='Series type to analyze (A, B, or C). Default: B')
    parser.add_argument('--data-dir', '-d', default='.',
                       help='Data directory containing results_* directories. Default: current directory')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory for results. Default: {series}_series_gcr_costs')
    parser.add_argument('--simple-cost', action='store_true',
                       help='Generate only simple cost comparison (Total Cost + Cost Breakdown)')
    
    args = parser.parse_args()
    
    # Initialize both calculator and visualizer
    calculator = GCRCostCalculator()
    visualizer = GCRCostVisualizer()
    
    # Set your data directory path
    data_directory = args.data_dir
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{args.series.lower()}_series_gcr_costs"
    
    print(f"Generating visualizations for {args.series}-series configurations...")
    print(f"Data directory: {data_directory}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Generate visualizations based on flag
    if args.simple_cost:
        results_df = visualizer.generate_simple_cost_comparison(calculator, data_directory, args.series, output_dir)
    else:
        results_df = visualizer.generate_all_visualizations(calculator, data_directory, args.series, output_dir)
    
    if results_df is not None:
        print("\nVisualization Summary:")
        print("=====================")
        print(f"Processed {len(results_df)} configurations")
        print(f"Cost range: ${results_df['total_cost'].min():.6f} - ${results_df['total_cost'].max():.6f}")
        print(f"Most cost-effective: {results_df.loc[results_df['total_cost'].idxmin(), 'config']}")
        print(f"Best cost per pod: {results_df.loc[(results_df['total_cost']/results_df['pods']).idxmin(), 'config']}")
