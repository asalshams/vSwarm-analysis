import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
import argparse

class GCRCostCalculator:
    def __init__(self):
        # GCR pricing rates (Iowa region, request-based)
        self.CPU_ACTIVE_RATE = 0.000024  # per vCPU-second
        self.MEMORY_ACTIVE_RATE = 0.0000025  # per GiB-second
        self.REQUEST_RATE = 0.40  # per million requests
        self.BILLING_GRANULARITY = 0.1  # 100ms minimum
        
    def parse_telemetry_file(self, filepath):
        """Parse telemetry CSV file and return DataFrame"""
        try:
            df = pd.read_csv(filepath)
            # Convert timestamp to datetime for interval calculation
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
    
    def calculate_interval_cost(self, cpu_millicores, memory_mib, duration_seconds=2):
        """Calculate cost for a single interval"""
        # Convert to GCR billing units
        vcpus = cpu_millicores / 1000
        gib = memory_mib / 1024
        
        # Calculate costs
        cpu_cost = vcpus * self.CPU_ACTIVE_RATE * duration_seconds
        memory_cost = gib * self.MEMORY_ACTIVE_RATE * duration_seconds
        
        return cpu_cost, memory_cost, cpu_cost + memory_cost
    
    def calculate_telemetry_costs(self, df):
        """Calculate total compute costs from telemetry data"""
        if df is None or df.empty:
            return 0, 0, 0, {}
        
        total_cpu_cost = 0
        total_memory_cost = 0
        interval_costs = []
        
        # Calculate time intervals between measurements
        df = df.sort_values('timestamp')
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        df['time_diff'] = df['time_diff'].fillna(2)  # Default 2-second intervals
        
        for _, row in df.iterrows():
            duration = min(row['time_diff'], 10)  # Cap at 10 seconds for outliers
            cpu_cost, mem_cost, total_cost = self.calculate_interval_cost(
                row['cpu_usage_millicores'], 
                row['memory_usage_mib'], 
                duration
            )
            
            total_cpu_cost += cpu_cost
            total_memory_cost += mem_cost
            interval_costs.append({
                'timestamp': row['timestamp'],
                'cpu_usage': row['cpu_usage_millicores'],
                'memory_usage': row['memory_usage_mib'],
                'duration': duration,
                'cpu_cost': cpu_cost,
                'memory_cost': mem_cost,
                'total_cost': total_cost
            })
        
        stats = {
            'intervals': len(interval_costs),
            'total_duration': df['time_diff'].sum(),
            'avg_cpu': df['cpu_usage_millicores'].mean(),
            'max_cpu': df['cpu_usage_millicores'].max(),
            'min_cpu': df['cpu_usage_millicores'].min(),
            'avg_memory': df['memory_usage_mib'].mean(),
            'max_memory': df['memory_usage_mib'].max(),
            'min_memory': df['memory_usage_mib'].min()
        }
        
        return total_cpu_cost, total_memory_cost, total_cpu_cost + total_memory_cost, stats
    
    def parse_performance_file(self, filepath):
        """Parse performance results to get request counts"""
        try:
            df = pd.read_csv(filepath)
            # Sum up all total_requests from the detailed analysis
            total_requests = df['total_requests'].sum()
            return int(total_requests)
        except Exception as e:
            print(f"Error parsing performance file {filepath}: {e}")
            return 0

    def parse_performance_file_extended(self, filepath):
        """Parse performance results to get request counts AND RPS data"""
        try:
            df = pd.read_csv(filepath)
            total_requests = df['total_requests'].sum()
            
            # Get the "Avg of Max" from achieved_rps_summary.csv file
            # The performance file is in the same directory as the summary file
            results_dir = os.path.dirname(filepath)
            summary_file = os.path.join(results_dir, 'achieved_rps_summary.csv')
            
            avg_of_max_rps = None
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Look for "Avg of Max" line
                    for line in lines:
                        if line.startswith('Avg of Max'):
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                avg_of_max_rps = float(parts[1])
                                break
                except Exception as e:
                    print(f"Error reading summary file {summary_file}: {e}")
            else:
                pass
            
            # Fallback: calculate from performance data if summary not available
            if avg_of_max_rps is None:
                if 'throughput_rps' in df.columns:
                    # Group by target_rps and get max achieved RPS for each target
                    max_per_target = df.groupby('target_rps')['throughput_rps'].max()
                    avg_of_max_rps = max_per_target.mean()  # Average of the maximums
                else:
                    avg_of_max_rps = None
            
            avg_achieved_rps = df['throughput_rps'].mean() if 'throughput_rps' in df.columns else None
            
            return int(total_requests), avg_of_max_rps, avg_achieved_rps
        except Exception as e:
            print(f"Error parsing performance file {filepath}: {e}")
            return 0, None, None
    
    def calculate_request_costs(self, total_requests):
        """Calculate request-based costs"""
        return (total_requests / 1000000) * self.REQUEST_RATE
    
    def process_b_series_configuration(self, telemetry_file, performance_file, config_name, expected_pods=None):
        """Process a single configuration"""
        print(f"\nProcessing {config_name}:")
        print("-" * 50)
        
        # Parse telemetry data
        df = self.parse_telemetry_file(telemetry_file)
        if df is None:
            return None
        
        # Calculate compute costs
        cpu_cost, memory_cost, total_compute_cost, stats = self.calculate_telemetry_costs(df)
        
        # Parse request data
        total_requests, avg_of_max_rps, avg_achieved_rps = self.parse_performance_file_extended(performance_file)
        request_cost = self.calculate_request_costs(total_requests)
        
        # Calculate total cost
        total_cost = total_compute_cost + request_cost
        
        # Use expected pods if provided, otherwise read from data
        if expected_pods is not None:
            pods = expected_pods
        else:
            pods = df['pod_count'].iloc[0] if not df.empty else None
        
        # Display results
        print(f"Configuration: {config_name}")
        print(f"Pods: {pods}")
        print(f"Test duration: {stats['total_duration']:.1f} seconds")
        print(f"Telemetry intervals: {stats['intervals']}")
        print()
        print("Resource Usage:")
        print(f"  CPU: {stats['min_cpu']}-{stats['max_cpu']}m (avg: {stats['avg_cpu']:.1f}m)")
        print(f"  Memory: {stats['min_memory']}-{stats['max_memory']} MiB (avg: {stats['avg_memory']:.1f} MiB)")
        print()
        print("Cost Breakdown:")
        print(f"  CPU cost:     ${cpu_cost:.6f}")
        print(f"  Memory cost:  ${memory_cost:.6f}")
        print(f"  Compute total: ${total_compute_cost:.6f}")
        print(f"  Request cost: ${request_cost:.6f} ({total_requests:,} requests)")
        print(f"  TOTAL COST:   ${total_cost:.6f}")
        print()
        print("Cost Distribution:")
        print(f"  Compute: {(total_compute_cost/total_cost*100):.1f}%")
        print(f"  Requests: {(request_cost/total_cost*100):.1f}%")
        
        return {
            'config': config_name,
            'pods': pods,
            'duration_seconds': stats['total_duration'],
            'intervals': stats['intervals'],
            'min_cpu': stats['min_cpu'],
            'max_cpu': stats['max_cpu'],
            'avg_cpu': stats['avg_cpu'],
            'min_memory': stats['min_memory'],
            'max_memory': stats['max_memory'],
            'avg_memory': stats['avg_memory'],
            'cpu_cost': cpu_cost,
            'memory_cost': memory_cost,
            'compute_cost': total_compute_cost,
            'total_requests': total_requests,
            'avg_of_max_rps': avg_of_max_rps,
            'avg_achieved_rps': avg_achieved_rps,
            'cost_per_rps': total_cost / avg_of_max_rps if avg_of_max_rps and avg_of_max_rps > 0 else None,
            'request_cost': request_cost,
            'total_cost': total_cost,
            'compute_percentage': (total_compute_cost/total_cost*100),
            'request_percentage': (request_cost/total_cost*100)
        }
    
    def get_series_configs(self, series_type):
        """Get configuration for different series types"""
        if series_type.upper() == 'A':
            return [
                {'name': 'Python A1 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_python_a1', 'telemetry_file': 'fibonacci_python_a1_pod_monitoring.csv', 'perf_file': 'fibonacci_python_a1_detailed_analysis.csv'},
                {'name': 'Python A2 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_python_a2', 'telemetry_file': 'fibonacci_python_a2_pod_monitoring.csv', 'perf_file': 'fibonacci_python_a2_detailed_analysis.csv'},
                {'name': 'Python A3 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_python_a3', 'telemetry_file': 'fibonacci_python_a3_pod_monitoring.csv', 'perf_file': 'fibonacci_python_a3_detailed_analysis.csv'},
                {'name': 'Python A4 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_python_a4', 'telemetry_file': 'fibonacci_python_a4_pod_monitoring.csv', 'perf_file': 'fibonacci_python_a4_detailed_analysis.csv'},
                {'name': 'Python A5 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_python_a5', 'telemetry_file': 'fibonacci_python_a5_pod_monitoring.csv', 'perf_file': 'fibonacci_python_a5_detailed_analysis.csv'},
                {'name': 'NodeJS A1 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_nodejs_a1', 'telemetry_file': 'fibonacci_nodejs_a1_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_a1_detailed_analysis.csv'},
                {'name': 'NodeJS A2 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_nodejs_a2', 'telemetry_file': 'fibonacci_nodejs_a2_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_a2_detailed_analysis.csv'},
                {'name': 'NodeJS A3 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_nodejs_a3', 'telemetry_file': 'fibonacci_nodejs_a3_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_a3_detailed_analysis.csv'},
                {'name': 'NodeJS A4 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_nodejs_a4', 'telemetry_file': 'fibonacci_nodejs_a4_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_a4_detailed_analysis.csv'},
                {'name': 'NodeJS A5 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_nodejs_a5', 'telemetry_file': 'fibonacci_nodejs_a5_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_a5_detailed_analysis.csv'},
                {'name': 'Go A1 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_go_a1', 'telemetry_file': 'fibonacci_go_a1_pod_monitoring.csv', 'perf_file': 'fibonacci_go_a1_detailed_analysis.csv'},
                {'name': 'Go A2 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_go_a2', 'telemetry_file': 'fibonacci_go_a2_pod_monitoring.csv', 'perf_file': 'fibonacci_go_a2_detailed_analysis.csv'},
                {'name': 'Go A3 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_go_a3', 'telemetry_file': 'fibonacci_go_a3_pod_monitoring.csv', 'perf_file': 'fibonacci_go_a3_detailed_analysis.csv'},
                {'name': 'Go A4 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_go_a4', 'telemetry_file': 'fibonacci_go_a4_pod_monitoring.csv', 'perf_file': 'fibonacci_go_a4_detailed_analysis.csv'},
                {'name': 'Go A5 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_go_a5', 'telemetry_file': 'fibonacci_go_a5_pod_monitoring.csv', 'perf_file': 'fibonacci_go_a5_detailed_analysis.csv'}
            ]
        elif series_type.upper() == 'B':
            return [
                {'name': 'Python B1 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_python_b1', 'telemetry_file': 'fibonacci_python_b1_pod_monitoring.csv', 'perf_file': 'fibonacci_python_b1_detailed_analysis.csv'},
                {'name': 'Python B2 (2 pods)', 'pods': 2, 'dir_pattern': 'results_fibonacci_python_b2', 'telemetry_file': 'fibonacci_python_b2_pod_monitoring.csv', 'perf_file': 'fibonacci_python_b2_detailed_analysis.csv'},
                {'name': 'Python B3 (4 pods)', 'pods': 4, 'dir_pattern': 'results_fibonacci_python_b3', 'telemetry_file': 'fibonacci_python_b3_pod_monitoring.csv', 'perf_file': 'fibonacci_python_b3_detailed_analysis.csv'},
                {'name': 'Python B4 (8 pods)', 'pods': 8, 'dir_pattern': 'results_fibonacci_python_b4', 'telemetry_file': 'fibonacci_python_b4_pod_monitoring.csv', 'perf_file': 'fibonacci_python_b4_detailed_analysis.csv'},
                {'name': 'Python B5 (10 pods)', 'pods': 10, 'dir_pattern': 'results_fibonacci_python_b5', 'telemetry_file': 'fibonacci_python_b5_pod_monitoring.csv', 'perf_file': 'fibonacci_python_b5_detailed_analysis.csv'},
                {'name': 'NodeJS B1 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_nodejs_b1', 'telemetry_file': 'fibonacci_nodejs_b1_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_b1_detailed_analysis.csv'},
                {'name': 'NodeJS B2 (2 pods)', 'pods': 2, 'dir_pattern': 'results_fibonacci_nodejs_b2', 'telemetry_file': 'fibonacci_nodejs_b2_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_b2_detailed_analysis.csv'},
                {'name': 'NodeJS B3 (4 pods)', 'pods': 4, 'dir_pattern': 'results_fibonacci_nodejs_b3', 'telemetry_file': 'fibonacci_nodejs_b3_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_b3_detailed_analysis.csv'},
                {'name': 'NodeJS B4 (8 pods)', 'pods': 8, 'dir_pattern': 'results_fibonacci_nodejs_b4', 'telemetry_file': 'fibonacci_nodejs_b4_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_b4_detailed_analysis.csv'},
                {'name': 'NodeJS B5 (10 pods)', 'pods': 10, 'dir_pattern': 'results_fibonacci_nodejs_b5', 'telemetry_file': 'fibonacci_nodejs_b5_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_b5_detailed_analysis.csv'},
                {'name': 'Go B1 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_go_b1', 'telemetry_file': 'fibonacci_go_b1_pod_monitoring.csv', 'perf_file': 'fibonacci_go_b1_detailed_analysis.csv'},
                {'name': 'Go B2 (2 pods)', 'pods': 2, 'dir_pattern': 'results_fibonacci_go_b2', 'telemetry_file': 'fibonacci_go_b2_pod_monitoring.csv', 'perf_file': 'fibonacci_go_b2_detailed_analysis.csv'},
                {'name': 'Go B3 (4 pods)', 'pods': 4, 'dir_pattern': 'results_fibonacci_go_b3', 'telemetry_file': 'fibonacci_go_b3_pod_monitoring.csv', 'perf_file': 'fibonacci_go_b3_detailed_analysis.csv'},
                {'name': 'Go B4 (8 pods)', 'pods': 8, 'dir_pattern': 'results_fibonacci_go_b4', 'telemetry_file': 'fibonacci_go_b4_pod_monitoring.csv', 'perf_file': 'fibonacci_go_b4_detailed_analysis.csv'},
                {'name': 'Go B5 (10 pods)', 'pods': 10, 'dir_pattern': 'results_fibonacci_go_b5', 'telemetry_file': 'fibonacci_go_b5_pod_monitoring.csv', 'perf_file': 'fibonacci_go_b5_detailed_analysis.csv'}
            ]
        elif series_type.upper() == 'C':
            return [
                {'name': 'Python C1 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_python_c1', 'telemetry_file': 'fibonacci_python_c1_pod_monitoring.csv', 'perf_file': 'fibonacci_python_c1_detailed_analysis.csv'},
                {'name': 'Python C2 (2 pods)', 'pods': 2, 'dir_pattern': 'results_fibonacci_python_c2', 'telemetry_file': 'fibonacci_python_c2_pod_monitoring.csv', 'perf_file': 'fibonacci_python_c2_detailed_analysis.csv'},
                {'name': 'Python C3 (4 pods)', 'pods': 4, 'dir_pattern': 'results_fibonacci_python_c3', 'telemetry_file': 'fibonacci_python_c3_pod_monitoring.csv', 'perf_file': 'fibonacci_python_c3_detailed_analysis.csv'},
                {'name': 'Python C4 (7 pods)', 'pods': 7, 'dir_pattern': 'results_fibonacci_python_c4', 'telemetry_file': 'fibonacci_python_c4_pod_monitoring.csv', 'perf_file': 'fibonacci_python_c4_detailed_analysis.csv'},
                {'name': 'Python C5 (8 pods)', 'pods': 8, 'dir_pattern': 'results_fibonacci_python_c5', 'telemetry_file': 'fibonacci_python_c5_pod_monitoring.csv', 'perf_file': 'fibonacci_python_c5_detailed_analysis.csv'},
                {'name': 'NodeJS C1 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_nodejs_c1', 'telemetry_file': 'fibonacci_nodejs_c1_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_c1_detailed_analysis.csv'},
                {'name': 'NodeJS C2 (2 pods)', 'pods': 2, 'dir_pattern': 'results_fibonacci_nodejs_c2', 'telemetry_file': 'fibonacci_nodejs_c2_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_c2_detailed_analysis.csv'},
                {'name': 'NodeJS C3 (4 pods)', 'pods': 4, 'dir_pattern': 'results_fibonacci_nodejs_c3', 'telemetry_file': 'fibonacci_nodejs_c3_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_c3_detailed_analysis.csv'},
                {'name': 'NodeJS C4 (7 pods)', 'pods': 7, 'dir_pattern': 'results_fibonacci_nodejs_c4', 'telemetry_file': 'fibonacci_nodejs_c4_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_c4_detailed_analysis.csv'},
                {'name': 'NodeJS C5 (8 pods)', 'pods': 8, 'dir_pattern': 'results_fibonacci_nodejs_c5', 'telemetry_file': 'fibonacci_nodejs_c5_pod_monitoring.csv', 'perf_file': 'fibonacci_nodejs_c5_detailed_analysis.csv'},
                {'name': 'Go C1 (1 pod)', 'pods': 1, 'dir_pattern': 'results_fibonacci_go_c1', 'telemetry_file': 'fibonacci_go_c1_pod_monitoring.csv', 'perf_file': 'fibonacci_go_c1_detailed_analysis.csv'},
                {'name': 'Go C2 (2 pods)', 'pods': 2, 'dir_pattern': 'results_fibonacci_go_c2', 'telemetry_file': 'fibonacci_go_c2_pod_monitoring.csv', 'perf_file': 'fibonacci_go_c2_detailed_analysis.csv'},
                {'name': 'Go C3 (4 pods)', 'pods': 4, 'dir_pattern': 'results_fibonacci_go_c3', 'telemetry_file': 'fibonacci_go_c3_pod_monitoring.csv', 'perf_file': 'fibonacci_go_c3_detailed_analysis.csv'},
                {'name': 'Go C4 (7 pods)', 'pods': 7, 'dir_pattern': 'results_fibonacci_go_c4', 'telemetry_file': 'fibonacci_go_c4_pod_monitoring.csv', 'perf_file': 'fibonacci_go_c4_detailed_analysis.csv'},
                {'name': 'Go C5 (8 pods)', 'pods': 8, 'dir_pattern': 'results_fibonacci_go_c5', 'telemetry_file': 'fibonacci_go_c5_pod_monitoring.csv', 'perf_file': 'fibonacci_go_c5_detailed_analysis.csv'}
            ]
        else:
            raise ValueError(f"Unsupported series type: {series_type}. Use 'A', 'B', or 'C'")

    def process_series_configurations(self, data_directory, series_type):
        """Process configurations for a specific series type"""
        series_configs = self.get_series_configs(series_type)
        
        results = []
        
        for config in series_configs:
            # Check if directory exists
            config_dir = os.path.join(data_directory, config['dir_pattern'])
            if not os.path.exists(config_dir):
                print(f"Directory not found: {config['dir_pattern']}")
                continue
                
            # Find telemetry and performance files
            telemetry_file = os.path.join(config_dir, config['telemetry_file'])
            perf_file = os.path.join(config_dir, config['perf_file'])
            
            if os.path.exists(telemetry_file) and os.path.exists(perf_file):
                result = self.process_b_series_configuration(
                    telemetry_file, 
                    perf_file, 
                    config['name'],
                    config['pods']  # Use predefined pod count
                )
                if result:
                    results.append(result)
            else:
                print(f"Files not found for {config['name']}")
                print(f"  Telemetry file: {telemetry_file}")
                print(f"  Performance file: {perf_file}")
        
        return results

    def process_all_b_series(self, data_directory):
        """Process all B-series configurations in a directory (legacy method)"""
        return self.process_series_configurations(data_directory, 'B')
    
    def generate_comparison_report(self, results, series_type="B"):
        """Generate a comparison report for all configurations"""
        if not results:
            print("No results to compare")
            return
        
        print("\n" + "="*80)
        print(f"{series_type.upper()}-SERIES COST COMPARISON REPORT")
        print("="*80)
        
        df = pd.DataFrame(results)
        
        print(f"\n{'Config':<15} {'Pods':<5} {'Duration':<10} {'Avg CPU':<10} {'Total Cost':<12} {'$/Pod':<10}")
        print("-" * 70)
        
        for _, row in df.iterrows():
            cost_per_pod = row['total_cost'] / row['pods'] if row['pods'] > 0 else 0
            print(f"{row['config']:<15} {row['pods']:<5} {row['duration_seconds']:<10.1f} {row['avg_cpu']:<10.1f} ${row['total_cost']:<11.6f} ${cost_per_pod:<9.6f}")
        
        print(f"\nCost Efficiency Analysis:")
        print(f"Most cost-effective: {df.loc[df['total_cost'].idxmin(), 'config']}")
        print(f"Best cost per pod: {df.loc[(df['total_cost']/df['pods']).idxmin(), 'config']}")
        print(f"Highest throughput: {df.loc[df['total_requests'].idxmax(), 'config']}")
        
        return df

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCR Cost Calculator for A, B, and C Series')
    parser.add_argument('--series', '-s', choices=['A', 'B', 'C'], default='B',
                       help='Series type to analyze (A, B, or C). Default: B')
    parser.add_argument('--data-dir', '-d', default='.',
                       help='Data directory containing results_* directories. Default: current directory')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory for results. Default: {series}_series_gcr_costs')
    
    args = parser.parse_args()
    
    calculator = GCRCostCalculator()
    
    # Use current directory as data directory
    data_directory = args.data_dir
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{args.series.lower()}_series_gcr_costs"
    
    print(f"Processing {args.series}-series configurations...")
    print(f"Data directory: {data_directory}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    results = calculator.process_series_configurations(data_directory, args.series)
    
    if results:
        comparison_df = calculator.generate_comparison_report(results, args.series)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to CSV
        if comparison_df is not None:
            series_prefix = f"{args.series.lower()}_series"
            comparison_df.to_csv(os.path.join(output_dir, f'{series_prefix}_cost_analysis.csv'), index=False)
            print(f"\nResults saved to '{output_dir}/{series_prefix}_cost_analysis.csv'")
            
            # Also save detailed results with all metrics
            detailed_results = []
            for result in results:
                detailed_results.append({
                    'Configuration': result['config'],
                    'Pods': result['pods'],
                    'Duration_Seconds': result['duration_seconds'],
                    'Intervals': result['intervals'],
                    'Min_CPU_millicores': result['min_cpu'],
                    'Max_CPU_millicores': result['max_cpu'],
                    'Avg_CPU_millicores': result['avg_cpu'],
                    'Min_Memory_MiB': result['min_memory'],
                    'Max_Memory_MiB': result['max_memory'],
                    'Avg_Memory_MiB': result['avg_memory'],
                    'CPU_Cost_USD': result['cpu_cost'],
                    'Memory_Cost_USD': result['memory_cost'],
                    'Compute_Cost_USD': result['compute_cost'],
                    'Total_Requests': result['total_requests'],
                    'Request_Cost_USD': result['request_cost'],
                    'Total_Cost_USD': result['total_cost'],
                    'Compute_Percentage': result['compute_percentage'],
                    'Request_Percentage': result['request_percentage'],
                    'Cost_Per_Pod_USD': result['total_cost'] / result['pods'] if result['pods'] > 0 else 0,
                    'Cost_Per_Million_Requests_USD': (result['total_cost'] / result['total_requests'] * 1000000) if result['total_requests'] > 0 else 0
                })
            
            detailed_df = pd.DataFrame(detailed_results)
            detailed_df.to_csv(os.path.join(output_dir, f'{series_prefix}_detailed_cost_analysis.csv'), index=False)
            print(f"Detailed results saved to '{output_dir}/{series_prefix}_detailed_cost_analysis.csv'")
            
            # Save summary report to text file
            with open(os.path.join(output_dir, f'{series_prefix}_cost_summary_report.txt'), 'w') as f:
                f.write(f"{args.series.upper()}-SERIES COST ANALYSIS SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Configurations Analyzed: {len(results)}\n\n")
                
                f.write("COST EFFICIENCY RANKINGS:\n")
                f.write("-" * 30 + "\n")
                
                # Most cost-effective (lowest total cost)
                most_cost_effective = min(results, key=lambda x: x['total_cost'])
                f.write(f"Most Cost-Effective: {most_cost_effective['config']} (${most_cost_effective['total_cost']:.6f})\n")
                
                # Best cost per pod
                best_cost_per_pod = min(results, key=lambda x: x['total_cost'] / x['pods'] if x['pods'] > 0 else float('inf'))
                f.write(f"Best Cost per Pod: {best_cost_per_pod['config']} (${best_cost_per_pod['total_cost'] / best_cost_per_pod['pods']:.6f})\n")
                
                # Highest throughput
                highest_throughput = max(results, key=lambda x: x['total_requests'])
                f.write(f"Highest Throughput: {highest_throughput['config']} ({highest_throughput['total_requests']:,} requests)\n\n")
                
                f.write("DETAILED RESULTS:\n")
                f.write("-" * 20 + "\n")
                for result in results:
                    f.write(f"\n{result['config']}:\n")
                    f.write(f"  Pods: {result['pods']}\n")
                    f.write(f"  Duration: {result['duration_seconds']:.1f} seconds\n")
                    f.write(f"  Avg CPU: {result['avg_cpu']:.1f} millicores\n")
                    f.write(f"  Avg Memory: {result['avg_memory']:.1f} MiB\n")
                    f.write(f"  Total Requests: {result['total_requests']:,}\n")
                    f.write(f"  CPU Cost: ${result['cpu_cost']:.6f}\n")
                    f.write(f"  Memory Cost: ${result['memory_cost']:.6f}\n")
                    f.write(f"  Request Cost: ${result['request_cost']:.6f}\n")
                    f.write(f"  Total Cost: ${result['total_cost']:.6f}\n")
                    f.write(f"  Cost per Pod: ${result['total_cost'] / result['pods']:.6f}\n")
                    f.write(f"  Cost per Million Requests: ${(result['total_cost'] / result['total_requests'] * 1000000):.2f}\n")
            
            print(f"Summary report saved to '{output_dir}/{series_prefix}_cost_summary_report.txt'")
    else:
        print("No configurations processed successfully")
        print("Please check your file paths and naming conventions")
