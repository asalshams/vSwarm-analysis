#!/usr/bin/env python3
import pandas as pd
import argparse
import sys
import os
import glob
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class MultiPodCostCalculator:
    """
    Calculate Cloud Run costs for A, B, and C series configurations.
    
    A-series: Single pod scale-up (varying resources per pod, containerConcurrency=1)
    B-series: Multi-pod scale-out (constant total resources, varying containerConcurrency)
    C-series: Multi-pod scale-out (constant total resources, varying containerConcurrency)
    """
    
    # Google Cloud Run Pricing (corrected rates)
    INSTANCE_CPU_PRICE_PER_VCPU_SEC = 0.000018    # $0.000018 per vCPU-second
    INSTANCE_MEM_PRICE_PER_GIB_SEC = 0.000002     # $0.000002 per GiB-second
    REQUEST_CPU_PRICE_PER_VCPU_SEC = 0.000024     # $0.000024 per vCPU-second
    REQUEST_MEM_PRICE_PER_GIB_SEC = 0.0000025     # $0.0000025 per GiB-second
    REQUEST_PER_REQUEST_FEE = 0.0004               # $0.40/M = $0.0004 per 1k requests
    
    def __init__(self, stats_pattern, config_file_path=None, output_dir="multi_pod_analysis"):
        self.stats_pattern = stats_pattern
        self.config_file_path = config_file_path
        self.output_dir = output_dir
        self.config_df = None
        self.all_results = []
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define all series configurations
        self.series_configs = {
            # A-Series: Single pod scale-up
            'A1': {'vcpu': 1.0, 'memory_gib': 1.0, 'pods': 1, 'concurrency': 1},
            'A2': {'vcpu': 0.5, 'memory_gib': 0.5, 'pods': 1, 'concurrency': 1},
            'A3': {'vcpu': 0.25, 'memory_gib': 0.25, 'pods': 1, 'concurrency': 1},
            'A4': {'vcpu': 0.125, 'memory_gib': 0.125, 'pods': 1, 'concurrency': 1},
            'A5': {'vcpu': 0.1, 'memory_gib': 0.1, 'pods': 1, 'concurrency': 1},
            
            # B-Series: Multi-pod scale-out, constant total resources
            'B1': {'vcpu': 1.0, 'memory_gib': 1.0, 'pods': 1, 'concurrency': 100},
            'B2': {'vcpu': 0.5, 'memory_gib': 0.5, 'pods': 2, 'concurrency': 50},
            'B3': {'vcpu': 0.25, 'memory_gib': 0.25, 'pods': 4, 'concurrency': 25},
            'B4': {'vcpu': 0.125, 'memory_gib': 0.125, 'pods': 8, 'concurrency': 12},
            'B5': {'vcpu': 0.1, 'memory_gib': 0.1, 'pods': 10, 'concurrency': 10},
            
            # C-Series: Multi-pod scale-out with varying containerConcurrency
            'C1': {'vcpu': 2.0, 'memory_gib': 2.0, 'pods': 1, 'concurrency': 200},
            'C2': {'vcpu': 1.0, 'memory_gib': 1.0, 'pods': 2, 'concurrency': 100},
            'C3': {'vcpu': 0.5, 'memory_gib': 0.5, 'pods': 4, 'concurrency': 50},
            'C4': {'vcpu': 0.25, 'memory_gib': 0.25, 'pods': 8, 'concurrency': 25},
            'C5': {'vcpu': 0.2, 'memory_gib': 0.2, 'pods': 10, 'concurrency': 20},
        }
    
    def extract_runtime_and_config(self, filename):
        """Extract runtime and configuration from filename."""
        base_name = os.path.basename(filename).lower()
        
        # Extract runtime
        runtime = "unknown"
        if "python" in base_name:
            runtime = "python"
        elif "nodejs" in base_name or "node" in base_name:
            runtime = "nodejs"
        elif "go" in base_name:
            runtime = "go"
        
        # Extract configuration (A1-A5, B1-B5, C1-C5)
        config = "unknown"
        for cfg in ["a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4", "b5", "c1", "c2", "c3", "c4", "c5"]:
            if cfg in base_name:
                config = cfg.upper()
                break
        
        return runtime, config
    
    def get_series_from_config(self, config):
        """Extract series (A, B, C) from configuration."""
        if config.startswith('A'):
            return 'A'
        elif config.startswith('B'):
            return 'B' 
        elif config.startswith('C'):
            return 'C'
        return 'Unknown'
    
    def create_achieved_rps_lookup(self, achieved_rps_df):
        """Create a lookup dictionary for achieved RPS by target RPS."""
        if achieved_rps_df is None:
            return {}
        
        lookup = {}
        for _, row in achieved_rps_df.iterrows():
            # Skip header rows and non-numeric data
            try:
                target_rps = float(row['target_rps'])
                avg_achieved = float(row['avg_achieved'])
                efficiency_pct = float(row.get('efficiency_pct', 0))
                lookup[target_rps] = {
                    'avg_achieved': avg_achieved,
                    'efficiency_pct': efficiency_pct
                }
            except (ValueError, TypeError):
                # Skip non-numeric rows (headers, summaries, etc.)
                continue
        return lookup
    
    def find_statistics_files(self):
        """Find all statistics files matching the pattern."""
        files = glob.glob(self.stats_pattern)
        if not files:
            print(f"No statistics files found matching pattern: {self.stats_pattern}")
            return []
        
        print(f"Found {len(files)} statistics files:")
        for f in sorted(files):
            print(f"  - {f}")
        return sorted(files)
    
    def calculate_costs_for_file(self, stats_file, capacity_threshold=95.0):
        """Calculate costs for a single statistics file with multi-pod support."""
        print(f"\n--- Processing: {os.path.basename(stats_file)} ---")
        
        try:
            stats_df = pd.read_csv(stats_file)
        except Exception as e:
            print(f"Error reading {stats_file}: {e}")
            return []
        
        # Load achieved RPS data from the corresponding achieved_rps_summary.csv
        stats_dir = os.path.dirname(stats_file)
        achieved_rps_file = os.path.join(stats_dir, "achieved_rps_summary.csv")
        achieved_rps_df = None
        
        if os.path.exists(achieved_rps_file):
            try:
                achieved_rps_df = pd.read_csv(achieved_rps_file)
                achieved_rps_lookup = self.create_achieved_rps_lookup(achieved_rps_df)
                print(f"Found achieved RPS data: {achieved_rps_file}")
                print(f"Loaded achieved RPS for targets: {list(achieved_rps_lookup.keys())}")
            except Exception as e:
                print(f"Warning: Could not read achieved RPS file {achieved_rps_file}: {e}")
                achieved_rps_lookup = {}
        else:
            print(f"Warning: No achieved RPS file found at {achieved_rps_file}")
            print("Using throughput_rps from statistics file (may be inaccurate at high loads)")
            achieved_rps_lookup = {}
        
        runtime, config = self.extract_runtime_and_config(stats_file)
        series = self.get_series_from_config(config)
        print(f"Detected: Runtime={runtime}, Config={config}, Series={series}")
        
        # Get configuration details
        if config not in self.series_configs:
            print(f"Warning: Unknown configuration {config}")
            return []
        
        config_details = self.series_configs[config]
        vcpu_per_pod = config_details['vcpu']
        memory_per_pod = config_details['memory_gib'] 
        num_pods = config_details['pods']
        concurrency = config_details['concurrency']
        
        print(f"Configuration: {vcpu_per_pod}vCPU × {memory_per_pod}GB × {num_pods} pods (concurrency: {concurrency}/pod)")
        
        # Filter to multiple RPS levels for analysis
        rps_levels = [100, 200, 300, 400, 500]
        benchmark_rows = stats_df[stats_df['rps'].isin(rps_levels)]
        
        if benchmark_rows.empty:
            print(f"No data found for RPS levels {rps_levels}")
            return []
        
        print(f"Found data for RPS levels: {sorted(benchmark_rows['rps'].unique())}")
        
        results = []
        
        for _, row in benchmark_rows.iterrows():
            target_rps = float(row['rps'])
            
            # Get achieved RPS from achieved_rps_summary.csv if available, otherwise use throughput_rps
            if target_rps in achieved_rps_lookup:
                achieved_rps = achieved_rps_lookup[target_rps]['avg_achieved']
                efficiency_pct = achieved_rps_lookup[target_rps]['efficiency_pct']
                print(f"  Using achieved RPS from summary: {target_rps} -> {achieved_rps:.2f} ({efficiency_pct:.1f}% efficiency)")
            else:
                achieved_rps = row['throughput_rps']
                print(f"  No achieved RPS data for {target_rps}, using throughput_rps: {achieved_rps:.2f}")
            
            if achieved_rps < capacity_threshold:
                print(f"Skipping {target_rps} RPS: achieved {achieved_rps:.1f} RPS < {capacity_threshold} threshold")
                continue
            
            # Calculate costs with multi-pod considerations
            duration_seconds = 1000 / achieved_rps
            latency_us = row['average']  # Use average latency instead of min for realistic load calculation
            latency_seconds = latency_us / 1_000_000
            
            # Instance-based cost: ALL pods run for the entire duration
            total_vcpu = vcpu_per_pod * num_pods
            total_memory = memory_per_pod * num_pods
            
            instance_cost = duration_seconds * (
                total_vcpu * self.INSTANCE_CPU_PRICE_PER_VCPU_SEC +
                total_memory * self.INSTANCE_MEM_PRICE_PER_GIB_SEC
            )
            
            # Calculate busy pods using Little's Law for reporting (achieved RPS × latency)
            N_in_service = achieved_rps * latency_seconds  # Little's law: λ × L
            busy_pods = min(num_pods, math.ceil(N_in_service / concurrency))
            
            # Calculate fleet utilization
            total_fleet_capacity = num_pods * concurrency
            fleet_utilization = N_in_service / total_fleet_capacity if total_fleet_capacity > 0 else 0
            
            print(f"  {target_rps} RPS: {busy_pods}/{num_pods} pods busy (avg latency: {latency_us:.0f}μs, N_in_service: {N_in_service:.1f}, utilization: {fleet_utilization:.1%})")
            
            # Request-based cost: Pay per request processing time (independent of pod count)
            request_cost = (
                1000 * latency_seconds * (
                    vcpu_per_pod * self.REQUEST_CPU_PRICE_PER_VCPU_SEC +
                    memory_per_pod * self.REQUEST_MEM_PRICE_PER_GIB_SEC
                ) +
                self.REQUEST_PER_REQUEST_FEE  # Fee per 1k requests
            )
            
            cost_difference = instance_cost - request_cost
            percent_savings = (cost_difference / instance_cost) * 100 if instance_cost > 0 else 0
            
            # Calculate cost per pod for analysis
            instance_cost_per_pod = instance_cost / num_pods
            
            # Get efficiency percentage
            efficiency_pct = achieved_rps_lookup.get(target_rps, {}).get('efficiency_pct', achieved_rps/target_rps*100)
            
            results.append({
                'statistics_file': os.path.basename(stats_file),
                'runtime': runtime,
                'config': config,
                'series': series,
                'vcpu_per_pod': vcpu_per_pod,
                'memory_per_pod': memory_per_pod,
                'num_pods': num_pods,
                'busy_pods': busy_pods,
                'N_in_service': N_in_service,
                'fleet_utilization': fleet_utilization,
                'total_vcpu': total_vcpu,
                'total_memory': total_memory,
                'concurrency': concurrency,
                'target_rps': target_rps,
                'achieved_rps': achieved_rps,
                'efficiency_pct': efficiency_pct,
                'duration_seconds': duration_seconds,
                'latency_us': latency_us,
                'latency_seconds': latency_seconds,
                'instance_cost_per_1k': instance_cost,
                'instance_cost_per_pod_per_1k': instance_cost_per_pod,
                'request_cost_per_1k': request_cost,
                'cost_difference': cost_difference,
                'percent_savings': percent_savings,
                'cheaper_model': 'Instance' if instance_cost < request_cost else 'Request'
            })
        
        print(f"Processed: {len(results)} configurations")
        return results
    
    def process_all_files(self, capacity_threshold=95.0):
        """Process all statistics files."""
        stats_files = self.find_statistics_files()
        if not stats_files:
            return False
        
        print(f"\n{'='*80}")
        print(f"MULTI-POD COST ANALYSIS: A/B/C SERIES COMPARISON")
        print(f"Multiple RPS levels: 100, 200, 300, 400, 500 RPS")
        print(f"{'='*80}")
        
        self.all_results = []
        
        for stats_file in stats_files:
            results = self.calculate_costs_for_file(stats_file, capacity_threshold)
            self.all_results.extend(results)
        
        print(f"\nTotal configurations processed: {len(self.all_results)}")
        return len(self.all_results) > 0
    
    def generate_comprehensive_report(self):
        """Generate analysis across all series."""
        if not self.all_results:
            return None
        
        df = pd.DataFrame(self.all_results)
        
        print(f"\n{'='*80}")
        print(f"MULTI-POD CLOUD RUN COST ANALYSIS")
        print(f"{'='*80}")
        
        # Series descriptions
        print(f"\nSERIES SCALING STRATEGIES:")
        print(f"  A-Series: Single pod scale-up (varying resources per pod, containerConcurrency=1)")
        print(f"  B-Series: Multi-pod scale-out (constant 1.0 vCPU total, varying containerConcurrency)")
        print(f"  C-Series: Multi-pod scale-out (constant 2.0 vCPU total, varying containerConcurrency)")
        
        # Analyze each series separately
        for series in ['A', 'B', 'C']:
            series_df = df[df['series'] == series]
            if series_df.empty:
                continue
                
            print(f"\n{series}-SERIES ANALYSIS:")
            
            if series == 'A':
                print(f"  Strategy: Single pod, varying resources")
                for config in ['A1', 'A2', 'A3', 'A4', 'A5']:
                    if config in series_df['config'].values:
                        config_details = self.series_configs[config]
                        print(f"  {config}: {config_details['vcpu']}vCPU × {config_details['memory_gib']}GB × {config_details['pods']} pod")
            
            elif series == 'B':
                print(f"  Strategy: Constant total resources (1.0 vCPU total), increasing pods")
                for config in ['B1', 'B2', 'B3', 'B4', 'B5']:
                    if config in series_df['config'].values:
                        config_details = self.series_configs[config]
                        total_vcpu = config_details['vcpu'] * config_details['pods']
                        print(f"  {config}: {config_details['vcpu']}vCPU × {config_details['pods']} pods = {total_vcpu} total vCPU (Conc={config_details['concurrency']})")
            
            elif series == 'C':
                print(f"  Strategy: Constant total resources (2.0 vCPU total), increasing pods")
                for config in ['C1', 'C2', 'C3', 'C4', 'C5']:
                    if config in series_df['config'].values:
                        config_details = self.series_configs[config]
                        total_vcpu = config_details['vcpu'] * config_details['pods']
                        print(f"  {config}: {config_details['vcpu']}vCPU × {config_details['pods']} pods = {total_vcpu} total vCPU (Conc={config_details['concurrency']})")
            
            # Cost analysis for this series
            avg_instance = series_df['instance_cost_per_1k'].mean()
            avg_request = series_df['request_cost_per_1k'].mean()
            instance_wins = len(series_df[series_df['cheaper_model'] == 'Instance'])
            total_configs = len(series_df)
            
            print(f"  Average instance cost: ${avg_instance:.6f}")
            print(f"  Average request cost: ${avg_request:.6f}")
            print(f"  Instance cheaper: {instance_wins}/{total_configs} ({instance_wins/total_configs*100:.1f}%)")
        
        # Cross-series comparison
        print(f"\nCROSS-SERIES COST COMPARISON:")
        series_summary = df.groupby('series').agg({
            'instance_cost_per_1k': ['mean', 'min', 'max'],
            'request_cost_per_1k': ['mean', 'min', 'max'],
            'num_pods': ['mean', 'min', 'max'],
            'total_vcpu': ['mean', 'min', 'max']
        }).round(6)
        
        print(series_summary.to_string())
        
        # Busy pods and utilization analysis at different RPS levels
        print(f"\nBUSY PODS & FLEET UTILIZATION BY RPS LEVEL:")
        for rps_level in [100, 200, 300, 400, 500]:
            rps_df = df[df['target_rps'] == rps_level]
            if not rps_df.empty:
                print(f"  {rps_level} RPS:")
                for series in ['A', 'B', 'C']:
                    series_rps_df = rps_df[rps_df['series'] == series]
                    if not series_rps_df.empty:
                        avg_busy = series_rps_df['busy_pods'].mean()
                        max_busy = series_rps_df['busy_pods'].max()
                        min_busy = series_rps_df['busy_pods'].min()
                        avg_util = series_rps_df['fleet_utilization'].mean()
                        max_util = series_rps_df['fleet_utilization'].max()
                        min_util = series_rps_df['fleet_utilization'].min()
                        print(f"    {series}-Series: {min_busy:.1f}-{max_busy:.1f} busy pods (avg: {avg_busy:.1f}), utilization: {min_util:.1%}-{max_util:.1%} (avg: {avg_util:.1%})")
        
        # Multi-pod penalty analysis
        print(f"\nMULTI-POD PENALTY ANALYSIS:")
        for series in ['B', 'C']:
            series_df = df[df['series'] == series]
            if len(series_df) > 1:
                # Analyze penalty at each RPS level
                for rps_level in [100, 200, 300]:
                    rps_series_df = series_df[series_df['target_rps'] == rps_level]
                    if len(rps_series_df) > 1:
                        single_pod = rps_series_df[rps_series_df['num_pods'] == 1]
                        multi_pod = rps_series_df[rps_series_df['num_pods'] > 1]
                        
                        if not single_pod.empty and not multi_pod.empty:
                            single_cost = single_pod['instance_cost_per_1k'].mean()
                            multi_cost = multi_pod['instance_cost_per_1k'].mean()
                            penalty = (multi_cost / single_cost - 1) * 100
                            
                            print(f"  {series}-Series at {rps_level} RPS: Multi-pod adds {penalty:.1f}% cost penalty")
        
        # Break-even analysis
        print(f"\nBREAK-EVEN ANALYSIS (Request = Instance cost):")
        print(f"Break-even latency per configuration (when request-based becomes competitive):")
        
        configs_analyzed = set()
        for _, row in df.iterrows():
            config_key = (row['series'], row['config'], row['num_pods'], row['vcpu_per_pod'], row['memory_per_pod'])
            if config_key not in configs_analyzed:
                configs_analyzed.add(config_key)
                
                # Calculate break-even latency: solve instance_cost = request_cost for latency
                # instance_cost = duration_seconds * (total_vcpu * inst_cpu + total_mem * inst_mem)
                # request_cost = 1000 * latency_seconds * (vcpu_per_pod * req_cpu + mem_per_pod * req_mem) + fee
                # At 100 RPS: duration_seconds = 10, so:
                # 10 * (total_resources * inst_rates) = 1000 * latency * (per_pod_resources * req_rates) + fee
                
                total_vcpu = row['vcpu_per_pod'] * row['num_pods']
                total_memory = row['memory_per_pod'] * row['num_pods']
                
                # Instance cost per 1k at 100 RPS (10 second duration)
                instance_cost_base = 10 * (
                    total_vcpu * self.INSTANCE_CPU_PRICE_PER_VCPU_SEC +
                    total_memory * self.INSTANCE_MEM_PRICE_PER_GIB_SEC
                )
                
                # Solve for latency: instance_cost_base = 1000 * latency * (per_pod_rates) + fee
                per_pod_rate = (
                    row['vcpu_per_pod'] * self.REQUEST_CPU_PRICE_PER_VCPU_SEC +
                    row['memory_per_pod'] * self.REQUEST_MEM_PRICE_PER_GIB_SEC
                )
                
                if per_pod_rate > 0:
                    breakeven_latency_seconds = (instance_cost_base - self.REQUEST_PER_REQUEST_FEE) / (1000 * per_pod_rate)
                    breakeven_latency_ms = breakeven_latency_seconds * 1000
                    
                    print(f"  {row['config']} ({row['num_pods']} pods, {row['vcpu_per_pod']}vCPU): {breakeven_latency_ms:.1f}ms")
        
        print(f"\nRUNTIME PERFORMANCE:")
        for runtime in sorted(df['runtime'].unique()):
            runtime_df = df[df['runtime'] == runtime]
            if not runtime_df.empty:
                avg_instance = runtime_df['instance_cost_per_1k'].mean()
                avg_request = runtime_df['request_cost_per_1k'].mean()
                print(f"  {runtime.capitalize()}: Instance avg=${avg_instance:.6f}, Request avg=${avg_request:.6f}")
        
        return df
    
    def create_series_comparison_chart(self, df):
        """Create comprehensive chart comparing A, B, C series."""
        # Set up consistent styling
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Chart 1: Total Cost by Series and Config
        ax1 = axes[0, 0]
        series_order = ['A', 'B', 'C']
        colors = {'A': '#2196F3', 'B': '#4CAF50', 'C': '#FF9800'}  # Blue, Green, Orange
        
        for series in series_order:
            series_df = df[df['series'] == series]
            if not series_df.empty:
                # Group by config and calculate means
                series_summary = series_df.groupby('config').agg({
                    'instance_cost_per_1k': 'mean',
                    'request_cost_per_1k': 'mean'
                }).reset_index()
                
                # Extract numeric part of config for plotting
                configs = [int(c[1:]) for c in series_summary['config']]
                x_pos = np.array(configs)
                
                ax1.plot(x_pos, series_summary['instance_cost_per_1k'], 
                        marker='o', label=f'{series}-Series Instance', color=colors[series], linestyle='-', linewidth=2)
                ax1.plot(x_pos, series_summary['request_cost_per_1k'],
                        marker='s', label=f'{series}-Series Request', color=colors[series], linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Configuration Level (1-5)', fontweight='bold')
        ax1.set_ylabel('Cost per 1,000 requests ($)', fontweight='bold')
        ax1.set_title('Cost Comparison Across Series', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Chart 2: Pod Count vs Cost
        ax2 = axes[0, 1]
        for series in series_order:
            series_df = df[df['series'] == series]
            if not series_df.empty:
                ax2.scatter(series_df['num_pods'], series_df['instance_cost_per_1k'],
                           label=f'{series}-Series', color=colors[series], s=60, alpha=0.7)
        
        ax2.set_xlabel('Number of Pods', fontweight='bold')
        ax2.set_ylabel('Instance Cost per 1,000 requests ($)', fontweight='bold')
        ax2.set_title('Pod Count vs Cost', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Chart 3: Total Resources vs Cost Efficiency
        ax3 = axes[1, 0]
        for series in series_order:
            series_df = df[df['series'] == series]
            if not series_df.empty:
                cost_ratio = series_df['request_cost_per_1k'] / series_df['instance_cost_per_1k']
                ax3.scatter(series_df['total_vcpu'], cost_ratio,
                           label=f'{series}-Series', color=colors[series], s=60, alpha=0.7)
        
        ax3.set_xlabel('Total vCPU Allocation', fontweight='bold')
        ax3.set_ylabel('Request/Instance Cost Ratio', fontweight='bold')
        ax3.set_title('Resource Allocation vs Cost Efficiency', fontweight='bold')
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Break-even')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Runtime Comparison
        ax4 = axes[1, 1]
        runtime_summary = df.groupby(['runtime', 'series']).agg({
            'instance_cost_per_1k': 'mean'
        }).reset_index()
        
        runtime_colors = {'python': '#FFC107', 'nodejs': '#4CAF50', 'go': '#2196F3'}  # Consistent with other scripts
        
        for runtime in runtime_summary['runtime'].unique():
            runtime_df = runtime_summary[runtime_summary['runtime'] == runtime]
            series_pos = [series_order.index(s) for s in runtime_df['series'] if s in series_order]
            runtime_costs = runtime_df[runtime_df['series'].isin(series_order)]['instance_cost_per_1k'].values
            
            if len(series_pos) > 0:
                ax4.plot(series_pos, runtime_costs,
                        marker='o', label=runtime.capitalize(), color=runtime_colors.get(runtime, 'gray'), linewidth=2)
        
        ax4.set_xlabel('Series', fontweight='bold')
        ax4.set_ylabel('Average Instance Cost per 1,000 requests ($)', fontweight='bold')
        ax4.set_title('Runtime Performance Across Series', fontweight='bold')
        ax4.set_xticks(range(len(series_order)))
        ax4.set_xticklabels(series_order)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "multi_pod_series_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Series comparison chart saved to: {output_path}")
        plt.close()
    
    def save_results(self, df):
        """Save all analysis results."""
        # Main results
        main_output = os.path.join(self.output_dir, "multi_pod_cost_analysis.csv")
        df.to_csv(main_output, index=False)
        print(f"Detailed results saved to: {main_output}")
        
        # Series summary
        series_summary = df.groupby(['series', 'config']).agg({
            'runtime': 'count',
            'num_pods': 'first',
            'total_vcpu': 'first',
            'total_memory': 'first',
            'instance_cost_per_1k': ['mean', 'std'],
            'request_cost_per_1k': ['mean', 'std'],
            'percent_savings': 'mean'
        }).round(6)
        
        series_output = os.path.join(self.output_dir, "series_summary.csv")
        series_summary.to_csv(series_output)
        print(f"Series summary saved to: {series_output}")
        
        # Multi-pod penalty analysis
        penalty_analysis = []
        for series in ['A', 'B', 'C']:
            series_df = df[df['series'] == series]
            if not series_df.empty:
                for config in series_df['config'].unique():
                    config_df = series_df[series_df['config'] == config]
                    if not config_df.empty:
                        config_details = self.series_configs[config]
                        penalty_analysis.append({
                            'series': series,
                            'config': config,
                            'num_pods': config_details['pods'],
                            'vcpu_per_pod': config_details['vcpu'],
                            'total_vcpu': config_details['vcpu'] * config_details['pods'],
                            'avg_instance_cost': config_df['instance_cost_per_1k'].mean(),
                            'avg_request_cost': config_df['request_cost_per_1k'].mean(),
                            'multi_pod_penalty_pct': 0  # Will calculate this separately
                        })
        
        penalty_df = pd.DataFrame(penalty_analysis)
        penalty_output = os.path.join(self.output_dir, "multi_pod_penalty_analysis.csv")
        penalty_df.to_csv(penalty_output, index=False)
        print(f"Multi-pod penalty analysis saved to: {penalty_output}")
    
    def run_analysis(self, capacity_threshold=95.0):
        """Run complete multi-pod analysis."""
        if not self.process_all_files(capacity_threshold):
            return False
        
        df = self.generate_comprehensive_report()
        if df is not None:
            self.save_results(df)
            self.create_series_comparison_chart(df)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Multi-pod Cloud Run cost analysis for A, B, C series')
    parser.add_argument('stats_pattern', help='Glob pattern for statistics CSV files')
    parser.add_argument('--output-dir', '-o', default='multi_pod_analysis', help='Output directory')
    parser.add_argument('--threshold', '-t', type=float, default=95.0, help='Minimum RPS threshold')
    
    args = parser.parse_args()
    
    try:
        calculator = MultiPodCostCalculator(args.stats_pattern, output_dir=args.output_dir)
        
        success = calculator.run_analysis(capacity_threshold=args.threshold)
        
        if success:
            print(f"\n✓ Multi-pod analysis completed!")
            print(f"✓ Results saved in: {args.output_dir}/")
        else:
            print("✗ Analysis failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
