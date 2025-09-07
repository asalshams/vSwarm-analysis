#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import json
import time
import re
from pathlib import Path
import pandas as pd
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# COMPREHENSIVE TEST RUNNER FOR VSWARM WITH POD MONITORING
# =============================================================================

class ComprehensiveTestRunner:
    # Default test configuration
    DEFAULT_RPS_VALUES = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
    DEFAULT_TEST_DURATION = 60
    DEFAULT_ITERATIONS_PER_RPS = 3
    
    # Adaptive testing configuration
    ADAPTIVE_START_RPS = 20
    ADAPTIVE_INCREMENT = 20
    ADAPTIVE_MAX_RPS = 2000
    
    # Stopping conditions (simplified - focus on throughput)
    MIN_THROUGHPUT_RATIO = 0.85  # Stop if actual throughput < 85% of target (primary condition)
    MIN_CPU_UTILIZATION = 0.80   # Only stop if CPU utilization > 80% (prevents premature stopping)
    MAX_ERROR_RATE = 0.30        # Stop if error rate > 30% (infrastructure bottleneck)
    CONSECUTIVE_FAILURES = 1     # Stop after 1 failed RPS level (immediate stop)
    
    def __init__(self, title, invoker_dir="../tools/invoker", service_name=None, namespace="default", skip_visualizations=False, adaptive_mode=False):
        self.title = title
        self.adaptive_mode = adaptive_mode
        
        if adaptive_mode:
            # Use adaptive RPS values (will be generated dynamically)
            self.rps_values = []
            self.adaptive_current_rps = self.ADAPTIVE_START_RPS
            self.consecutive_failures = 0
            self.test_results = []  # Store results for adaptive decisions
        else:
            # Use fixed RPS values
            self.rps_values = self.DEFAULT_RPS_VALUES
            
        self.test_duration = self.DEFAULT_TEST_DURATION
        self.iterations_per_rps = self.DEFAULT_ITERATIONS_PER_RPS
        self.durations = [self.test_duration] * len(self.rps_values) if not adaptive_mode else []
        self.invoker_dir = invoker_dir
        self.results_dir = f"results_{title}"
        self.stats_file = os.path.join(self.results_dir, f"{title}_statistics.csv")
        self.service_name = service_name
        self.namespace = namespace
        self.monitoring_running = False
        self.monitoring_file = None
        self.skip_visualizations = skip_visualizations
        
        # Create a test-specific directory inside invoker dir for raw files
        self.raw_files_dir = os.path.join(self.invoker_dir, f"raw_files_{title}")
        os.makedirs(self.raw_files_dir, exist_ok=True)
        
        print(f"=== Comprehensive Test Runner ===")
        print(f"Test Title: {title}")
        
        if adaptive_mode:
            print(f"Mode: ADAPTIVE (stops when throughput drops)")
            print(f"Starting RPS: {self.ADAPTIVE_START_RPS}")
            print(f"RPS Increment: {self.ADAPTIVE_INCREMENT}")
            print(f"Maximum RPS: {self.ADAPTIVE_MAX_RPS}")
            print(f"Stopping Conditions:")
            print(f"  - Throughput < {self.MIN_THROUGHPUT_RATIO*100:.1f}% of target RPS AND")
            print(f"  - CPU utilization > {self.MIN_CPU_UTILIZATION*100:.1f}%")
            print(f"  - OR Error rate > {self.MAX_ERROR_RATE*100:.1f}% (infrastructure bottleneck)")
        else:
            print(f"Mode: FIXED RPS")
            print(f"RPS Values: {self.rps_values}")
            print(f"Total Tests: {len(self.rps_values) * self.iterations_per_rps}")
            
        print(f"Duration: {self.test_duration} seconds (per RPS level)")
        print(f"Iterations per RPS: {self.iterations_per_rps}")
        print(f"Results Directory: {self.results_dir}")
        print(f"Statistics File: {self.stats_file}")
        print(f"Raw Files Directory: {self.raw_files_dir}")
        if service_name:
            print(f"Pod Monitoring: {service_name} (namespace: {namespace})")
        if self.skip_visualizations:
            print(f"Visualizations: DISABLED (--skip-visualizations flag used)")
        else:
            print(f"Visualizations: ENABLED")
        print("=" * 50)
    
    def check_prerequisites(self):
        """Check if all required tools are available."""
        print("Checking prerequisites...")
        
        # Check if invoker exists
        invoker_path = os.path.join(self.invoker_dir, "invoker")
        if not os.path.exists(invoker_path):
            print(f" Invoker not found at {invoker_path}")
            print("Building invoker...")
            self.build_invoker()
        else:
            print(" Invoker found")
        
        # Check if endpoints.json exists
        endpoints_path = os.path.join(self.invoker_dir, "endpoints.json")
        if not os.path.exists(endpoints_path):
            print(f" endpoints.json not found at {endpoints_path}")
            return False
        else:
            print(" endpoints.json found")
        
        # Update endpoints.json with current service if service_name is provided
        if not self.update_endpoints_json():
            print(" Failed to update endpoints.json")
            return False
        
        # Check if matplotlib and seaborn are available (only if visualizations are enabled)
        if not self.skip_visualizations:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                print(" matplotlib and seaborn found")
            except ImportError as e:
                print(f" Required visualization libraries not found: {e}")
                print("Install with: pip install matplotlib seaborn")
                return False
        else:
            print(" matplotlib and seaborn check skipped (visualizations disabled)")
        
        return True
    
    def build_invoker(self):
        """Build the invoker binary."""
        try:
            subprocess.run(["make", "invoker"], cwd=self.invoker_dir, check=True)
            print(" Invoker built successfully")
        except subprocess.CalledProcessError as e:
            print(f" Failed to build invoker: {e}")
            sys.exit(1)
    
    def detect_constrained_namespace(self, service_name):
        """Detect required constrained namespace from service name."""
        import re
        
        # Extract configuration (a1, b2, c4, etc.) from service name
        # Matches patterns like: fibonacci-nodejs-a1, fibonacci-python-b3, fibonacci-go-c5
        match = re.search(r'fibonacci-[a-z]+-([abc][1-5])', service_name)
        if match:
            config = match.group(1)
            return f"constrained-tests-{config}"
        return None
    
    def verify_constrained_namespace(self, constrained_namespace):
        """Verify that the constrained namespace exists and has proper resource policies."""
        print(f" Verifying constrained namespace: {constrained_namespace}")
        
        try:
            # Check if namespace exists
            result = subprocess.run([
                "kubectl", "get", "namespace", constrained_namespace
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f" ERROR: Constrained namespace {constrained_namespace} not found!")
                print(f" This service requires a constrained namespace for proper resource enforcement.")
                print(f" Please deploy the namespace configuration first:")
                print(f" kubectl apply -f ../benchmarks/fibonacci/yamls/knative/namespace-{constrained_namespace}.yaml")
                return False
            
            # Verify LimitRange exists
            result = subprocess.run([
                "kubectl", "get", "limitrange", "-n", constrained_namespace, "--no-headers"
            ], capture_output=True, text=True)
            
            if result.returncode != 0 or not result.stdout.strip():
                print(f" ERROR: No LimitRange found in {constrained_namespace}!")
                print(f" The constrained namespace exists but lacks resource enforcement policies.")
                print(f" Please redeploy the namespace configuration:")
                print(f" kubectl apply -f ../benchmarks/fibonacci/yamls/knative/namespace-{constrained_namespace}.yaml")
                return False
            
            # Verify ResourceQuota exists
            result = subprocess.run([
                "kubectl", "get", "resourcequota", "-n", constrained_namespace, "--no-headers"
            ], capture_output=True, text=True)
            
            if result.returncode != 0 or not result.stdout.strip():
                print(f" ERROR: No ResourceQuota found in {constrained_namespace}!")
                print(f" The constrained namespace exists but lacks resource quota policies.")
                print(f" Please redeploy the namespace configuration:")
                print(f" kubectl apply -f ../benchmarks/fibonacci/yamls/knative/namespace-{constrained_namespace}.yaml")
                return False
            
            print(f" ✓ Constrained namespace validation passed")
            print(f" ✓ LimitRange and ResourceQuota policies are active")
            return True
            
        except Exception as e:
            print(f" Failed to verify constrained namespace: {e}")
            return False

    def update_endpoints_json(self):
        """Update endpoints.json with the current service URL if service_name is provided."""
        if not self.service_name:
            print(" No service name provided, skipping endpoints.json update")
            return True
        
        # Detect and verify constrained namespace
        constrained_namespace = self.detect_constrained_namespace(self.service_name)
        if constrained_namespace:
            print(f" Detected constrained namespace: {constrained_namespace}")
            if not self.verify_constrained_namespace(constrained_namespace):
                return False
            # Update namespace to use the constrained one
            self.namespace = constrained_namespace
            print(f" → Using constrained namespace: {self.namespace}")
        else:
            print(f" No constrained namespace detected for service: {self.service_name}")
            print(f" → Using namespace: {self.namespace}")
        
        print(f" Updating endpoints.json for service: {self.service_name}")
        try:
            # Get the service URL from kubectl
            cmd = [
                "kubectl", "get", "ksvc", self.service_name, 
                "-n", self.namespace, 
                "-o", "jsonpath={.status.url}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            service_url = result.stdout.strip()
            
            if not service_url:
                print(f" Failed to get service URL for {self.service_name}")
                return False
            
            # Extract hostname from URL (remove http:// prefix)
            hostname = service_url.replace("http://", "")
            print(f" Service URL: {service_url}")
            print(f" Hostname: {hostname}")
            
            # Create endpoints.json content
            endpoints_data = [{"hostname": hostname}]
            endpoints_json = json.dumps(endpoints_data, indent=2)
            
            # Write to endpoints.json
            endpoints_path = os.path.join(self.invoker_dir, "endpoints.json")
            with open(endpoints_path, 'w') as f:
                f.write(endpoints_json)
            
            print(f" ✓ endpoints.json updated with: {hostname}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f" Failed to get service URL: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f" Failed to update endpoints.json: {e}")
            return False
    
    def check_resource_quota_status(self):
        """Check if resource quota is near exhaustion."""
        if not self.service_name:
            return False, "No service specified"
            
        # Detect constrained namespace
        constrained_namespace = self.detect_constrained_namespace(self.service_name)
        if not constrained_namespace:
            return False, "No constrained namespace detected"
            
        try:
            # Get resource quota status
            result = subprocess.run([
                "kubectl", "get", "resourcequota", "-n", constrained_namespace,
                "-o", "jsonpath={.items[0].status}"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, f"Failed to get resource quota: {result.stderr}"
                
            import json
            quota_status = json.loads(result.stdout)
            
            # Check CPU usage
            if 'used' in quota_status and 'hard' in quota_status:
                used_cpu = quota_status['used'].get('requests.cpu', '0m')
                hard_cpu = quota_status['hard'].get('requests.cpu', '0m')
                
                # Convert to millicores for comparison
                used_millicores = self._parse_cpu_to_millicores(used_cpu)
                hard_millicores = self._parse_cpu_to_millicores(hard_cpu)
                
                if hard_millicores > 0:
                    usage_ratio = used_millicores / hard_millicores
                    if usage_ratio > 0.95:  # 95% usage
                        return True, f"Resource quota near exhaustion: {usage_ratio*100:.1f}% CPU used"
                        
            return False, "Resource quota OK"
            
        except Exception as e:
            return False, f"Error checking resource quota: {e}"
    
    def _parse_cpu_to_millicores(self, cpu_str):
        """Convert CPU string to millicores."""
        if cpu_str.endswith('m'):
            return int(cpu_str[:-1])
        else:
            return int(float(cpu_str) * 1000)
    
    def get_current_cpu_utilization(self):
        """Get current CPU utilization from pod monitoring data.
        
        Returns:
            tuple: (current_cpu_millicores, cpu_limit_millicores, utilization_ratio)
                   Returns (0, 0, 0) if data unavailable
        """
        if not hasattr(self, 'monitoring_file') or not os.path.exists(self.monitoring_file):
            return 0, 0, 0
            
        try:
            # Read the last few lines of monitoring data
            with open(self.monitoring_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:  # Need at least header + 1 data line
                return 0, 0, 0
            
            # Get the most recent CPU usage (last 3 lines to get average)
            recent_lines = lines[-3:] if len(lines) >= 4 else lines[-2:]  # Skip header
            cpu_values = []
            
            for line in recent_lines:
                if line.strip() and not line.startswith('timestamp'):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        try:
                            cpu_millicores = int(parts[3])  # cpu_usage_millicores column
                            cpu_values.append(cpu_millicores)
                        except ValueError:
                            continue
            
            if not cpu_values:
                return 0, 0, 0
                
            current_cpu = sum(cpu_values) / len(cpu_values)  # Average recent usage
            
            # Get CPU limits from service configuration
            cpu_limit_millicores = self.get_service_cpu_limits()
            
            utilization_ratio = current_cpu / cpu_limit_millicores if cpu_limit_millicores > 0 else 0
            
            return current_cpu, cpu_limit_millicores, utilization_ratio
            
        except Exception as e:
            print(f" Warning: Could not get CPU utilization: {e}")
            return 0, 0, 0
    
    def get_service_cpu_limits(self):
        """Get CPU limits for the current service from Kubernetes.
        
        Returns:
            int: Total CPU limits in millicores for all containers in the pod
        """
        if not self.service_name or not self.namespace:
            return 0
            
        try:
            # Get the service configuration
            cmd = [
                "kubectl", "get", "ksvc", self.service_name,
                "-n", self.namespace,
                "-o", "json"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            service_data = json.loads(result.stdout)
            
            # Extract container specs from the service
            containers = service_data.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
            
            total_cpu_limit = 0
            for container in containers:
                resources = container.get('resources', {})
                limits = resources.get('limits', {})
                cpu_limit = limits.get('cpu', '0')
                
                # Convert to millicores and add to total
                total_cpu_limit += self._parse_cpu_to_millicores(cpu_limit)
            
            return total_cpu_limit
            
        except Exception as e:
            print(f" Warning: Could not get service CPU limits: {e}")
            return 0
    
    def get_error_rate_from_stats(self, stats):
        """Calculate error rate from statistics.
        
        Args:
            stats: Dictionary containing test statistics
            
        Returns:
            float: Error rate (0.0 to 1.0), or 0.0 if cannot determine
        """
        try:
            total_requests = stats.get('total_requests', 0)
            if total_requests == 0:
                return 0.0
                
            # Look for error indicators in the stats
            # This is a heuristic based on typical invoker behavior
            actual_rps = stats.get('throughput_rps', 0)
            # Estimate expected requests based on test duration
            expected_requests = actual_rps * self.test_duration if actual_rps > 0 else total_requests
            
            # If we got significantly fewer requests than expected, likely errors
            if expected_requests > 0:
                completion_rate = total_requests / expected_requests
                error_rate = max(0.0, 1.0 - completion_rate)
                return min(1.0, error_rate)
            
            return 0.0
            
        except Exception as e:
            print(f" Warning: Could not calculate error rate: {e}")
            return 0.0
    
    def detect_infrastructure_bottleneck(self, throughput_ratio, cpu_utilization, error_rate):
        """Detect if we're hitting infrastructure bottlenecks vs computational limits.
        
        Returns:
            tuple: (is_infrastructure_bottleneck, reason)
        """
        # High error rate indicates infrastructure issues
        if error_rate > self.MAX_ERROR_RATE:  # 30% error rate
            return True, f"High error rate ({error_rate*100:.1f}%) - likely queue-proxy or network bottleneck"
            
        # Low throughput + low CPU + moderate errors = infrastructure bottleneck
        if throughput_ratio < 0.85 and cpu_utilization < 0.5 and error_rate > 0.1:
            return True, f"Low throughput ({throughput_ratio*100:.1f}%) + low CPU ({cpu_utilization*100:.1f}%) + errors ({error_rate*100:.1f}%) - infrastructure bottleneck"
            
        # Very low throughput with very low CPU = clear infrastructure issue
        if throughput_ratio < 0.7 and cpu_utilization < 0.3:
            return True, f"Very low throughput ({throughput_ratio*100:.1f}%) with minimal CPU usage ({cpu_utilization*100:.1f}%) - infrastructure bottleneck"
            
        return False, ""
    
    def analyze_test_results(self, latency_file, target_rps):
        """Analyze test results to determine if we should continue or stop."""
        if not os.path.exists(latency_file):
            return False, "Test failed - no latency file generated"
            
        # Load and analyze latency data
        latency_data = self.load_latency_data(latency_file)
        if latency_data is None or len(latency_data) == 0:
            return False, "Test failed - no latency data"
            
        # Calculate statistics
        stats = self.calculate_statistics(latency_data, self.test_duration, os.path.basename(latency_file))
        if not stats:
            return False, "Test failed - could not calculate statistics"
            
        # Store results for trend analysis
        self.test_results.append({
            'target_rps': target_rps,
            'actual_rps': stats.get('throughput_rps', 0),
            'p99_latency_ms': stats.get('p99', 0) / 1000,  # Convert μs to ms
            'total_requests': stats.get('total_requests', 0),
            'stats': stats
        })
        
        # Primary stopping condition: throughput ratio
        actual_rps = stats.get('throughput_rps', 0)
        throughput_ratio = actual_rps / target_rps if target_rps > 0 else 0
        
        # Check CPU utilization to avoid premature stopping
        current_cpu, cpu_limit, cpu_utilization = self.get_current_cpu_utilization()
        
        # Check for infrastructure bottleneck (high error rate)
        error_rate = self.get_error_rate_from_stats(stats)
        
        # Check for infrastructure bottlenecks
        is_infra_bottleneck, bottleneck_reason = self.detect_infrastructure_bottleneck(
            throughput_ratio, cpu_utilization, error_rate)
        
        if throughput_ratio < self.MIN_THROUGHPUT_RATIO:
            # Check if this is a true computational limit or infrastructure bottleneck
            if cpu_utilization > self.MIN_CPU_UTILIZATION:  # True CPU bottleneck
                return False, f"CAPACITY REACHED: {actual_rps:.1f} RPS achieved ({throughput_ratio*100:.1f}% of target {target_rps}), CPU: {cpu_utilization*100:.1f}%"
            elif is_infra_bottleneck:  # Infrastructure bottleneck - stop gracefully
                return False, f"INFRASTRUCTURE LIMIT: {bottleneck_reason} - stopping to avoid error flood"
            elif throughput_ratio < 0.6:  # Very low throughput - likely resource constraint
                p99_latency_ms = stats.get('p99', 0) / 1000  # Convert μs to ms
                return False, f"RESOURCE CONSTRAINT: Only {actual_rps:.1f} RPS achieved ({throughput_ratio*100:.1f}% of target {target_rps}), P99 = {p99_latency_ms:.1f}ms - likely resource limits"
            else:
                # Moderate throughput drop but unclear why - continue with warning for one more level
                p99_latency_ms = stats.get('p99', 0) / 1000  # Convert μs to ms
                print(f" ⚠️  Low throughput ({throughput_ratio*100:.1f}%) but CPU utilization is only {cpu_utilization*100:.1f}% - continuing...")
                return True, f"⚠️  RPS {target_rps}: {actual_rps:.1f} actual RPS ({throughput_ratio*100:.1f}%), CPU: {cpu_utilization*100:.1f}%, P99 = {p99_latency_ms:.1f}ms"
            
        # Test passed - system can handle this RPS level
        p99_latency_ms = stats.get('p99', 0) / 1000  # Convert μs to ms
        cpu_info = f", CPU: {cpu_utilization*100:.1f}%" if cpu_utilization > 0 else ""
        return True, f"✓ RPS {target_rps}: {actual_rps:.1f} actual RPS ({throughput_ratio*100:.1f}%){cpu_info}, P99 = {p99_latency_ms:.1f}ms"
    
    def get_next_adaptive_rps(self):
        """Get the next RPS value for adaptive testing."""
        if self.adaptive_current_rps >= self.ADAPTIVE_MAX_RPS:
            return None  # Reached maximum
            
        current_rps = self.adaptive_current_rps
        self.adaptive_current_rps += self.ADAPTIVE_INCREMENT
        return current_rps
    
    def should_stop_adaptive_testing(self):
        """Check if adaptive testing should stop based on throughput failure."""
        return self.consecutive_failures >= self.CONSECUTIVE_FAILURES
    
    def parse_latency_filename(self, filename):
        """Parse latency filename and extract RPS values and iteration info.
        
        Returns:
            dict: Contains target_rps, actual_rps, iteration (if available), or None if parsing fails
        """
        # Try new format with iteration: target_rps{R}_iter{II}_rps{A}_lat.csv
        match = re.search(r'target_rps(\d+\.?\d*)_iter(\d+)_rps(\d+\.?\d*)_lat\.csv', filename)
        if match:
            return {
                'target_rps': float(match.group(1)),
                'iteration': int(match.group(2)),
                'actual_rps': float(match.group(3)),
                'has_iteration': True
            }
        
        # Try old format: target_rps{R}_actual_rps{A}_lat.csv
        match = re.search(r'target_rps(\d+\.?\d*)_actual_rps(\d+\.?\d*)_lat\.csv', filename)
        if match:
            return {
                'target_rps': float(match.group(1)),
                'actual_rps': float(match.group(2)),
                'iteration': 1,
                'has_iteration': False
            }
        
        return None

    def get_latency_files(self):
        """Get all latency files in the invoker directory."""
        files = glob.glob(os.path.join(self.invoker_dir, "target_rps*_actual_rps*_lat.csv"))
        rps_files = {}
        for f in files:
            filename = os.path.basename(f)
            parsed = self.parse_latency_filename(filename)
            if parsed:
                target_rps = parsed['target_rps']
                if target_rps not in rps_files:
                    rps_files[target_rps] = []
                rps_files[target_rps].append({
                    'path': f,
                    'filename': filename,
                    'mtime': os.path.getmtime(f),
                    'parsed': parsed
                })
        return rps_files

    def run_invoker_test(self, rps, duration, iteration=1):
        """Run a single invoker test."""
        print(f"\n--- Running Test: RPS={rps}, Duration={duration}s, Iteration={iteration}/{self.iterations_per_rps} ---")
        
        # Build the invoker command
        cmd = [
            "./invoker",
            "-time", str(duration),
            "-rps", str(rps),
            "-port", "50000"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Run the invoker from the invoker directory
        try:
            print(" Running invoker (output will be shown below):")
            result = subprocess.run(cmd, cwd=self.invoker_dir, check=True)
            print(" Test completed successfully")
            
            # Find the newly generated latency file
            latency_files = glob.glob(os.path.join(self.invoker_dir, "rps*_lat.csv"))
            if latency_files:
                # Get the most recent file
                latest_file = max(latency_files, key=os.path.getctime)
                
                # Create a new filename with target RPS and iteration at the beginning
                original_name = os.path.basename(latest_file)
                new_name = f"target_rps{rps:.2f}_iter{iteration:02d}_{original_name}"
                new_path = os.path.join(self.results_dir, new_name)
                
                # Copy file to results directory
                os.makedirs(self.results_dir, exist_ok=True)
                try:
                    subprocess.run(["cp", latest_file, new_path], check=True)
                    print(f" Copied to: {new_name}")
                    
                    # Clean up original file
                    os.remove(latest_file)
                    print(f" Cleaned up: {original_name}")
                except (subprocess.CalledProcessError, OSError) as e:
                    print(f" Failed to handle file {original_name}: {e}")
            else:
                print("  No latency file was generated")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f" Test failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def manage_raw_files(self):
        """Move raw latency files to test directory."""
        print("\n--- Managing Raw Latency Files ---")
        
        # Move any existing latency files to the test directory
        existing_files = glob.glob(os.path.join(self.invoker_dir, "target_rps*_actual_rps*_lat.csv"))
        if existing_files:
            for file in existing_files:
                filename = os.path.basename(file)
                dest = os.path.join(self.raw_files_dir, filename)
                try:
                    shutil.move(file, dest)
                    print(f" Moved to test directory: {filename}")
                except Exception as e:
                    print(f" Failed to move {filename}: {e}")
            print(f" Total files moved: {len(existing_files)}")
        else:
            print("No latency files found to move.")

    def find_latency_files_for_rps(self, rps):
        """Find all latency files for a given RPS value (all iterations)."""
        # Look for files with exact matching target RPS
        pattern = f"target_rps{rps:.2f}_iter*_rps*_lat.csv"
        files = glob.glob(os.path.join(self.results_dir, pattern))
        
        if not files:
            print(f" No latency files found for target RPS {rps}")
            return []
        
        # Sort files by iteration number
        files.sort(key=lambda x: os.path.basename(x))
        print(f" Found {len(files)} latency files for RPS {rps}")
        for f in files:
            print(f"   - {os.path.basename(f)}")
        return files
    
    def copy_latency_files(self):
        """Copy latency files to results directory."""
        print(f"\n--- Finding Latency Files in {self.results_dir} ---")
        
        all_files = []
        rps_files_map = {}
        
        # Find all latency files in results directory
        for rps in self.rps_values:
            pattern = f"target_rps{rps:.2f}_iter*_rps*_lat.csv"
            files = glob.glob(os.path.join(self.results_dir, pattern))
            
            if files:
                # Sort files by iteration number
                files.sort(key=lambda x: os.path.basename(x))
                all_files.extend(files)
                rps_files_map[rps] = files
                print(f" Found {len(files)} files for RPS {rps}")
                for f in files:
                    print(f"   - {os.path.basename(f)}")
            else:
                print(f" No latency files found for target RPS {rps}")
        
        if not all_files:
            print(" No latency files found")
        else:
            print(f" Total files found: {len(all_files)} out of {len(self.rps_values) * self.iterations_per_rps} expected files")
        
        return all_files, rps_files_map
    
    def map_rps_to_latency_files(self, latency_files):
        """Map RPS values to their corresponding latency files."""
        rps_to_file = {}
        
        for rps in self.rps_values:
            # Find the latency file that corresponds to this RPS
            for latency_file in latency_files:
                filename = os.path.basename(latency_file)
                try:
                    match = re.search(r'target_rps(\d+\.?\d*)_actual_rps(\d+\.?\d*)_lat\.csv', filename)
                    if match:
                        file_target_rps = float(match.group(1))
                        file_actual_rps = float(match.group(2))
                        # Check if this file is close to our target RPS
                        if abs(file_target_rps - rps) <= 5 or abs(file_actual_rps - rps) <= 5:  # Accept files within 5 RPS difference
                            rps_to_file[rps] = latency_file
                            break
                except (ValueError, AttributeError):
                    continue
        
        return rps_to_file
    
    def generate_statistics(self, latency_files, rps_files_map):
        """Generate comprehensive statistics with averaging across iterations."""
        print(f"\n--- Generating Statistics (Averaging {self.iterations_per_rps} iterations per RPS) ---")
        
        all_stats = []
        all_data = {}  # Store all latency data for visualization
        
        # Process each RPS level and average across iterations
        for rps in self.rps_values:
            if rps not in rps_files_map:
                print(f" No files found for RPS {rps}")
                continue
                
            files_for_rps = rps_files_map[rps]
            print(f"\n--- Analyzing RPS {rps} ({len(files_for_rps)} iterations) ---")
            
            # Collect statistics for all iterations
            iteration_stats = []
            
            for i, latency_file in enumerate(files_for_rps, 1):
                print(f"  Iteration {i}/{len(files_for_rps)}: {os.path.basename(latency_file)}")
                
                # Load and analyze latency data using integrated function
                latency_data = self.load_latency_data(latency_file)
                if latency_data is not None:
                    filename = os.path.basename(latency_file)
                    stats = self.calculate_statistics(latency_data, self.test_duration, filename)
                    if stats:
                        iteration_stats.append(stats)
                        # Store data for visualization
                        all_data[filename] = latency_data
                        print(f"     Statistics generated")
                    else:
                        print(f"     Failed to calculate statistics for {latency_file}")
                else:
                    print(f"     Failed to load data from {latency_file}")
            
            # Average statistics across iterations
            if iteration_stats:
                averaged_stats = self.average_statistics(iteration_stats, rps)
                all_stats.append(averaged_stats)
                print(f" Averaged statistics generated for RPS {rps}")
            else:
                print(f" No valid statistics for RPS {rps}")
        
        # Save statistics to CSV
        if all_stats:
            self.save_statistics_csv(all_stats)
        
        # Generate detailed analysis CSV
        print(f"\n--- Generating Detailed Analysis CSV ---")
        analysis_csv_path = os.path.join(self.results_dir, f"{self.title}_detailed_analysis.csv")
        
        detailed_stats = []
        for filename, data in all_data.items():
            if data is not None:
                stats = self.calculate_statistics(data, self.test_duration, filename)
                if stats:
                    stats['filename'] = filename
                    detailed_stats.append(stats)
        
        if detailed_stats:
            df = pd.DataFrame(detailed_stats)
            df.to_csv(analysis_csv_path, index=False)
            print(f" Detailed analysis CSV saved to: {analysis_csv_path}")
        
        # Store all_data for visualization
        self.all_data = all_data
        self.all_stats_dict = {stats.get('rps', 0): stats for stats in all_stats}
        
        return all_stats
    
    def parse_latency_with_units(self, latency_str):
        """Parse latency string with units and convert to milliseconds."""
        latency_str = latency_str.strip()
        
        # Handle different time units and convert to milliseconds
        if 'μs' in latency_str:
            # Microseconds to milliseconds
            value = float(latency_str.replace('μs', ''))
            return value / 1000.0
        elif 'ms' in latency_str:
            # Already in milliseconds
            return float(latency_str.replace('ms', ''))
        elif 's' in latency_str and 'ms' not in latency_str:
            # Seconds to milliseconds (but not catching 'ms' again)
            value = float(latency_str.replace('s', ''))
            return value * 1000.0
        else:
            # Assume milliseconds if no unit specified
            try:
                return float(latency_str)
            except ValueError:
                print(f"  Warning: Could not parse latency value: {latency_str}")
                return 0.0

    def parse_analysis_output(self, output, rps, duration):
        """Parse the analysis script output to extract statistics."""
        lines = output.strip().split('\n')
        
        # Find the statistics section
        stats = {
            'rps': rps,
            'duration': duration,
            'test_title': self.title
        }
        
        for line in lines:
            line = line.strip()
            if 'Data points:' in line:
                stats['total_requests'] = int(line.split(':')[1].replace(',', ''))
            elif 'Minimum latency:' in line:
                latency_str = line.split(':')[1].strip()
                stats['min_latency_us'] = self.parse_latency_with_units(latency_str)
            elif 'Maximum latency:' in line:
                latency_str = line.split(':')[1].strip()
                stats['max_latency_us'] = self.parse_latency_with_units(latency_str)
            elif 'Average latency:' in line:
                latency_str = line.split(':')[1].strip()
                stats['avg_latency_us'] = self.parse_latency_with_units(latency_str)
            elif 'Median latency:' in line:
                latency_str = line.split(':')[1].strip()
                stats['median_latency_us'] = self.parse_latency_with_units(latency_str)
            elif '95th percentile:' in line:
                latency_str = line.split(':')[1].strip()
                stats['p95_latency_us'] = self.parse_latency_with_units(latency_str)
            elif '99th percentile:' in line:
                latency_str = line.split(':')[1].strip()
                stats['p99_latency_us'] = self.parse_latency_with_units(latency_str)
            elif 'Throughput:' in line:
                throughput_str = line.split(':')[1].strip()
                if 'RPS' in throughput_str:
                    stats['throughput_rps'] = float(throughput_str.replace(' RPS', ''))
                else:
                    stats['throughput_rps'] = 0.0
        
        return stats
    
    def average_statistics(self, iteration_stats, rps):
        """Average statistics across multiple iterations."""
        if not iteration_stats:
            return None
        
        # Initialize averaged stats with the first iteration's structure
        averaged_stats = {
            'rps': rps,
            'duration': self.test_duration,
            'test_title': self.title,
            'iterations': len(iteration_stats)
        }
        
        # Calculate averages for numeric fields (using integrated stats keys)
        numeric_fields = [
            'total_requests', 'min', 'max', 'average',
            'median', 'p95', 'p99', 'throughput_rps'
        ]
        
        for field in numeric_fields:
            values = [stats.get(field, 0) for stats in iteration_stats if field in stats]
            if values:
                averaged_stats[f'avg_{field}'] = sum(values) / len(values)
                averaged_stats[f'std_{field}'] = pd.Series(values).std()
                averaged_stats[f'min_{field}'] = min(values)
                averaged_stats[f'max_{field}'] = max(values)
        
        # For backward compatibility, also include the main fields
        for field in numeric_fields:
            if f'avg_{field}' in averaged_stats:
                averaged_stats[field] = averaged_stats[f'avg_{field}']
        
        return averaged_stats
    
    def save_statistics_csv(self, all_stats):
        """Save statistics to CSV file."""
        df = pd.DataFrame(all_stats)
        df.to_csv(self.stats_file, index=False)
        print(f" Statistics saved to: {self.stats_file}")
        
        # Print summary
        print("\n=== Statistics Summary (Averaged Across Iterations) ===")
        print("Note: Only showing statistics for RPS values where latency files were found")
        print("-" * 120)
        
        # Create a more readable summary
        summary_df = df.copy()
        # Round numeric columns to 2 decimal places
        numeric_cols = summary_df.select_dtypes(include=['float64']).columns
        summary_df[numeric_cols] = summary_df[numeric_cols].round(2)
        
        # Reorder columns for better readability
        desired_order = ['rps', 'iterations', 'throughput_rps', 'std_throughput_rps', 
                        'average', 'p95', 'p99', 'total_requests']
        other_cols = [col for col in summary_df.columns if col not in desired_order]
        final_order = desired_order + other_cols
        summary_df = summary_df[final_order]
        
        print(summary_df.to_string(index=False))
        print("-" * 120)
        
        # Print analysis of missing RPS values
        missing_rps = set(self.rps_values) - set(df['rps'].values)
        if missing_rps:
            print("\nMissing RPS values (no statistics generated):")
            for rps in sorted(missing_rps):
                print(f"- RPS {rps}")
        
        # Print consistency analysis
        print("\n=== Consistency Analysis ===")
        high_variance_rps = []
        for _, row in df.iterrows():
            if 'std_throughput_rps' in row and row['std_throughput_rps'] > 5.0:
                high_variance_rps.append((row['rps'], row['std_throughput_rps']))
        
        if high_variance_rps:
            print("  High variance detected (>5 RPS std dev):")
            for rps, std in high_variance_rps:
                print(f"   - RPS {rps}: ±{std:.1f} RPS")
        else:
            print(" All RPS levels show consistent results")
        print()
    
    def create_visualizations(self, latency_files):
        """Create comprehensive visualizations using integrated functions."""
        print(f"\n--- Creating Visualizations ---")
        
        charts_dir = os.path.join(self.results_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Use the stored data from generate_statistics
        if not hasattr(self, 'all_data') or not hasattr(self, 'all_stats_dict'):
            print(" No data available for visualization")
            return False
        
        all_data = self.all_data
        all_stats = self.all_stats_dict
        
        # Create comprehensive time series chart
        if self.service_name:
            monitoring_file = os.path.join(self.results_dir, f"{self.title}_pod_monitoring.csv")
            if os.path.exists(monitoring_file):
                resource_data = self.load_resource_data(monitoring_file)
                if resource_data is not None:
                    self.create_comprehensive_time_series(all_data, all_stats, resource_data)
        
        # Create latency vs throughput chart
        self.create_latency_vs_throughput_chart(all_stats)
        
        # Create percentile charts
        self.create_percentile_charts(all_stats)
        
        print(f" Visualizations created in: {charts_dir}")
        return True
    
    def create_resource_evolution_charts(self, latency_files):
        """Create the new resource evolution charts that show changes throughout all requests."""
        if not self.service_name:
            return False
            
        monitoring_file = os.path.join(self.results_dir, f"{self.title}_pod_monitoring.csv")
        if not os.path.exists(monitoring_file):
            print("  No pod monitoring data found for resource evolution charts")
            return False
            
        print(f"\n--- Creating Resource Evolution Charts ---")
        
        try:
            # Import visualization functions
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from visualize_latency_enhanced import (
                load_latency_data, 
                load_resource_data, 
                calculate_statistics,
                create_resource_evolution_chart,
                create_resource_vs_latency_correlation
            )
            
            # Load the comprehensive resource data
            full_resource_data = load_resource_data(monitoring_file)
            if full_resource_data is None:
                print(" Failed to load resource monitoring data")
                return False
            
            # Load all latency data and calculate statistics
            all_stats = {}
            all_resource_data = {}
            
            for latency_file in latency_files:
                base_name = os.path.splitext(os.path.basename(latency_file))[0]
                
                # Load latency data
                latency_data = load_latency_data(latency_file)
                if latency_data is not None:
                    filename = os.path.basename(latency_file)
                    all_stats[base_name] = calculate_statistics(latency_data, filename=filename)
                    # Use the full resource dataset for each phase
                    all_resource_data[base_name] = full_resource_data
            
            if not all_stats:
                print(" No valid latency data found for resource evolution")
                return False
            
            # Create charts directory
            charts_dir = os.path.join(self.results_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # Generate resource evolution charts
            evolution_path = os.path.join(charts_dir, "resource_evolution_comprehensive.png")
            create_resource_evolution_chart(all_resource_data, all_stats, evolution_path)
            
            correlation_path = os.path.join(charts_dir, "resource_vs_latency_correlation.png")
            create_resource_vs_latency_correlation(all_resource_data, all_stats, correlation_path)
            
            # Show summary
            rps_values = [stats.get('target_rps', 0) for stats in all_stats.values() if stats]
            rps_values.sort()
            
            print(f" Resource evolution charts created:")
            print(f"    Resource Evolution: resource_evolution_comprehensive.png")
            print(f"    Resource vs Latency Correlation: resource_vs_latency_correlation.png")
            print(f"    Covering {len(all_stats)} test phases ({min(rps_values):.1f} - {max(rps_values):.1f} RPS)")
            print(f"    {len(full_resource_data)} resource monitoring points")
            
            return True
            
        except Exception as e:
            print(f" Failed to create resource evolution charts: {e}")
            return False


    def start_monitoring(self):
        """Start pod monitoring in background."""
        if not self.service_name:
            return False
        
        print(f"\n--- Starting Pod Monitoring ---")
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.monitoring_file = os.path.join(self.results_dir, f"{self.title}_pod_monitoring.csv")
        self.monitoring_running = True
        
        # Start monitoring in a separate thread
        import threading
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        print(f" Pod monitoring started")
        print(f" Monitoring data will be saved to: {self.monitoring_file}")
        time.sleep(2)  # Give monitoring time to start
        
        return True
    
    def _monitor_resources(self):
        """Internal method to monitor Kubernetes pod resources."""
        import time
        from datetime import datetime
        
        # Create monitoring file with headers
        with open(self.monitoring_file, 'w') as f:
            f.write("timestamp,pod_count,ready_pods,cpu_usage_millicores,memory_usage_mib,pod_details\n")
        
        # Determine current revision for more precise pod monitoring
        self.label_selector = None
        self.current_revision = None
        
        # First get the current revision
        try:
            result = subprocess.run([
                'kubectl', 'get', 'ksvc', self.service_name, '-n', self.namespace,
                '-o', 'jsonpath={.status.latestReadyRevisionName}'
            ], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                self.current_revision = result.stdout.strip()
        except Exception:
            pass
        
        # Build selector to target only current revision pods
        if self.current_revision:
            selector_candidates = [
                f"serving.knative.dev/revision={self.current_revision}",
                f"serving.knative.dev/service={self.service_name},serving.knative.dev/revision={self.current_revision}",
            ]
        else:
            # Fallback to service-level selectors
            selector_candidates = [
                f"serving.knative.dev/service={self.service_name}",
                f"app={self.service_name}",
                f"serving.knative.dev/configuration={self.service_name}"
            ]
            
        for sel in selector_candidates:
            probe = subprocess.run([
                'kubectl', 'get', 'pods', '-n', self.namespace,
                '-l', sel, '--no-headers'
            ], capture_output=True, text=True)
            if probe.returncode == 0 and probe.stdout.strip():
                self.label_selector = sel
                break
        if not self.label_selector:
            # Fall back to the most probable Knative label
            self.label_selector = f"serving.knative.dev/service={self.service_name}"
        
        while self.monitoring_running:
            try:
                # Get current timestamp
                timestamp = datetime.now().isoformat()
                
                # Get detailed pod information
                result = subprocess.run([
                    'kubectl', 'get', 'pods', '-n', self.namespace,
                    '-l', self.label_selector, '--no-headers',
                    '-o', 'custom-columns=NAME:.metadata.name,READY:.status.containerStatuses[*].ready,STATUS:.status.phase'
                ], capture_output=True, text=True)
                
                pod_count = 0
                ready_pods = 0
                pod_details = ""
                
                if result.returncode == 0:
                    pod_lines = [ln for ln in result.stdout.strip().split('\n') if ln.strip()]
                    pod_count = len(pod_lines)
                    
                    # Count ready pods and create details string
                    pod_statuses = []
                    for line in pod_lines:
                        parts = line.split()
                        if len(parts) >= 3:
                            name = parts[0]
                            ready_status = parts[1]
                            phase = parts[2]
                            
                            # Check if pod is ready (all containers ready)
                            if 'true' in ready_status.lower() and phase == 'Running':
                                ready_pods += 1
                                pod_statuses.append(f"{name}:Ready")
                            else:
                                pod_statuses.append(f"{name}:{phase}")
                    
                    pod_details = "|".join(pod_statuses)
                
                # Get CPU and memory usage
                cpu_millicores = 0
                memory_mib = 0
                
                if pod_count > 0:
                    # Get resource usage for pods (requires metrics-server)
                    result = subprocess.run([
                        'kubectl', 'top', 'pods', '-n', self.namespace,
                        '-l', self.label_selector
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        # Some kubectl versions include a header line; detect and skip if present
                        if lines and ('CPU' in lines[0] or 'MEMORY' in lines[0] or 'NAME' in lines[0]):
                            lines = lines[1:]
                        total_cpu_millicores = 0
                        total_memory_mib = 0
                        
                        for line in lines:
                            if line.strip():
                                parts = line.split()
                                if len(parts) >= 3:
                                    # Parse CPU (e.g., "100m" -> 100 millicores)
                                    cpu_str = parts[1]
                                    if cpu_str.endswith('m'):
                                        cpu_val = int(cpu_str[:-1])
                                    else:
                                        # e.g., "0" or cores; convert to millicores
                                        cpu_val = int(float(cpu_str) * 1000)
                                    
                                    # Parse memory (e.g., "50Mi" -> 50 MiB, "1Gi" -> 1024 MiB)
                                    memory_str = parts[2]
                                    if memory_str.endswith('Mi'):
                                        mem_val = int(memory_str[:-2])
                                    elif memory_str.endswith('Gi'):
                                        mem_val = int(float(memory_str[:-2]) * 1024)
                                    else:
                                        # Fallback: try to parse as MiB-equivalent
                                        mem_val = int(float(memory_str))
                                    
                                    total_cpu_millicores += cpu_val
                                    total_memory_mib += mem_val
                        
                        cpu_millicores = total_cpu_millicores
                        memory_mib = total_memory_mib
                
                # Write data to file
                with open(self.monitoring_file, 'a') as f:
                    f.write(f"{timestamp},{pod_count},{ready_pods},{cpu_millicores},{memory_mib},\"{pod_details}\"\n")
                
                # Wait for next interval
                time.sleep(2)
                
            except Exception as e:
                print(f"  Monitoring error: {e}")
                time.sleep(2)

    def stop_monitoring(self):
        """Stop pod monitoring process."""
        if hasattr(self, 'monitoring_running') and self.monitoring_running:
            print(f"\n--- Stopping Pod Monitoring ---")
            try:
                # Stop the monitoring thread
                self.monitoring_running = False
                print(" Waiting for monitoring to save final data...")
                time.sleep(3)  # Give thread time to finish
                
                print(" Pod monitoring stopped")
                
                # Verify monitoring file was created
                if hasattr(self, 'monitoring_file') and os.path.exists(self.monitoring_file):
                    print(f" Monitoring data saved: {self.monitoring_file}")
                else:
                    print(f"  Warning: Monitoring file not found")
                    
            except Exception as e:
                print(f" Error stopping monitoring: {e}")
            finally:
                self.monitoring_running = False

    def convert_monitoring_data(self, monitoring_file):
        """Convert monitoring data to format expected by visualizer."""
        try:
            converted_file = monitoring_file.replace('.csv', '_converted.csv')
            
            # Read our monitoring data
            df = pd.read_csv(monitoring_file)
            
            # Create new dataframe with expected format
            converted_df = pd.DataFrame()
            converted_df['timestamp'] = df['timestamp']
            converted_df['pod_count'] = df['pod_count']
            
            # Convert CPU from millicores to approximate percentage
            # Assuming 1000 millicores = 100% of 1 CPU core
            converted_df['cpu_percent'] = (df['cpu_usage_millicores'] / 1000) * 100
            
            # Convert memory from MiB to approximate percentage 
            # Assuming 1024 MiB = 100% (rough approximation)
            converted_df['memory_percent'] = (df['memory_usage_mib'] / 1024) * 100
            
            # Save converted data
            converted_df.to_csv(converted_file, index=False)
            print(f" Converted monitoring data for visualizer: {converted_file}")
            
            return converted_file
            
        except Exception as e:
            print(f" Failed to convert monitoring data: {e}")
            return None

    def _run_test_execution_phase(self):
        """Execute all invoker tests with iterations."""
        if self.adaptive_mode:
            return self._run_adaptive_test_execution()
        else:
            return self._run_fixed_test_execution()
    
    def _run_fixed_test_execution(self):
        """Execute fixed RPS tests with iterations."""
        print(f"\n--- Running {len(self.rps_values) * self.iterations_per_rps} Fixed RPS Tests ---")
        
        for rps, duration in zip(self.rps_values, self.durations):
            print(f"\n{'='*50}")
            print(f"Testing RPS {rps} ({self.iterations_per_rps} iterations)")
            print(f"{'='*50}")
            
            for iteration in range(1, self.iterations_per_rps + 1):
                if not self.run_invoker_test(rps, duration, iteration):
                    print(f" Test failed for RPS {rps}, iteration {iteration}")
                    return False
                time.sleep(2)  # Small delay between iterations
            
            print(f" Completed all {self.iterations_per_rps} iterations for RPS {rps}")
            time.sleep(5)  # Longer delay between RPS levels
        
        return True
    
    def _run_adaptive_test_execution(self):
        """Execute adaptive RPS tests that stop when system reaches limits."""
        print(f"\n--- Running Adaptive RPS Tests (stops at system limits) ---")
        
        test_count = 0
        while True:
            # Get next RPS value
            current_rps = self.get_next_adaptive_rps()
            if current_rps is None:
                print(f"\n Reached maximum RPS limit ({self.ADAPTIVE_MAX_RPS})")
                break
                
            print(f"\n{'='*60}")
            print(f"Testing RPS {current_rps} (adaptive test #{test_count + 1})")
            print(f"{'='*60}")
            
            # Run iterations for this RPS level
            rps_success = True
            iteration_results = []
            
            for iteration in range(1, self.iterations_per_rps + 1):
                print(f"\n--- Iteration {iteration}/{self.iterations_per_rps} ---")
                
                if not self.run_invoker_test(current_rps, self.test_duration, iteration):
                    print(f" ✗ Test execution failed for RPS {current_rps}, iteration {iteration}")
                    rps_success = False
                    break
                    
                # Find and analyze the latest result
                latency_files = glob.glob(os.path.join(self.results_dir, f"target_rps{current_rps:.2f}_iter{iteration:02d}_*.csv"))
                if latency_files:
                    latest_file = max(latency_files, key=os.path.getctime)
                    success, message = self.analyze_test_results(latest_file, current_rps)
                    iteration_results.append(success)
                    print(f" → {message}")
                    
                    if not success:
                        print(f" ✗ {message}")
                        rps_success = False
                        break
                else:
                    print(f" ✗ No latency file found for iteration {iteration}")
                    rps_success = False
                    break
                    
                time.sleep(2)  # Small delay between iterations
            
            # Analyze RPS level results
            if rps_success and all(iteration_results):
                print(f" ✓ RPS {current_rps} completed successfully - system can handle this load")
                self.consecutive_failures = 0
                self.rps_values.append(current_rps)  # Add to tested RPS values
            else:
                self.consecutive_failures += 1
                print(f"\n🛑 STOPPING: System capacity reached at RPS {current_rps}")
                print(f"   Last successful RPS: {max(self.rps_values) if self.rps_values else 'None'}")
                break
            
            test_count += 1
            print(f" Completed adaptive test #{test_count}")
            time.sleep(5)  # Longer delay between RPS levels
        
        print(f"\n=== Adaptive Testing Complete ===")
        print(f"Total RPS levels tested: {len(self.rps_values)}")
        if self.rps_values:
            print(f"RPS range: {min(self.rps_values)} - {max(self.rps_values)}")
            print(f"Maximum sustainable RPS: {max(self.rps_values)}")
        
        return True

    def _run_analysis_phase(self):
        """Run analysis and visualization phase."""
        # Copy latency files
        latency_files, rps_files_map = self.copy_latency_files()
        if not latency_files:
            print(" No latency files found")
            return False, None
        
        # Generate statistics
        stats = self.generate_statistics(latency_files, rps_files_map)
        if not stats:
            print(" Failed to generate statistics")
            return False, None
        
        # Create visualizations (unless skipped)
        if latency_files and not self.skip_visualizations:
            if not self.create_visualizations(latency_files):
                print(" Failed to create visualizations")
                # Don't return False here, as the statistics are still useful
            
            # Create resource evolution charts
            if self.service_name:
                self.create_resource_evolution_charts(latency_files)
        elif self.skip_visualizations:
            print("  Skipping visualization generation (--skip-visualizations flag used)")
        else:
            print("  No latency files available for visualizations")
        
        return True, latency_files

    def _print_pipeline_summary(self, latency_files):
        """Print summary of pipeline execution."""
        print(f"\n Pipeline completed!")
        print(f" Statistics: {self.stats_file}")
        print(f" Results: {self.results_dir}")
        
        if latency_files and not self.skip_visualizations:
            print(f" Charts: {self.results_dir}/charts")
            if self.service_name:
                print(f" Resource Evolution Charts: {self.results_dir}/charts/resource_evolution_comprehensive.png")
                print(f" Resource vs Latency Correlation: {self.results_dir}/charts/resource_vs_latency_correlation.png")
        elif self.skip_visualizations:
            print(f"  Charts: Skipped (--skip-visualizations flag used)")
        else:
            print(f"  No charts generated (no latency files available)")
        
        if self.service_name:
            monitoring_file = os.path.join(self.results_dir, f"{self.title}_pod_monitoring.csv")
            print(f" Pod Monitoring: {monitoring_file}")

    def run_complete_pipeline(self, skip_tests=False):
        """Run the complete test pipeline with optional pod monitoring."""
        print("Starting comprehensive test pipeline...")
        
        # Step 1: Move any existing files to test directory
        self.manage_raw_files()
        
        # Step 2: Check prerequisites
        if not self.check_prerequisites():
            print(" Prerequisites check failed")
            return False
        
        # Step 3: Start pod monitoring if service specified
        monitoring_started = False
        if self.service_name and not skip_tests:
            monitoring_started = self.start_monitoring()
        
        try:
            # Step 4: Execute tests or skip
            if not skip_tests:
                if not self._run_test_execution_phase():
                    return False
            else:
                print("\n--- Skipping invoker tests, using existing files ---")
        finally:
            # Always stop monitoring if it was started
            if monitoring_started:
                self.stop_monitoring()
        
        # Step 5-7: Analysis and visualization
        success, latency_files = self._run_analysis_phase()
        if not success:
            return False
        
        # Step 8: Print summary
        self._print_pipeline_summary(latency_files)
        
        return True

    # =============================================================================
    # INTEGRATED ANALYSIS AND VISUALIZATION FUNCTIONS
    # =============================================================================
    
    def load_latency_data(self, file_path):
        """Load latency data from CSV file."""
        try:
            data = pd.read_csv(file_path, header=None, names=['latency'])
            return data['latency'].values
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def load_resource_data(self, file_path):
        """Load resource usage data from CSV file."""
        try:
            data = pd.read_csv(file_path)
            if 'timestamp' in data.columns:
                try:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                except:
                    pass
            return data
        except Exception as e:
            print(f"Error reading resource data {file_path}: {e}")
            return None
    
    def calculate_statistics(self, latency_data, test_duration_seconds=None, filename=None):
        """Calculate statistical measures for latency data.
        
        Note: latency_data is expected to be in microseconds (μs) from the invoker.
        All returned latency values are in microseconds.
        """
        if latency_data is None or len(latency_data) == 0:
            return None
        
        # All calculations done in microseconds (μs)
        stats = {
            'min': np.min(latency_data),      # μs
            'max': np.max(latency_data),      # μs
            'average': np.mean(latency_data), # μs
            'median': np.median(latency_data), # μs
            'p95': np.percentile(latency_data, 95), # μs
            'p99': np.percentile(latency_data, 99), # μs
            'std': np.std(latency_data)       # μs
        }
        
        total_requests = len(latency_data)
        
        if filename:
            import re
            # Try the new filename format first: target_rps400.00_iter01_rps191.27_lat.csv
            match = re.search(r'target_rps(\d+\.?\d*)_iter\d+_rps(\d+\.?\d*)_lat\.csv', filename)
            if match:
                stats['target_rps'] = float(match.group(1))
                stats['throughput_rps'] = float(match.group(2))
            else:
                # Fallback to old format: target_rps400.00_actual_rps191.27_lat.csv
                match = re.search(r'target_rps(\d+\.?\d*)_actual_rps(\d+\.?\d*)_lat\.csv', filename)
                if match:
                    stats['target_rps'] = float(match.group(1))
                    stats['throughput_rps'] = float(match.group(2))
                else:
                    if test_duration_seconds:
                        stats['throughput_rps'] = total_requests / test_duration_seconds
                    else:
                        avg_latency_seconds = stats['average'] / 1000
                        if avg_latency_seconds > 0:
                            stats['throughput_rps'] = 1 / avg_latency_seconds
                        else:
                            stats['throughput_rps'] = 0
                    stats['target_rps'] = stats['throughput_rps']
        else:
            if test_duration_seconds:
                stats['throughput_rps'] = total_requests / test_duration_seconds
            else:
                avg_latency_seconds = stats['average'] / 1000
                if avg_latency_seconds > 0:
                    stats['throughput_rps'] = 1 / avg_latency_seconds
                else:
                    stats['throughput_rps'] = 0
            stats['target_rps'] = stats['throughput_rps']
        
        stats['total_requests'] = total_requests
        return stats
    
    def create_comprehensive_time_series(self, all_data, all_stats, resource_data=None):
        """Create comprehensive time series chart for the whole test."""
        if resource_data is None or resource_data.empty:
            print("  No resource data available for comprehensive time series")
            return False
        
        print(f"\n--- Creating Comprehensive Time Series Chart ---")
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            # Build ordered list of (target_rps, iteration, data)
            ordered_entries = []
            for filename, data in all_data.items():
                if data is None:
                    continue
                try:
                    # Expect: target_rps{R}_iter{II}_rps{A}_lat.csv
                    m = re.search(r'target_rps(\d+\.?\d*)_iter(\d+)_rps(\d+\.?\d*)_lat\.csv', filename)
                    if m:
                        target_rps = float(m.group(1))
                        iteration = int(m.group(2))
                        ordered_entries.append((target_rps, iteration, data, filename))
                    else:
                        # Fallback: put at end if pattern doesn't match
                        ordered_entries.append((float('inf'), 999, data, filename))
                except Exception:
                    ordered_entries.append((float('inf'), 999, data, filename))
            
            # Sort by target RPS then iteration
            ordered_entries.sort(key=lambda x: (x[0], x[1]))
            
            # Concatenate all latency datasets for the full test series
            concatenated = np.concatenate([e[2] for e in ordered_entries]) if ordered_entries else None
            total_requests = int(len(concatenated)) if concatenated is not None else 0
            
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
                ax1.set_title(f'Comprehensive Test Timeline - {self.title}')
                ax1.grid(True, alpha=0.3)
                
                # Moving average over a window proportional to phase size
                window_size = max(20, min(1000, len(latency_ms) // 200))
                if window_size > 1:
                    moving_avg = pd.Series(latency_ms).rolling(window=window_size).mean()
                    ax1.plot(range(len(moving_avg)), moving_avg, 'r-', linewidth=2, 
                             label=f'Moving Average ({window_size} requests)')
                    ax1.legend()
            
            plt.tight_layout()
            
            # Save the chart
            charts_dir = os.path.join(self.results_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            output_path = os.path.join(charts_dir, "comprehensive_timeseries_with_pods.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f" Comprehensive time series chart saved: {output_path}")
            return True
            
        except Exception as e:
            print(f" Failed to create comprehensive time series chart: {e}")
            return False
    
    def create_latency_vs_throughput_chart(self, all_stats):
        """Create latency vs throughput scatter plot."""
        if not all_stats:
            return False
        
        print(f"\n--- Creating Latency vs Throughput Chart ---")
        
        try:
            throughputs = []
            avg_latencies = []
            names = []
            
            for name, stats in all_stats.items():
                if stats and 'throughput_rps' in stats:
                    throughputs.append(stats['throughput_rps'])
                    avg_latencies.append(stats['average'] / 1000)  # Convert μs to ms
                    names.append(name)
            
            if throughputs:
                plt.figure(figsize=(10, 8))
                plt.scatter(throughputs, avg_latencies, s=100, alpha=0.7)
                
                # Add labels
                for i, name in enumerate(names):
                    plt.annotate(name, (throughputs[i], avg_latencies[i]), 
                                xytext=(5, 5), textcoords='offset points')
                
                plt.xlabel('Throughput (RPS)')
                plt.ylabel('Average Latency (ms)')
                plt.title(f'Throughput vs Average Latency - {self.title}')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                charts_dir = os.path.join(self.results_dir, "charts")
                os.makedirs(charts_dir, exist_ok=True)
                output_path = os.path.join(charts_dir, "latency_vs_throughput.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f" Latency vs throughput chart saved: {output_path}")
                return True
                
        except Exception as e:
            print(f" Failed to create latency vs throughput chart: {e}")
            return False
    
    def create_percentile_charts(self, all_stats):
        """Create P95, P99, and Average latency vs RPS charts."""
        if not all_stats:
            return False
        
        print(f"\n--- Creating Percentile Charts ---")
        
        try:
            charts_dir = os.path.join(self.results_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # Extract data
            rps_values = []
            p95_values = []
            p99_values = []
            avg_values = []
            names = []
            
            for name, stats in all_stats.items():
                if stats and 'throughput_rps' in stats:
                    rps_values.append(stats['throughput_rps'])
                    p95_values.append(stats['p95'] / 1000)  # Convert μs to ms
                    p99_values.append(stats['p99'] / 1000)  # Convert μs to ms
                    avg_values.append(stats['average'] / 1000)  # Convert μs to ms
                    names.append(name)
            
            if rps_values:
                # Sort by RPS for proper line plot
                sorted_data = sorted(zip(rps_values, p95_values, p99_values, avg_values, names))
                rps_sorted, p95_sorted, p99_sorted, avg_sorted, names_sorted = zip(*sorted_data)
                
                # P95 chart
                plt.figure(figsize=(12, 8))
                plt.plot(rps_sorted, p95_sorted, 'bo-', linewidth=3, markersize=8, alpha=0.8)
                plt.xlabel('Throughput (RPS)')
                plt.ylabel('P95 Latency (ms)')
                plt.title(f'P95 Latency vs Throughput - {self.title}')
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(os.path.join(charts_dir, "p95_vs_rps.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # P99 chart
                plt.figure(figsize=(12, 8))
                plt.plot(rps_sorted, p99_sorted, 'go-', linewidth=3, markersize=8, alpha=0.8)
                plt.xlabel('Throughput (RPS)')
                plt.ylabel('P99 Latency (ms)')
                plt.title(f'P99 Latency vs Throughput - {self.title}')
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(os.path.join(charts_dir, "p99_vs_rps.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Average chart
                plt.figure(figsize=(12, 8))
                plt.plot(rps_sorted, avg_sorted, 'mo-', linewidth=3, markersize=8, alpha=0.8)
                plt.xlabel('Throughput (RPS)')
                plt.ylabel('Average Latency (ms)')
                plt.title(f'Average Latency vs Throughput - {self.title}')
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(os.path.join(charts_dir, "avg_vs_rps.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f" Percentile charts saved in: {charts_dir}")
                return True
                
        except Exception as e:
            print(f" Failed to create percentile charts: {e}")
            return False
    



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Comprehensive Test Runner for vSwarm with Pod Monitoring')
    parser.add_argument('--title', '-t', required=True, help='Test title (used for file naming)')
    parser.add_argument('--invoker-dir', default='../tools/invoker', help='Directory containing invoker binary')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running invoker tests and just do analysis')
    parser.add_argument('--service', '-s', help='Kubernetes service name to monitor (enables pod monitoring)')
    parser.add_argument('--namespace', '-n', default='default', help='Kubernetes namespace (default: default)')
    parser.add_argument('--skip-visualizations', action='store_true', help='Skip generating visualization charts')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive RPS testing (stops when system reaches limits)')
    
    args = parser.parse_args()
    return args.title, args.invoker_dir, args.skip_tests, args.service, args.namespace, args.skip_visualizations, args.adaptive


def main():
    """Main function."""
    try:
        title, invoker_dir, skip_tests, service_name, namespace, skip_visualizations, adaptive_mode = parse_arguments()
        
        # Create and run the test runner
        runner = ComprehensiveTestRunner(title, invoker_dir, service_name, namespace, skip_visualizations, adaptive_mode)
        success = runner.run_complete_pipeline(skip_tests)
        
        if success:
            print("\n All tests completed successfully!")
            sys.exit(0)
        else:
            print("\n Pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
