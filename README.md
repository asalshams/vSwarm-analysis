# vSwarm Analysis - Serverless Performance Testing Suite 

A comprehensive performance analysis framework for serverless functions built on top of the vSwarm benchmarking suite. This repository contains tools for running controlled performance tests, analysing latency patterns, monitoring resource usage, and generating detailed cost analysis for Google Cloud Run deployments.

## Project Structure

```
vswarm-analysis/
├── benchmarks/                    # Working copy of vSwarm benchmarks
│   ├── fibonacci/                 # Fibonacci benchmark implementations
│   │   ├── go/                   # Go implementation
│   │   ├── nodejs/               # Node.js implementation  
│   │   ├── python/               # Python implementation
│   │   └── yamls/                # Kubernetes/Knative configurations
│   └── [other benchmarks]/       # Additional vSwarm benchmarks
├── external/vswarm/              # Upstream vSwarm repository (git submodule)
├── tools/
│   └── invoker/                  # gRPC client for load testing
│       ├── invoker               # Compiled binary
│       └── endpoints.json        # Service endpoints configuration
└── latency_analysis/             # Main analysis and testing framework
    ├── run_test.py              # Comprehensive test runner
    ├── fair_run.sh              # Standardised fair testing script
    ├── individual_test_visualizations.py
    ├── a_series_visualizations.py
    ├── b_series_visualizations.py
    ├── c_series_visualizations.py
    ├── gcr_cost_calculator_inst.py   # Google Cloud Run cost analysis
    ├── gcr_cost_calculator.py
    ├── gcr_cost_visualizer_inst.py
    ├── gcr_cost_visualizer.py
    ├── check_system_state.sh    # System health checks
    └── results_*/               # Test results directories
```

## Quick Start

### Prerequisites

1. **Kubernetes cluster** with Knative Serving installed
2. **Python 3.8+** with required packages
3. **kubectl** configured for your cluster
4. **Go 1.19+** (for building invoker)
5. **System monitoring tools** (optional): `sysstat`, `iostat`

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd vswarm-analysis

# Install Python dependencies
cd latency_analysis
pip install -r requirements.txt

# Build the invoker tool
cd ../tools/invoker
make invoker
```

### Basic Usage

```bash
# Run a single test with pod monitoring
cd latency_analysis
./fair_run.sh --title fibonacci_go_a1 --service fibonacci-go-a1 --expected-pods 1

# Run adaptive testing (stops at system limits)
./fair_run.sh --title fibonacci_go_a1 --service fibonacci-go-a1 --expected-pods 1 --adaptive

# Skip visualisations (visualisation scripts of run_test.py) for faster execution
./fair_run.sh --title fibonacci_go_a1 --service fibonacci-go-a1 --expected-pods 1 --skip-visualizations
```

## Testing Framework

### Test Types

#### 1. **Fixed RPS Testing** (Default)
- Tests predefined RPS values: `[20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500]`
- 3 iterations per RPS level
- 60 seconds per test
- Comprehensive statistics and visualizations

#### 2. **Adaptive RPS Testing** (`--adaptive`)
- Starts at 20 RPS, increments by 20
- Automatically stops when system reaches capacity
- Stops when throughput drops below 85% of target AND CPU > 80%
- Or stops on infrastructure bottlenecks (error rate > 30%)

### Test Configurations

The framework supports three test series with different resource constraints:

#### **A-Series (A1-A5)**: Single-pod
- `containerConcurrency: 100`

#### **B-Series (B1-B5)**: Multi-pod horizontal scaling
- `containerConcurrency: 100`

#### **C-Series (C1-C5)**: High-performance vertical and horizontal scaling
- `containerConcurrency: 200`

## Detailed Usage

### Running Individual Tests

```bash
# Basic test execution
python3 run_test.py --title fibonacci_go_a1 --service fibonacci-go-a1 --namespace default

# With pod monitoring and constrained namespace
python3 run_test.py --title fibonacci_go_a1 --service fibonacci-go-a1 --namespace constrained-tests-a1

# Adaptive testing
python3 run_test.py --title fibonacci_go_a1 --service fibonacci-go-a1 --adaptive

# Skip tests, only analyse existing data
python3 run_test.py --title fibonacci_go_a1 --skip-tests --skip-visualizations
```

### Fair Testing Protocol

The `fair_run.sh` script provides standardized, controlled testing:

```bash
# Full fair test with cleanup and monitoring
./fair_run.sh --title fibonacci_go_a1 --service fibonacci-go-a1 --expected-pods 1

# Skip cleanup (when handled externally)
./fair_run.sh --title fibonacci_go_a1 --service fibonacci-go-a1 --expected-pods 1 --skip-cleanup

# Skip visualisations (visualisation scripts of run_test.py) for faster execution
./fair_run.sh --title fibonacci_go_a1 --service fibonacci-go-a1 --expected-pods 1 --skip-visualizations
```

**Fair Testing Features:**
- Pre-test cleanup of pods and services
- System load monitoring and waiting
- Pod count validation
- CPU pinning for consistent performance
- Comprehensive telemetry collection
- Post-test validation

### Constrained Namespace Testing

For resource-constrained testing, the framework automatically detects and validates constrained namespaces:

```bash
# Deploy constrained namespace first
kubectl apply -f ../benchmarks/fibonacci/yamls/knative/namespace-constrained-tests-a1.yaml

# Run test (automatically uses constrained namespace)
./fair_run.sh --title fibonacci_go_a1 --service fibonacci-go-a1 --expected-pods 1
```

## Analysis and Visualization

### Generated Outputs

Each test run creates a results directory with:

```
results_fibonacci_go_a1/
├── fibonacci_go_a1_statistics.csv          # Averaged statistics across iterations
├── fibonacci_go_a1_detailed_analysis.csv   # Per-iteration detailed data
├── fibonacci_go_a1_pod_monitoring.csv      # Resource usage time series
├── system_metrics/                         # System telemetry
│   ├── mpstat_*.log                       # CPU usage statistics
│   ├── iostat_*.log                       # I/O statistics
│   ├── load_*.log                         # System load
│   └── test_conditions_*.log              # Test environment details
├── charts/                                 # Generated visualisations
│   ├── comprehensive_timeseries_with_pods.png
│   ├── latency_vs_throughput.png
│   ├── p95_vs_rps.png
│   ├── p99_vs_rps.png
│   ├── avg_vs_rps.png
│   ├── resource_evolution_comprehensive.png
│   └── resource_vs_latency_correlation.png
└── target_rps*_iter*_rps*_lat.csv         # Raw latency data files
```

### Cross-Series Analysis

Generate comprehensive cross-runtime and cross-configuration analysis:

```bash
# A-series analysis (single-threaded performance)
python3 a_series_visualizations.py

# B-series analysis (multi-threaded performance)  
python3 b_series_visualizations.py

# C-series analysis (high-performance)
python3 c_series_visualizations.py

# Individual test visualizations
python3 individual_test_visualizations.py
```

### Cost Analysis

Calculate Google Cloud Run costs from telemetry data for either request-based billing or instance-based billing methods:

```bash
# Calculate costs for A-series tests
python3 gcr_cost_calculator.py --series a

# Calculate costs for specific test
python3 gcr_cost_calculator.py --test fibonacci_go_a1

# Generate cost visualizations
python3 gcr_cost_visualizer.py --series a
```

## Monitoring and Telemetry

### Pod Monitoring

The framework automatically monitors Kubernetes pods during tests:

- **Pod count tracking**: Real-time pod scaling
- **CPU usage**: Millicores consumption
- **Memory usage**: MiB consumption  
- **Pod status**: Ready/running state tracking
- **Resource limits**: CPU and memory constraints

### System Monitoring

When using `fair_run.sh`, comprehensive system telemetry is collected:

- **CPU usage**: Per-core statistics via `mpstat`
- **I/O statistics**: Disk usage via `iostat`
- **System load**: 1-minute load averages
- **Memory usage**: Available memory tracking

### Health Checks

Before running tests, the system performs health checks:

```bash
# Manual health check
./check_system_state.sh

# Checks performed:
# - System load < 2.0
# - Available memory > 8GB
# - No excessive udevd processes
# - Kubernetes cluster health
```

## Advanced Configuration

### Custom RPS Values

Modify `run_test.py` to change RPS values:

```python
# In ComprehensiveTestRunner class
DEFAULT_RPS_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Custom values
```

### Adaptive Testing Parameters

Customize adaptive testing behavior:

```python
# Stopping conditions
MIN_THROUGHPUT_RATIO = 0.85  # Stop if throughput < 85% of target
MIN_CPU_UTILIZATION = 0.80   # Only stop if CPU > 80%
MAX_ERROR_RATE = 0.30        # Stop if error rate > 30%
CONSECUTIVE_FAILURES = 1     # Stop after 1 failed RPS level
```

### Resource Constraints

Test different resource configurations by modifying the Knative service YAML files in `benchmarks/fibonacci/yamls/knative/`.

## Understanding Results

### Key Metrics

- **Throughput (RPS)**: Requests per second achieved
- **Latency percentiles**: P50, P95, P99 response times
- **CPU utilisation**: Resource usage efficiency
- **Memory usage**: Memory consumption patterns
- **Error rates**: Request failure rates
- **Pod scaling**: Auto-scaling behavior

### Performance Analysis

1. **Latency vs Throughput**: Identify performance bottlenecks
2. **Resource Evolution**: Track resource usage over time
3. **Scaling Behaviour**: Analyze pod auto-scaling patterns
4. **Cost Efficiency**: Compare cost per request across configurations

**Debug Commands:**

```bash
# Check service status
kubectl get ksvc fibonacci-go-a1 -o wide

# Check pod logs
kubectl logs -l serving.knative.dev/service=fibonacci-go-a1

# Check resource usage
kubectl top pods -l serving.knative.dev/service=fibonacci-go-a1

# Check system state
./check_system_state.sh
```

## Workflow Examples

### Complete A-Series Testing

```bash
# 1. Deploy constrained namespaces one at a time
kubectl apply -f ../benchmarks/fibonacci/yamls/knative/namespace-constrained-tests-a1.yaml

# 2. Deploy services one at a time (example for Go) 
kubectl apply -f ../benchmarks/fibonacci/yamls/knative/fibonacci-go-a1.yaml

# 3. Run tests
./fair_run.sh --title fibonacci_go_a1 --service fibonacci-go-a1 --expected-pods 1

# 4. Generate cross-series analysis
python3 a_series_visualizations.py
python3 gcr_cost_calculator.py --series a
```

### Adaptive Testing for Capacity Discovery

```bash
# Find maximum sustainable throughput
./fair_run.sh --title fibonacci_go_a1 --service fibonacci-go-a1 --expected-pods 1 --adaptive

# Results will show:
# - Maximum sustainable RPS
# - Resource utilisation at capacity
# - Performance degradation patterns
```

## Requirements

### System Requirements

- **CPU**: 4+ cores recommended
- **Memory**: 8+ GB available
- **Storage**: 10+ GB for results and logs
- **Network**: Stable connection to Kubernetes cluster

### Software Dependencies

- **Python 3.8+**
- **Go 1.19+**
- **kubectl** (configured)
- **Kubernetes cluster** with Knative Serving
- **Optional**: `sysstat`, `iostat` for system monitoring

### Python Packages

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## License

This project follows the same MIT License as the upstream vSwarm project.
This project was done in line with the MSc Computing Individual Project at Imperial College London.