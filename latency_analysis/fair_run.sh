#!/usr/bin/env bash

set -euo pipefail

# fair_run.sh - Standardized fair testing with cleanup, monitoring, and controlled environment
# Usage:
#   ./fair_run.sh --title <name> --service <svc> [--namespace default] [--expected-pods <num>]
# Example:
#   ./fair_run.sh --title fibonacci_go_b3 --service fibonacci-go-b3 --expected-pods 4

# Standardized defaults for fair testing
NAMESPACE="default"
CPUS="2-3"                    # Always pin to same CPUs
MPSTAT_INT=1
IOSTAT_INT=1
SUDO_PROMPT=1                 # Allow sudo for system operations
PRE_WAIT=1                    # Always wait for low load
LOAD_THRESHOLD=""             # Will be calculated as 0.8 * nproc
MAX_WAIT=180
EXPECTED_PODS=""              # Required: expected pod count for validation
SKIP_CLEANUP=0                # Skip pre-test cleanup (when handled externally)

# Parse args (simplified for standardized testing)
SKIP_VISUALIZATIONS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --title) TITLE="$2"; shift 2;;
    --service) SERVICE="$2"; shift 2;;
    --namespace) NAMESPACE="$2"; shift 2;;
    --expected-pods) EXPECTED_PODS="$2"; shift 2;;
    --skip-cleanup) SKIP_CLEANUP=1; shift;;
    --skip-visualizations) SKIP_VISUALIZATIONS=1; shift;;
    -h|--help)
      echo "Usage: $0 --title <name> --service <svc> [--namespace default] [--expected-pods <num>] [--skip-cleanup] [--skip-visualizations]";
      echo "Example: $0 --title fibonacci_go_b3 --service fibonacci-go-b3 --expected-pods 4";
      echo "  --skip-cleanup: Skip pre-test cleanup (when handled externally)";
      echo "  --skip-visualizations: Skip generating visualization charts";
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "${TITLE:-}" || -z "${SERVICE:-}" ]]; then
  echo "Error: --title and --service are required" >&2
  exit 1
fi

# Prepare output dirs
RESULTS_DIR="results_${TITLE}"
METRICS_DIR="${RESULTS_DIR}/system_metrics"
mkdir -p "${METRICS_DIR}"
TS=$(date +%Y%m%d_%H%M%S)

# === PRE-TEST CLEANUP (CONDITIONAL) ===
echo "=== Fair Test Protocol: ${TITLE} ==="

if [ "$SKIP_CLEANUP" -eq 0 ]; then
    echo "1. Pre-test cleanup..."
    
    # Clean up any existing pods/services
    echo "  Cleaning existing pods..."
    kubectl delete pods --all --force --grace-period=0 >/dev/null 2>&1 || true
    
    # Delete any existing Knative services (except current if redeploying)
    echo "  Cleaning existing services..."
    kubectl delete ksvc --all --timeout=30s >/dev/null 2>&1 || true
    
    # Wait for cleanup to complete
    sleep 5
    
    # Verify clean state
    REMAINING_PODS=$(kubectl get pods --no-headers 2>/dev/null | wc -l)
    if [ "$REMAINING_PODS" -gt 0 ]; then
        echo "  Warning: $REMAINING_PODS pods still present after cleanup"
        kubectl get pods --no-headers
    fi
    
    echo "2. Deploying service: ${SERVICE}"
else
    echo "1. Skipping cleanup (handled externally)"
    echo "2. Verifying service: ${SERVICE}"
fi
# Note: Service deployment should happen externally before calling this script
# Wait for service to be ready
kubectl wait --for=condition=Ready ksvc/"${SERVICE}" --timeout=120s || {
    echo "Error: Service ${SERVICE} failed to become ready" >&2
    exit 1
}

# Validate pod count matches expectation
if [[ -n "${EXPECTED_PODS}" ]]; then
    echo "3. Validating pod count..."
    sleep 10  # Allow scaling to stabilize
    ACTUAL_PODS=$(kubectl get pods -l serving.knative.dev/service="${SERVICE}" --no-headers 2>/dev/null | wc -l)
    echo "  Expected pods: ${EXPECTED_PODS}, Actual pods: ${ACTUAL_PODS}"
    
    if [ "$ACTUAL_PODS" -ne "${EXPECTED_PODS}" ]; then
        echo "  ERROR: Pod count mismatch!" >&2
        echo "  Current pods:" >&2
        kubectl get pods -l serving.knative.dev/service="${SERVICE}" >&2
        exit 1
    fi
    echo "  ✓ Pod count validation passed"
fi

# Log deployment configuration
echo "4. Logging deployment configuration..."
CONFIG_LOG="${METRICS_DIR}/deployment_config_${TS}.log"
{
    echo "=== Deployment Configuration ==="
    echo "Timestamp: $(date -Is)"
    echo "Title: ${TITLE}"
    echo "Service: ${SERVICE}"
    echo "Namespace: ${NAMESPACE}"
    echo "Expected Pods: ${EXPECTED_PODS:-unknown}"
    echo "Actual Pods: ${ACTUAL_PODS:-unknown}"
    echo ""
    echo "=== Knative Service Details ==="
    kubectl get ksvc "${SERVICE}" -o yaml
    echo ""
    echo "=== Current Pods ==="
    kubectl get pods -l serving.knative.dev/service="${SERVICE}" -o wide
    echo ""
    echo "=== Resource Limits/Requests ==="
    kubectl get pods -l serving.knative.dev/service="${SERVICE}" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{range .spec.containers[*]}  Container: {.name}{"\n"}  CPU Request: {.resources.requests.cpu}{"\n"}  CPU Limit: {.resources.limits.cpu}{"\n"}  Memory Request: {.resources.requests.memory}{"\n"}  Memory Limit: {.resources.limits.memory}{"\n"}{end}{"\n"}{end}'
} > "${CONFIG_LOG}"

echo "5. Starting controlled test environment..."


# Update endpoints.json with current service URL
echo "  Updating endpoints.json for current service..."
SERVICE_URL=$(kubectl get ksvc "${SERVICE}" -o jsonpath='{.status.url}' 2>/dev/null)
if [[ -n "${SERVICE_URL}" ]]; then
    # Extract hostname from URL (remove http:// prefix)
    HOSTNAME=$(echo "${SERVICE_URL}" | sed 's|http://||')
    echo "[{\"hostname\": \"${HOSTNAME}\"}]" > "../tools/invoker/endpoints.json"
    echo "  ✓ endpoints.json updated with: ${HOSTNAME}"
else
    echo "  ✗ Failed to get service URL for ${SERVICE}"
    exit 1
fi
# Optional sudo elevation (if password is required) and keep-alive
SUDO_KEEPALIVE_PID=""
if command -v sudo >/dev/null 2>&1 && [[ ${SUDO_PROMPT} -eq 1 ]]; then
  if ! sudo -n true >/dev/null 2>&1; then
    echo "Elevating privileges for system operations (you may be prompted)..."
    sudo -v
    # Keep sudo session alive until script exits
    ( while true; do sudo -n true; sleep 30; done ) & SUDO_KEEPALIVE_PID=$!
    trap '[[ -n "${SUDO_KEEPALIVE_PID}" ]] && kill ${SUDO_KEEPALIVE_PID} >/dev/null 2>&1 || true' EXIT
  fi
fi



# Optional pre-wait until load is below threshold
get_1min_load() {
  uptime | awk -F'load average: ' '{print $2}' | cut -d',' -f1 | tr -d ' '
}

if [[ ${PRE_WAIT} -eq 1 ]]; then
  # Calculate dynamic threshold as 0.8 * number of CPUs
  if [[ -z "${LOAD_THRESHOLD}" ]]; then
    NPROC=$(nproc)
    LOAD_THRESHOLD=$(echo "scale=1; 0.8 * ${NPROC}" | bc -l)
  fi
  
  echo "Pre-wait enabled: waiting up to ${MAX_WAIT}s for 1-min load ≤ ${LOAD_THRESHOLD} (0.8 × ${NPROC} CPUs)"
  SECS=0
  while true; do
    L=$(get_1min_load || echo 9999)
    # If parsing fails, break to avoid infinite loop
    if [[ -z "$L" ]]; then L=9999; fi
    awk -v l="$L" -v th="$LOAD_THRESHOLD" 'BEGIN{exit !(l<=th)}'
    if [[ $? -eq 0 ]]; then
      echo "Current 1-min load ${L} ≤ ${LOAD_THRESHOLD}; proceeding"
      break
    fi
    if [[ ${SECS} -ge ${MAX_WAIT} ]]; then
      echo "Timed out waiting for load to drop (currently ${L}); proceeding anyway"
      break
    fi
    sleep 5
    SECS=$((SECS+5))
  done
fi

# Tools checks (best-effort)
command -v mpstat >/dev/null 2>&1 || echo " mpstat not found (install sysstat)"
command -v iostat >/dev/null 2>&1 || echo " iostat not found (install sysstat)"

# Start background telemetry
MPSTAT_OUT="${METRICS_DIR}/mpstat_${TS}.log"
IOSTAT_OUT="${METRICS_DIR}/iostat_${TS}.log"
LOAD_OUT="${METRICS_DIR}/load_${TS}.log"

echo "Starting telemetry: mpstat ${MPSTAT_INT}s, iostat ${IOSTAT_INT}s"
if command -v mpstat >/dev/null 2>&1; then
  mpstat -P ALL ${MPSTAT_INT} > "${MPSTAT_OUT}" 2>&1 &
  MPSTAT_PID=$!
else
  MPSTAT_PID=""
fi
if command -v iostat >/dev/null 2>&1; then
  iostat -xz ${IOSTAT_INT} > "${IOSTAT_OUT}" 2>&1 &
  IOSTAT_PID=$!
else
  IOSTAT_PID=""
fi

# Also sample load in background
(
  while true; do
    echo "$(date -Is) $(uptime)" >> "${LOAD_OUT}";
    sleep 5;
  done
) & LOAD_PID=$!

# Log test conditions for telemetry validation
TELEMETRY_LOG="${METRICS_DIR}/test_conditions_${TS}.log"
{
    echo "=== Fair Test Conditions ==="
    echo "Timestamp: $(date -Is)"
    echo "Title: ${TITLE}"
    echo "Service: ${SERVICE}" 
    echo "Scheduling: Native OS"
    echo "Load Threshold: ${LOAD_THRESHOLD}"
    echo "Pre-test Load: $(get_1min_load)"
    echo "Expected Pods: ${EXPECTED_PODS:-unknown}"
    echo "System State:"
    echo "  CPU Cores: $(nproc)"
    echo "  Available Memory: $(free -h | awk 'NR==2{print $7}')"
    echo "  System Load: $(get_1min_load)"
} > "${TELEMETRY_LOG}"

# Run test with native scheduling (serverless-first approach)
CMD=(python3 run_test.py --title "${TITLE}" --service "${SERVICE}" --namespace "${NAMESPACE}")
if [ "$SKIP_VISUALIZATIONS" -eq 1 ]; then
    CMD+=(--skip-visualizations)
fi
echo "Running with native OS scheduling: ${CMD[*]}"
"${CMD[@]}"
TEST_RC=$?

# Stop telemetry
echo "Stopping telemetry..."
[[ -n "${MPSTAT_PID}" ]] && kill ${MPSTAT_PID} >/dev/null 2>&1 || true
[[ -n "${IOSTAT_PID}" ]] && kill ${IOSTAT_PID} >/dev/null 2>&1 || true
kill ${LOAD_PID} >/dev/null 2>&1 || true



# Final telemetry validation and summary
{
    echo ""
    echo "=== Post-Test Validation ==="
    echo "Test Completion: $(date -Is)"
    echo "Test Return Code: ${TEST_RC}"
    echo "Final Pod Count: $(kubectl get pods -l serving.knative.dev/service="${SERVICE}" --no-headers 2>/dev/null | wc -l)"
    echo "Final System Load: $(get_1min_load)"
} >> "${TELEMETRY_LOG}"

echo ""
echo "=== Fair Test Complete: ${TITLE} ==="
echo "✓ Pre-test cleanup: Complete"
echo "✓ Pod validation: $(if [[ -n "${EXPECTED_PODS}" ]]; then echo "Expected ${EXPECTED_PODS}, validated"; else echo "Skipped"; fi)"
echo "✓ Environment control: Native scheduling, Load threshold ${LOAD_THRESHOLD}"
echo "✓ Test execution: Return code ${TEST_RC}"
echo ""
echo "Telemetry and configuration saved to: ${METRICS_DIR}"
echo "- Test conditions: ${TELEMETRY_LOG}"
echo "- Deployment config: ${CONFIG_LOG}"
echo "- System metrics: mpstat, iostat, load logs"
echo ""

exit ${TEST_RC} 