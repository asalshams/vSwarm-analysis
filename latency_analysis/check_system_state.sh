#!/bin/bash

# Check system load
uptime

# Check memory
free -h

# Check CPU usage
ps aux --sort=-%cpu | head -10

# Check Kubernetes state
kubectl get pods -A --field-selector=status.phase!=Running

# Check node resources
kubectl top nodes

# Check for high CPU processes
ps aux | awk '$3 > 50 {print $3"%", $11}' | head -5

# Check udevd processes
UDEVD_COUNT=$(pgrep -c systemd-udevd)
if [ "$UDEVD_COUNT" -gt 5 ]; then
    exit 1
fi

# Check system load threshold
LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
LOAD_NUM=$(echo $LOAD | sed 's/,//')

if (( $(echo "$LOAD_NUM > 2.0" | bc -l) )); then
    exit 1
fi

# Check available memory
AVAIL_MEM=$(free -g | awk 'NR==2{print $7}')
if [ "$AVAIL_MEM" -lt 8 ]; then
    exit 1
fi 