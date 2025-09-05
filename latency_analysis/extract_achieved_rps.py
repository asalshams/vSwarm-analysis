#!/usr/bin/env python3
"""
Extract actual achieved RPS from filename patterns and create summary files.
Creates an 'achieved_rps_summary.csv' file in each results directory.

Filename pattern: target_rps{TARGET}_iter{ITERATION}_rps{ACHIEVED}_lat.csv
"""

import os
import re
import csv
import sys
from pathlib import Path

def extract_rps_data_from_filename(filename):
    """
    Extract target RPS, iteration, and achieved RPS from filename.
    Returns: (target_rps, iteration, achieved_rps) or None if pattern doesn't match
    """
    pattern = r'target_rps(\d+\.\d{2})_iter(\d{2})_rps(\d+\.\d{2})_lat\.csv'
    match = re.match(pattern, filename)
    
    if match:
        target_rps = float(match.group(1))
        iteration = int(match.group(2))
        achieved_rps = float(match.group(3))
        return target_rps, iteration, achieved_rps
    return None

def process_results_directory(results_dir):
    """
    Process a single results directory and create achieved_rps_summary.csv
    """
    print(f"Processing: {results_dir}")
    
    # Dictionary to store data: {target_rps: {iteration: achieved_rps}}
    rps_data = {}
    
    # Find all latency CSV files
    for filename in os.listdir(results_dir):
        if filename.endswith('_lat.csv'):
            data = extract_rps_data_from_filename(filename)
            if data:
                target_rps, iteration, achieved_rps = data
                
                if target_rps not in rps_data:
                    rps_data[target_rps] = {}
                
                rps_data[target_rps][iteration] = achieved_rps
    
    if not rps_data:
        print(f"  No matching files found in {results_dir}")
        return
    
    # Create summary CSV file
    summary_file = os.path.join(results_dir, 'achieved_rps_summary.csv')
    
    # Track maximum RPS for each iteration and overall
    max_iter1 = max_iter2 = max_iter3 = 0
    overall_max = 0
    best_target_iter1 = best_target_iter2 = best_target_iter3 = best_target_overall = 0
    
    with open(summary_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['target_rps', 'iter1_achieved', 'iter2_achieved', 'iter3_achieved', 
                        'avg_achieved', 'min_achieved', 'max_achieved', 'efficiency_pct'])
        
        # Sort by target RPS
        for target_rps in sorted(rps_data.keys()):
            iterations = rps_data[target_rps]
            
            # Get achieved RPS for each iteration (default to None if missing)
            iter1 = iterations.get(1, None)
            iter2 = iterations.get(2, None)
            iter3 = iterations.get(3, None)
            
            # Track maximum values
            if iter1 and iter1 > max_iter1:
                max_iter1 = iter1
                best_target_iter1 = target_rps
            if iter2 and iter2 > max_iter2:
                max_iter2 = iter2
                best_target_iter2 = target_rps
            if iter3 and iter3 > max_iter3:
                max_iter3 = iter3
                best_target_iter3 = target_rps
            
            # Track overall maximum
            for val in [iter1, iter2, iter3]:
                if val and val > overall_max:
                    overall_max = val
                    best_target_overall = target_rps
            
            # Calculate statistics for available iterations
            achieved_values = [v for v in [iter1, iter2, iter3] if v is not None]
            
            if achieved_values:
                avg_achieved = sum(achieved_values) / len(achieved_values)
                min_achieved = min(achieved_values)
                max_achieved = max(achieved_values)
                efficiency_pct = (avg_achieved / target_rps) * 100
            else:
                avg_achieved = min_achieved = max_achieved = efficiency_pct = None
            
            writer.writerow([
                f"{target_rps:.2f}",
                f"{iter1:.2f}" if iter1 else "",
                f"{iter2:.2f}" if iter2 else "",
                f"{iter3:.2f}" if iter3 else "",
                f"{avg_achieved:.2f}" if avg_achieved else "",
                f"{min_achieved:.2f}" if min_achieved else "",
                f"{max_achieved:.2f}" if max_achieved else "",
                f"{efficiency_pct:.1f}" if efficiency_pct else ""
            ])
        
        # Add summary statistics at the end
        writer.writerow([])  # Empty row
        writer.writerow(['=== MAXIMUM ACHIEVED RPS SUMMARY ==='])
        writer.writerow(['Metric', 'Value', 'Target_RPS'])
        writer.writerow(['Max Iter1', f"{max_iter1:.2f}", f"{best_target_iter1:.2f}"])
        writer.writerow(['Max Iter2', f"{max_iter2:.2f}", f"{best_target_iter2:.2f}"])
        writer.writerow(['Max Iter3', f"{max_iter3:.2f}", f"{best_target_iter3:.2f}"])
        writer.writerow(['Overall Max', f"{overall_max:.2f}", f"{best_target_overall:.2f}"])
        
        # Calculate average of maximums
        max_values = [v for v in [max_iter1, max_iter2, max_iter3] if v > 0]
        if max_values:
            avg_max = sum(max_values) / len(max_values)
            writer.writerow(['Avg of Max', f"{avg_max:.2f}", ''])
    
    print(f"  Created: {summary_file}")
    print(f"  Processed {len(rps_data)} target RPS levels")

def find_results_directories(base_dir):
    """
    Find all results directories that match the pattern 'results_fibonacci_*'
    """
    results_dirs = []
    
    # Check if we're already in the latency_analysis directory
    if os.path.basename(base_dir) == 'latency_analysis':
        search_dir = base_dir
    else:
        search_dir = os.path.join(base_dir, 'latency_analysis')
    
    if not os.path.exists(search_dir):
        print(f"Directory not found: {search_dir}")
        return results_dirs
    
    for item in os.listdir(search_dir):
        if item.startswith('results_fibonacci_'):
            full_path = os.path.join(search_dir, item)
            if os.path.isdir(full_path):
                results_dirs.append(full_path)
    
    return sorted(results_dirs)

def main():
    """
    Main function to process all results directories
    """
    # Get the current working directory or use provided argument
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = os.getcwd()
    
    print(f"Scanning for results directories in: {base_dir}")
    
    # Find all results directories
    results_dirs = find_results_directories(base_dir)
    
    if not results_dirs:
        print("No results directories found matching pattern 'results_fibonacci_*'")
        return
    
    print(f"Found {len(results_dirs)} results directories:")
    for dir_path in results_dirs:
        print(f"  {os.path.basename(dir_path)}")
    
    print("\n" + "="*60)
    
    # Process each directory
    for results_dir in results_dirs:
        try:
            process_results_directory(results_dir)
        except Exception as e:
            print(f"Error processing {results_dir}: {e}")
        print()
    
    print("="*60)
    print("Summary extraction completed!")
    print(f"Check each results directory for 'achieved_rps_summary.csv' files")

if __name__ == "__main__":
    main()
