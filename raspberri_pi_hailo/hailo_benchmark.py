#!/usr/bin/env python3
import subprocess
import argparse
import shutil
import re
import sys
import os
import time

def check_dependencies():
    if not shutil.which("hailortcli"):
        print("Error: 'hailortcli' not found. Please verify HailoRT is installed.")
        return False
    return True

def check_pcie_status():
    try:
        cmd = "sudo lspci -vv | grep -A 20 'Hailo' | grep 'LnkSta:' | grep 'Speed 8GT/s'"
        full_output = subprocess.check_output("sudo lspci -vv", shell=True).decode()
        
        if "Hailo" in full_output and "Speed 8GT/s" in full_output:
             return "Gen 3.0 (8GT/s) - OPTIMAL"
        elif "Hailo" in full_output and "Speed 5GT/s" in full_output:
             return "Gen 2.0 (5GT/s) - RESTRICTED"
        else:
             return "Unknown / Error reading PCIe"
    except Exception as e:
        return f"Error checking PCIe: {e}"

def run_command(command, dry_run=False):
    print(f"Executing: {' '.join(command)}")
    if dry_run:
        return "Dry run output"
    
    try:
        result = subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.stderr}")
        return None

def parse_benchmark_output(output):
    fps = None
    if output:
        match = re.search(r"FPS:\s*([\d\.]+)", output, re.IGNORECASE)
        if match:
            fps = float(match.group(1))
    return fps

def parse_run_stats(output):
    stats = {}
    if not output: return stats
    
    hw_lat = re.search(r"HW Latency:\s*([\d\.]+)\s*ms", output, re.IGNORECASE)
    if hw_lat: stats['hw_latency'] = float(hw_lat.group(1))

    ov_lat = re.search(r"Overall Latency:\s*([\d\.]+)\s*ms", output, re.IGNORECASE)
    if ov_lat: stats['overall_latency'] = float(ov_lat.group(1))
    
    avg_patterns = [
        r"Average\s+chip\s+temperature:\s*([\d\.]+)\s*[°]?\s*C",
        r"Average\s+temperature:\s*([\d\.]+)\s*[°]?\s*C",
        r"Avg(?:erage)?\s+chip\s+temp(?:erature)?:\s*([\d\.]+)\s*[°]?\s*C",
    ]
    min_patterns = [
        r"(?:Minimum|Min)\s+chip\s+temperature:\s*([\d\.]+)\s*[°]?\s*C",
        r"Chip\s+temperature\s+(?:minimum|min):\s*([\d\.]+)\s*[°]?\s*C",
        r"(?:Minimum|Min)\s+temperature:\s*([\d\.]+)\s*[°]?\s*C",
    ]
    max_patterns = [
        r"(?:Maximum|Max)\s+chip\s+temperature:\s*([\d\.]+)\s*[°]?\s*C",
        r"Chip\s+temperature\s+(?:maximum|max):\s*([\d\.]+)\s*[°]?\s*C",
        r"(?:Maximum|Max)\s+temperature:\s*([\d\.]+)\s*[°]?\s*C",
    ]

    def first_match(patterns):
        for pat in patterns:
            m = re.search(pat, output, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        return None

    t_avg = first_match(avg_patterns)
    t_min = first_match(min_patterns)
    t_max = first_match(max_patterns)

    if t_avg is not None:
        stats['temp_avg_c'] = t_avg
        stats['temp_c'] = t_avg
    if t_min is not None:
        stats['temp_min_c'] = t_min
    if t_max is not None:
        stats['temp_max_c'] = t_max
    if (t_min is not None) and (t_max is not None):
        stats['temp_delta_c'] = t_max - t_min
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="RPi5 + Hailo-8L Real-Time Benchmark")
    parser.add_argument("--hef", type=str, help="Path to HEF file")
    parser.add_argument("--time", type=int, default=60, help="Duration in seconds")
    args = parser.parse_args()

    hef_file = args.hef
    if not hef_file:
        files = [f for f in os.listdir('.') if f.endswith('.hef')]
        if files:
            hef_file = files[0]
            print(f"Using found HEF: {hef_file}")
        else:
            print("Error: No .hef file found.")
            sys.exit(1)

    if not check_dependencies(): sys.exit(1)

    print("\n" + "="*60)
    print("      HAILO REAL-TIME BENCHMARK (Batch=1)")
    print("="*60)
    
    pcie_status = check_pcie_status()
    print(f"Hardware Check: {pcie_status}")

    print("\n[Phase 1] Measuring Max FPS (Batch=1)...")
    bench_cmd = ["hailortcli", "benchmark", hef_file, "--batch-size", "1", "--time-to-run", "60"]
    bench_out = run_command(bench_cmd)
    fps = parse_benchmark_output(bench_out)

    print(f"\n[Phase 2] Measuring Stability & Latency ({args.time}s)...")
    run_cmd = [
        "hailortcli", "run", hef_file, 
        "--measure-latency", 
        "--measure-overall-latency",
        "--measure-temp",
        "--time-to-run", str(args.time)
    ]
    run_out = run_command(run_cmd)
    stats = parse_run_stats(run_out)

    print("\n" + "="*60)
    print(f" REPORT: {hef_file}")
    print("="*60)
    print(f"{'PCIe Status':<20} | {pcie_status}")
    print("-" * 45)
    print(f"{'FPS (Real-time)':<20} | {fps if fps else 'N/A'}")
    
    lat_hw = f"{stats.get('hw_latency', 'N/A')} ms"
    lat_ov = f"{stats.get('overall_latency', 'N/A')} ms"
    print(f"{'Latency (Chip)':<20} | {lat_hw}")
    print(f"{'Latency (Overall)':<20} | {lat_ov}")
    
    print("-" * 45)
    temp_avg = stats.get('temp_avg_c', stats.get('temp_c'))
    temp_min = stats.get('temp_min_c')
    temp_max = stats.get('temp_max_c')
    temp_delta = stats.get('temp_delta_c') if (stats.get('temp_delta_c') is not None) else ( (temp_max - temp_min) if (temp_max is not None and temp_min is not None) else None )

    avg_str = f"{temp_avg} C" if temp_avg is not None else "N/A"
    min_str = f"{temp_min} C" if temp_min is not None else "N/A"
    max_str = f"{temp_max} C" if temp_max is not None else "N/A"
    delta_str = f"{round(temp_delta, 2)} C" if temp_delta is not None else "N/A"

    if temp_max is not None and temp_max > 75:
        max_str += " (Throttling risk!)"

    print(f"{'Temperature (Avg)':<20} | {avg_str}")
    print(f"{'Temperature (Min)':<20} | {min_str}")
    print(f"{'Temperature (Max)':<20} | {max_str}")
    print(f"{'Temperature (Delta)':<20} | {delta_str}")
    print("="*60)

if __name__ == "__main__":
    main()