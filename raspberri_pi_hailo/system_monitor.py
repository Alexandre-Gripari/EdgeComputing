#!/usr/bin/env python3
import time
import psutil
import subprocess
import sys
import os
import argparse
import csv
import statistics

def _parse_throttle_flags(code):
    active = []
    occurred = []
    if code & 0x1: active.append("LOW_VOLT")
    if code & 0x2: active.append("FREQ_CAP")
    if code & 0x4: active.append("THROTTLED")
    if code & 0x8: active.append("TEMP_LIM")
    if code & 0x10000: occurred.append("LOW_VOLT")
    if code & 0x20000: occurred.append("FREQ_CAP")
    if code & 0x40000: occurred.append("THROTTLED")
    if code & 0x80000: occurred.append("TEMP_LIM")
    return active, occurred

def get_throttled_status():
    try:
        res = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, check=True)
        output = res.stdout.strip()
        parts = output.split('=')
        if len(parts) < 2:
            return {"active": [], "occurred": [], "raw": None, "label": "Err"}
        code = int(parts[1].strip(), 16)
        active, occurred = _parse_throttle_flags(code)
        label = "OK" if not active else "+".join(active)
        return {"active": active, "occurred": occurred, "raw": code, "label": label}
    except Exception:
        return {"active": [], "occurred": [], "raw": None, "label": "Err"}

def get_temp():
    try:
        temps = psutil.sensors_temperatures()
        preferred_keys = ["cpu_thermal", "soc_thermal", "cpu", "SoC"]
        for k in preferred_keys:
            if k in temps and temps[k]:
                return temps[k][0].current
        max_val = None
        for arr in temps.values():
            for entry in arr:
                cur = getattr(entry, "current", None)
                if cur is not None:
                    max_val = cur if max_val is None else max(max_val, cur)
        if max_val is not None:
            return max_val
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except Exception:
        return None

def get_fan_info():
    base = "/sys/class/thermal"
    try:
        devices = [d for d in os.listdir(base) if d.startswith("cooling_device")]
        best = None
        for d in devices:
            path = os.path.join(base, d)
            type_path = os.path.join(path, "type")
            cur_path = os.path.join(path, "cur_state")
            max_path = os.path.join(path, "max_state")
            try:
                with open(type_path) as tf:
                    typ = tf.read().strip().lower()
                with open(cur_path) as cf:
                    cur = int(cf.read().strip())
                with open(max_path) as mf:
                    mx = int(mf.read().strip())
                if best is None:
                    best = {"type": typ, "cur": cur, "max": mx}
                elif "fan" in typ or "pwm" in typ:
                    best = {"type": typ, "cur": cur, "max": mx}
            except Exception:
                continue
        return best
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="System monitoring with stats, throttling and fan info")
    parser.add_argument("--interval", type=float, default=0.5, help="Polling interval in seconds")
    parser.add_argument("--warn-temp", type=float, default=75.0, help="Temperature warning threshold (°C)")
    parser.add_argument("--duration", type=float, default=None, help="Stop after N seconds (optional)")
    parser.add_argument("--out", type=str, default=None, help="CSV output file path (optional)")
    args = parser.parse_args()

    print("-" * 80)
    print(f"{'TIME':<10} | {'CPU %':<8} | {'RAM %':<8} | {'TEMP':<8} | {'STATUS'}")
    print("-" * 80)

    hist_cpu = []
    hist_ram = []
    hist_temp = []
    throttled_active_events = []
    throttled_occurred_events = []

    psutil.cpu_percent(interval=None)

    csv_file = None
    csv_writer = None
    if args.out:
        try:
            csv_file = open(args.out, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["time", "cpu", "ram", "temp", "status_active", "status_occurred", "fan_cur", "fan_max"])
        except Exception:
            csv_file = None
            csv_writer = None

    start = time.monotonic()
    try:
        while True:
            if args.duration is not None and (time.monotonic() - start) >= args.duration:
                break
            time.sleep(args.interval)

            current_time = time.strftime("%H:%M:%S")
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            temp = get_temp()
            status = get_throttled_status()
            fan = get_fan_info()

            hist_cpu.append(cpu)
            hist_ram.append(ram)
            if temp is not None:
                hist_temp.append(temp)
            if status["active"]:
                throttled_active_events.extend(status["active"])
            if status["occurred"]:
                throttled_occurred_events.extend(status["occurred"])

            alerts = []
            if temp is not None and temp > args.warn_temp:
                alerts.append("HOT")
            if cpu > 90:
                alerts.append("CPU_HIGH")
            if status["active"]:
                alerts.append("THROTTLING")
            alert_str = " + ".join(alerts) if alerts else ""

            cpu_str = f"{cpu:.1f}%"
            temp_str = f"{temp:.1f}°C" if temp is not None else "N/A"
            label = status["label"]
            if alert_str:
                label = f"{label} <--- {alert_str}"

            print(f"{current_time:<10} | {cpu_str:<8} | {ram:<8.1f}% | {temp_str:<8} | {label}")

            if csv_writer:
                fan_cur = fan["cur"] if fan else None
                fan_max = fan["max"] if fan else None
                csv_writer.writerow([current_time, cpu, ram, temp if temp is not None else "", ",".join(status["active"]), ",".join(status["occurred"]), fan_cur, fan_max])

    except KeyboardInterrupt:
        pass
    finally:
        if csv_file:
            try:
                csv_file.close()
            except Exception:
                pass

    print("\n" + "=" * 80)
    print(" RAPPORT STATISTIQUE DETAILLE")
    print("=" * 80)

    if hist_cpu:
        def print_stat(name, data, unit):
            mn = min(data)
            mx = max(data)
            avg = statistics.mean(data)
            delta = mx - mn
            stdev = statistics.stdev(data) if len(data) > 1 else 0.0
            print(f"{name:<12} | Min: {mn:>5.1f} | Moy: {avg:>5.1f} | Max: {mx:>5.1f} | Delta: {delta:>5.1f} | StDev: {stdev:>4.2f} {unit}")

        print_stat("CPU Usage", hist_cpu, "%")
        print_stat("RAM Usage", hist_ram, "%")
        if hist_temp:
            print_stat("Temp CPU", hist_temp, "C")

    print("-" * 80)
    fan = get_fan_info()
    if fan:
        print(f"Ventilo    | {fan['type']} {fan['cur']} / {fan['max']}")
    else:
        print("Ventilo    | N/A")

    if throttled_active_events or throttled_occurred_events:
        print("THROTTLING | Evenements")
        if throttled_active_events:
            print(f"             Actifs: {set(throttled_active_events)}")
        if throttled_occurred_events:
            print(f"             Historique: {set(throttled_occurred_events)}")
    else:
        print("THROTTLING | Aucun (Systeme Stable)")

    print("=" * 80)

if __name__ == "__main__":
    main()