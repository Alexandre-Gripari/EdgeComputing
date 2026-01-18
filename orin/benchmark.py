import os
import csv
import time
import argparse
import threading
import statistics
import numpy as np
from ultralytics import YOLO
from jtop import jtop

VALID_EXTENSIONS = ('.pt', '.onnx', '.engine')

class HardwareMonitor:
    def __init__(self):
        self.running = False
        self.output_mapping = {
            'CPU Usage (%)': 'cpu_usage',
            'GPU Usage (%)': 'gpu_usage',
            'RAM Usage (%)': 'ram_usage',
            'Swap Usage (%)': 'swap_usage',
            'Fan PWM (%)': 'fan_pwm',
            'Temp CPU (C)': 'cpu_temp',
            'Temp GPU (C)': 'gpu_temp',
            'Power (mW)': 'power_total',
            'Power VDD_SOC (mW)': 'power_vdd_soc',
            'Power VDD_CPU_GPU_CV (mW)': 'power_gpu_cpu_cv'
        }
        self.stats = {v: [] for v in self.output_mapping.values()}
        self.thread = None

    def _monitor_loop(self):
        with jtop() as jetson:
            while self.running and jetson.ok():
                try:
                    data = {
                        'cpu_usage': sum(jetson.stats.get(f'CPU{i}', 0) for i in range(1, 7)) / 6,
                        'gpu_usage': jetson.stats.get('GPU', 0),
                        'ram_usage': jetson.stats.get('RAM', 0),
                        'swap_usage': jetson.stats.get('SWAP', 0),
                        'fan_pwm': jetson.stats.get('Fan pwmfan0', 0),
                        'cpu_temp': jetson.stats.get('Temp CPU', 0),
                        'gpu_temp': jetson.stats.get('Temp GPU', 0),
                        'power_total': jetson.stats.get('Power TOT', 0),
                        'power_vdd_soc': jetson.stats.get('Power VDD_SOC', 0),
                        'power_gpu_cpu_cv': jetson.stats.get('Power VDD_CPU_GPU_CV', 0)
                    }
                    for k, v in data.items():
                        self.stats[k].append(v)
                except Exception as e:
                    print(f"Monitoring error: {e}")

    def start(self):
        self.running = True
        self.stats = {k: [] for k in self.stats}
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_averages(self):
        gpu_data = self.stats['gpu_usage']
        if not gpu_data:
            return {k: 0.0 for k in self.output_mapping}

        active_indices = [i for i, x in enumerate(gpu_data) if x > 0]
        if not active_indices:
            active_indices = list(range(len(gpu_data)))

        def filtered_mean(key):
            data = self.stats[key]
            valid_values = [data[i] for i in active_indices if i < len(data) and data[i] is not None]
            return statistics.mean(valid_values) if valid_values else 0.0

        return {label: filtered_mean(key) for label, key in self.output_mapping.items()}

def run_benchmark(data_path, models_folder, output_csv):
    model_files = sorted([f for f in os.listdir(models_folder) if f.endswith(VALID_EXTENSIONS)])
    
    if not model_files:
        print(f"No models found in {models_folder}")
        return

    print(f"Found {len(model_files)} models. Starting benchmark on {data_path}")

    monitor = HardwareMonitor()
    hw_headers = list(monitor.output_mapping.keys())
    base_headers = ['Model', 'Format', 'File Size (MB)', 'mAP50-95', 'mAP50', 
                    'Inference Time (ms)', 'Preprocess Time (ms)', 'Postprocess Time (ms)']
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(base_headers + hw_headers)

        monitor.start()
        time.sleep(5)
        monitor.stop()
        hw_stats = monitor.get_averages()

        writer.writerow(['/'] * 8 + [hw_stats[k] for k in hw_headers])

        for model_file in model_files:
            model_path = os.path.join(models_folder, model_file)
            print(f"Testing: {model_file}")
            
            ext = os.path.splitext(model_file)[1]
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            try:
                model = YOLO(model_path, task='detect')
                
                dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                for _ in range(20):
                    model(dummy_input, verbose=False)
                
                monitor.start()
                results = model.val(data=data_path, imgsz=640, verbose=False, device=0)
                monitor.stop()
                
                hw_stats = monitor.get_averages()
                
                row_data = [
                    model_file,
                    ext,
                    f"{file_size_mb:.2f}",
                    f"{results.box.map:.4f}",
                    f"{results.box.map50:.4f}",
                    f"{results.speed['inference']:.2f}",
                    f"{results.speed['preprocess']:.2f}",
                    f"{results.speed['postprocess']:.2f}"
                ]
                
                writer.writerow(row_data + [hw_stats[k] for k in hw_headers])
                
                del model
                del results

            except Exception as e:
                print(f"Error: {e}")
                monitor.stop()

    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Jetson Orin YOLO Benchmark Tool')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='Path to dataset yaml (e.g. coco128.yaml)')
    parser.add_argument('--models', type=str, required=True, help='Folder containing model files')
    parser.add_argument('--output', type=str, default='benchmark_results.csv', help='Output CSV filename')
    
    args = parser.parse_args()
    run_benchmark(args.data, args.models, args.output)
