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
        self.stats = {
            'cpu_usage': [],
            'gpu_usage': [],
            'ram_usage': [],
            'swap_usage': [],
            'fan_pwm': [],
            'cpu_temp': [],
            'gpu_temp': [],
            'power_total': [],
            'power_vdd_soc': [],
            'power_gpu_cpu_cv': []
        }
        self.thread = None

    def _monitor_loop(self):
        with jtop() as jetson:
            while self.running and jetson.ok():
                try:
                    cpu_total = 0
                    for i in range(1,7):
                        cpu_total += jetson.stats.get(f'CPU{i}', 0)
                    self.stats['cpu_usage'].append(cpu_total / 6)

                    self.stats['gpu_usage'].append(jetson.stats.get('GPU', 0))
                    self.stats['ram_usage'].append(jetson.stats.get('RAM', 0))
                    self.stats['swap_usage'].append(jetson.stats.get('SWAP', 0))
                    self.stats['fan_pwm'].append(jetson.stats.get('Fan pwmfan0', 0))
                    self.stats['cpu_temp'].append(jetson.stats.get('Temp CPU', 0))
                    self.stats['gpu_temp'].append(jetson.stats.get('Temp GPU', 0))
                    self.stats['power_total'].append(jetson.stats.get('Power TOT', 0))
                    self.stats['power_vdd_soc'].append(jetson.stats.get('Power VDD_SOC', 0))
                    self.stats['power_gpu_cpu_cv'].append(jetson.stats.get('Power VDD_CPU_GPU_CV', 0))

                except Exception as e:
                    print(f"Monitoring error: {e}")
                    pass
                
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
        def safe_mean(data):
            valid_data = [x for x in data if x is not None]
            return statistics.mean(valid_data) if valid_data else 0.0

        return {
            'CPU Usage (%)': safe_mean(self.stats['cpu_usage']),
            'GPU Usage (%)': safe_mean(self.stats['gpu_usage']),
            'RAM Usage (%)': safe_mean(self.stats['ram_usage']),
            'Swap Usage (%)': safe_mean(self.stats['swap_usage']),
            'Fan PWM (%)': safe_mean(self.stats['fan_pwm']),
            'Temp CPU (C)': safe_mean(self.stats['cpu_temp']),
            'Temp GPU (C)': safe_mean(self.stats['gpu_temp']),
            'Power (mW)': safe_mean(self.stats['power_total']),
            'Power VDD_SOC (mW)': safe_mean(self.stats['power_vdd_soc']),
            'Power VDD_CPU_GPU_CV (mW)': safe_mean(self.stats['power_gpu_cpu_cv']),
        }

def run_benchmark(data_path, models_folder, output_csv):
    model_files = [f for f in os.listdir(models_folder) if f.endswith(VALID_EXTENSIONS)]
    model_files.sort()
    
    if not model_files:
        print(f"No models found in {models_folder}")
        return

    print(f"Found {len(model_files)} models. Starting benchmark on {data_path}")

    headers = ['Model', 'Format', 'File Size (MB)', 'mAP50-95', 'mAP50', 'Inference Time (ms)', 'Preprocess Time (ms)', 'Postprocess Time (ms)',
               'CPU Usage (%)', 'GPU Usage (%)', 'RAM Usage (%)', 'Swap Usage (%)', 'Fan PWM (%)', 'Temp CPU (C)', 'Temp GPU (C)',
               'Power (mW)', 'Power VDD_SOC (mW)', 'Power VDD_CPU_GPU_CV (mW)']
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

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
                
                monitor = HardwareMonitor()
                monitor.start()
                
                results = model.val(data=data_path, imgsz=640, verbose=False, device=0)
                
                monitor.stop()
                hw_stats = monitor.get_averages()
                
                map50_95 = results.box.map
                map50 = results.box.map50     
                t_inf = results.speed['inference']
                t_pre = results.speed['preprocess']
                t_post = results.speed['postprocess']

                writer.writerow([
                    model_file, 
                    ext, 
                    f"{file_size_mb:.2f}",
                    f"{map50_95:.4f}", 
                    f"{map50:.4f}", 
                    f"{t_inf:.2f}", 
                    f"{t_pre:.2f}",
                    f"{t_post:.2f}",
                    hw_stats['CPU Usage (%)'],
                    hw_stats['GPU Usage (%)'],
                    hw_stats['RAM Usage (%)'],
                    hw_stats['Swap Usage (%)'],
                    hw_stats['Fan PWM (%)'],
                    hw_stats['Temp CPU (C)'],
                    hw_stats['Temp GPU (C)'],
                    hw_stats['Power (mW)'],
                    hw_stats['Power VDD_SOC (mW)'],
                    hw_stats['Power VDD_CPU_GPU_CV (mW)']
                ])
                
                del model
                del results

            except Exception as e:
                print(f"Errore: {e}")
                monitor.stop()

    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Jetson Orin YOLO Benchmark Tool')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='Path to dataset yaml (e.g. coco128.yaml)')
    parser.add_argument('--models', type=str, required=True, help='Folder containing model files')
    parser.add_argument('--output', type=str, default='benchmark_results.csv', help='Output CSV filename')
    
    args = parser.parse_args()
    
    run_benchmark(args.data, args.models, args.output)
