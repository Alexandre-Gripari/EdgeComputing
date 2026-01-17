import pandas as pd
import os

DATA_DIR = "data"
OUTPUT_FILE = "data/nxp_mcuxn947_results.csv"

MODEL_PATHS = {
    "Original": "model/person_det_160x128.tflite",
    "Tiny": "model/student_tiny_quant.tflite",
    "Medium": "model/student_medium_quant.tflite",
    "Large": "model/student_large_quant.tflite",
}


def aggregate_results():
    results = []

    precision_file = os.path.join(DATA_DIR, "resultats_models.csv")
    precision_data = {}

    if os.path.exists(precision_file):
        try:
            df_prec = pd.read_csv(precision_file)
            df_prec.columns = [c.strip() for c in df_prec.columns]
            for _, row in df_prec.iterrows():
                precision_data[row["Model"]] = {
                    "mAP@50": row["mAP@50"],
                    "mAP@50-95": row["mAP@50-95"],
                }
        except Exception as e:
            print(f"Error reading precision file: {e}")

    for model_name, model_path in MODEL_PATHS.items():
        row = {"Model": model_name}

        if os.path.exists(model_path):
            try:
                size_bytes = os.path.getsize(model_path)
                row["Model_Size_kB"] = round(size_bytes / 1024, 2)
            except OSError:
                row["Model_Size_kB"] = None
        else:
            row["Model_Size_kB"] = None

        bench_file = os.path.join(DATA_DIR, f"benchmark_{model_name}.csv")
        if os.path.exists(bench_file):
            try:
                df_bench = pd.read_csv(bench_file)
                if "Inference_us" in df_bench.columns and not df_bench.empty:
                    row["Inference_Min_us"] = df_bench["Inference_us"].min()
                    row["Inference_Max_us"] = df_bench["Inference_us"].max()
                    row["Inference_Mean_us"] = round(df_bench["Inference_us"].mean(), 2)
                    row["Inference_Std_us"] = round(df_bench["Inference_us"].std(), 2)
            except Exception as e:
                print(f"Error reading {bench_file}: {e}")

        consommation_file = os.path.join(
            DATA_DIR, f"resultats_mesures_{model_name.lower()}.csv"
        )

        if os.path.exists(consommation_file):
            try:
                df_conso = pd.read_csv(consommation_file)
                if "Courant (A)" in df_conso.columns and not df_conso.empty:
                    row["Current_Min_A"] = df_conso["Courant (A)"].min()
                    row["Current_Max_A"] = df_conso["Courant (A)"].max()
                    row["Current_Mean_A"] = round(df_conso["Courant (A)"].mean(), 4)
                    row["Current_Std_A"] = round(df_conso["Courant (A)"].std(), 4)

                    if "Watt (W)" in df_conso.columns:
                        row["Power_Mean_W"] = round(df_conso["Watt (W)"].mean(), 4)
            except Exception as e:
                print(f"Error reading {consommation_file}: {e}")

        info_file = os.path.join(DATA_DIR, f"model_info_{model_name}.csv")
        if os.path.exists(info_file):
            try:
                df_info = pd.read_csv(info_file)
                if not df_info.empty:
                    last_line = df_info.iloc[-1]
                    row["RAM_Used_kB"] = last_line["Used_kB"]
                    row["RAM_Total_kB"] = last_line["Total_kB"]
                    row["RAM_Percent"] = last_line["Percent_Used"]
            except Exception:
                pass

        if model_name in precision_data:
            row.update(precision_data[model_name])

        results.append(row)

    if results:
        df_final = pd.DataFrame(results)

        cols_order = [
            "Model",
            "Model_Size_kB",
            "mAP@50",
            "mAP@50-95",
            "Inference_Min_us",
            "Inference_Mean_us",
            "Inference_Max_us",
            "Inference_Std_us",
            "RAM_Used_kB",
            "RAM_Percent",
            "Current_Min_A",
            "Current_Mean_A",
            "Current_Max_A",
            "Current_Std_A",
            "Power_Mean_W",
        ]

        final_cols = [c for c in cols_order if c in df_final.columns]
        remaining = [c for c in df_final.columns if c not in final_cols]
        df_final = df_final[final_cols + remaining]

        try:
            df_final.to_csv(OUTPUT_FILE, index=False)
            print(f"\nAggregated file created: {OUTPUT_FILE}")
            print(f"Results:\n")
            print(df_final.to_string())
        except Exception as e:
            print(f"Unable to write {OUTPUT_FILE}: {e}")
    else:
        print("No results found.")


if __name__ == "__main__":
    aggregate_results()
