import serial
import csv
import re
import sys
import time  # <--- AJOUT : Pour le timestamp côté PC

SERIAL_PORT = "COM4"
BAUD_RATE = 115200

MODELS = {"1": "Original", "2": "Tiny", "3": "Medium", "4": "Large"}


def get_model_selection():
    print("\n--- SÉLECTION DU MODÈLE ---")
    for key, name in MODELS.items():
        print(f"  [{key}] {name}")

    while True:
        choice = input(
            "\nQuel modèle est actuellement déployé sur la carte ? (1-4) : "
        ).strip()
        if choice in MODELS:
            return MODELS[choice]
        print("Choix invalide. Veuillez entrer 1, 2, 3 ou 4.")


def parse_serial_data():
    model_name = get_model_selection()

    benchmark_file = f"data/benchmark_{model_name}.csv"
    info_file = f"data/model_info_{model_name}.csv"

    # <--- MODIFICATION : Regex mise à jour pour capturer TS
    # Capture maintenant 4 groupes : TS, PRE, INF, POST
    log_pattern = re.compile(r"LOG_DATA: TS=(\d+) PRE=(\d+) INF=(\d+) POST=(\d+)")

    arena_pattern = re.compile(r"Arena: (\d+)/(\d+) kB \(([\d.]+)%\) used")

    print(f"\n--- Configuration pour le modèle : {model_name} ---")
    # ... (code affichage identique) ...

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Erreur port série: {e}")
        return

    try:
        with open(benchmark_file, mode="w", newline="") as f_bench, open(
            info_file, mode="w", newline=""
        ) as f_info:

            writer_bench = csv.writer(f_bench)
            writer_info = csv.writer(f_info)

            writer_bench.writerow(
                ["Board_Time_us", "PreProcess_us", "Inference_us", "PostProcess_us"]
            )

            writer_info.writerow(["Model_Name", "Used_kB", "Total_kB", "Percent_Used"])

            print(f"En attente de données pour {model_name}...")

            while True:
                try:
                    line_bytes = ser.readline()
                    if not line_bytes:
                        continue

                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                    print(f"[BOARD] {line}")

                    match_arena = arena_pattern.search(line)
                    if match_arena:
                        # ... (code arena identique) ...
                        used_kb = match_arena.group(1)
                        total_kb = match_arena.group(2)
                        percent = match_arena.group(3)
                        writer_info.writerow([model_name, used_kb, total_kb, percent])
                        f_info.flush()
                        print(
                            f"\n>>> RAM ({model_name}) : {used_kb}kB / {total_kb}kB ({percent}%)\n"
                        )

                    match_log = log_pattern.search(line)
                    if match_log:
                        board_ts = match_log.group(1)
                        pre_time = match_log.group(2)
                        inf_time = match_log.group(3)
                        post_time = match_log.group(4)

                        writer_bench.writerow([board_ts, pre_time, inf_time, post_time])
                        f_bench.flush()

                    if "FIN DU BENCHMARK" in line:
                        print(f"\n--- Benchmark de {model_name} terminé ---")
                        break

                except KeyboardInterrupt:
                    print("\nArrêt manuel.")
                    break

    except IOError as e:
        print(f"Erreur fichier: {e}")
    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()


if __name__ == "__main__":
    parse_serial_data()
