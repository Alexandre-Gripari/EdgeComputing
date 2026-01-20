import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

csv_files = glob.glob("../benchmark*.csv")

for filename in csv_files:
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Usage (%)')
    
    ax1.plot(df['Time (s)'], df['CPU Usage (%)'], label='CPU Usage')
    ax1.plot(df['Time (s)'], df['GPU Usage (%)'], label='GPU Usage')
    ax1.plot(df['Time (s)'], df['RAM Usage (%)'], label='RAM Usage')
    ax1.plot(df['Time (s)'], df['Swap Usage (%)'], label='Swap Usage')
    
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Power (mW)')
    ax2.plot(df['Time (s)'], df['Power (mW)'], color='black', linestyle='--', label='Power')
    ax2.tick_params(axis='y')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    plt.title(f"Analysis: {filename}")
    
    output_name = os.path.splitext(filename)[0] + ".png"
    plt.savefig(output_name)
    plt.close()
