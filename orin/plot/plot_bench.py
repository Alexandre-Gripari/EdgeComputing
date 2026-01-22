import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def load_data_and_baseline(filepath):
    df = pd.read_csv(filepath)
    
    baseline_row = df[df['Model'] == 'Baseline']
    if not baseline_row.empty:
        baseline_swap_pct = float(baseline_row['Swap Usage (%)'].iloc[0])
    else:
        baseline_swap_pct = 0.0

    df_clean = df[df['Model'] != 'Baseline'].copy()
    
    cols_to_numeric = ['mAP50', 'Inference Time (ms)', 'Preprocess Time (ms)', 
                       'Postprocess Time (ms)', 'Power (mW)', 'File Size (MB)', 
                       'CPU Usage (%)', 'GPU Usage (%)', 'Swap Usage (%)']
    
    for col in cols_to_numeric:
        df_clean[col] = pd.to_numeric(df_clean[col])

    df_clean['Total Time'] = (df_clean['Preprocess Time (ms)'] + 
                              df_clean['Inference Time (ms)'] + 
                              df_clean['Postprocess Time (ms)'])
    
    df_clean['Energy_Joules'] = (df_clean['Total Time'] * df_clean['Power (mW)']) / 1_000_000
    
    return df_clean, baseline_swap_pct

def plot_bubble_chart(df, title, filename, color='blue', filter_models=[], highlight_special=False):
    data = df.copy()
    
    for models in filter_models:
        data = data[data['Model'] != models]

    bubble_sizes = data['File Size (MB)'] * 100 
    
    final_colors = []
    legend_elements = []

    if highlight_special:
        fastest_model = data.loc[data['Total Time'].idxmin()]['Model']
        fastest_inference_time = data['Total Time'].min()
        
        for m in data['Model']:
            if m == fastest_model:
                final_colors.append('#00CC96')
            else:
                final_colors.append(color)
        
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Fastest Inference: {fastest_inference_time} ms', markerfacecolor='#00CC96', markersize=10, markeredgecolor='black'))
    else:
        final_colors = color
        
    plt.figure(figsize=(10, 6))

    plt.scatter(data['Energy_Joules'], data['mAP50'], s=bubble_sizes, 
                c=final_colors, alpha=0.7, edgecolors='black')

    for _, row in data.iterrows():
        plt.annotate(row['Model'], (row['Energy_Joules'], row['mAP50']),
                     xytext=(0, 10), textcoords='offset points', 
                     ha='center', fontsize=9, fontweight='bold')
    
    plt.legend(handles=legend_elements, title="Légende", 
               loc='lower right', labelspacing=1.2, borderpad=1.0)

    plt.title(title)
    plt.xlabel('Énergie par inférence (Joules)')
    plt.ylabel('Précision (mAP50)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_cpu_gpu_usage(df, filename, filter_models=[]):
    data = df.copy()
    
    for models in filter_models:
        data = data[data['Model'] != models]
    
    
    x = np.arange(len(data['Model']))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(x - width/2, data['CPU Usage (%)'], width, label='CPU Usage', color='#4e79a7')
    ax1.bar(x + width/2, data['GPU Usage (%)'], width, label='GPU Usage', color='#e15759')

    ax1.set_xlabel('Modèle')
    ax1.set_ylabel('Usage Ressources (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(data['Model'], rotation=15)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(x, data['mAP50'], color='purple', marker='o', 
             linewidth=2, linestyle='--', label='mAP50')
    ax2.set_ylabel('Précision (mAP50)', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    plt.title('CPU & GPU Usage vs Précision')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_time_breakdown(df, filename, filter_models=[]):
    data = df.copy()
    
    for models in filter_models:
        data = data[data['Model'] != models]

    data = data.reset_index(drop=True)

    models = data['Model']
    pre_time = data['Preprocess Time (ms)']
    inf_time = data['Inference Time (ms)']
    post_time = data['Postprocess Time (ms)']
    
    x = range(len(models))
    
    plt.figure(figsize=(12, 7))

    p1 = plt.bar(models, pre_time, label='Pre-process', color='#76b7b2', edgecolor='white', width=0.6)
    p2 = plt.bar(models, inf_time, bottom=pre_time, label='Inference', color='#4e79a7', edgecolor='white', width=0.6)
    p3 = plt.bar(models, post_time, bottom=pre_time + inf_time, label='Post-process', color='#e15759', edgecolor='white', width=0.6)
    
    for i in range(len(data)):
        val_pre = pre_time[i]
        val_inf = inf_time[i]
        val_post = post_time[i]
        total = val_pre + val_inf + val_post
        
        def place_text(value, bottom_start):
            pct = (value / total) * 100
            if pct > 2:
                plt.text(i, bottom_start + (value / 2), f'{pct:.0f}%', 
                         ha='center', va='center', color='white', 
                         fontsize=8, fontweight='bold')

        place_text(val_pre, 0)
        place_text(val_inf, val_pre)
        place_text(val_post, val_pre + val_inf)
    
    total_times = pre_time + inf_time + post_time
    for i, total in enumerate(total_times):
        plt.text(i, total, f'{total:.1f} ms', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    plt.title('Décomposition du Temps de Latence (Pre / Inf / Post)')
    plt.ylabel('Temps (ms)')
    plt.xlabel('Modèle')
    plt.xticks(rotation=20, ha='right')
    
    plt.legend(loc='upper left', title="Étapes")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_swap_usage_stacked(df, filename, baseline_pct, total_swap_gb=16, filter_models=[]):
    data = df.copy()
    plt.figure(figsize=(10, 6))
    
    for models in filter_models:
        data = data[data['Model'] != models]
    
    total_pct = data['Swap Usage (%)']
    
    inference_pct = (total_pct - baseline_pct).clip(lower=0)
    base_heights = np.minimum(total_pct, baseline_pct)
    
    bar_base = plt.bar(data['Model'], base_heights, color='#59a14f', label='Base Système', alpha=0.7)
    
    bar_inf = plt.bar(data['Model'], inference_pct, bottom=base_heights, 
                      color='red', label='Coût Inférence', alpha=0.8)

    for rect in bar_inf:
        height_red = rect.get_height()
        y_top = rect.get_y() + height_red
        
        if y_top > 0:
            added_gb = (height_red / 100) * total_swap_gb
            total_gb = (y_top / 100) * total_swap_gb
            
            label = f"{total_gb:.3f} GB\n(+{added_gb:.3f})"
            
            plt.text(rect.get_x() + rect.get_width()/2, 
                     y_top, 
                     label, 
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

    plt.title(f'Swap : Base Système vs Coût Inférence (Total Dispo: {total_swap_gb} GB)')
    plt.ylabel('Swap Usage (%)')
    plt.xlabel('Modèle')
    plt.xticks(rotation=15)
    
    plt.ylim(0, total_pct.max() * 1.35)
    
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    df_clean, baseline_val = load_data_and_baseline('benchmark/benchmark_results.csv')

    plot_bubble_chart(
        df_clean, 
        title='Précision vs Énergie', 
        filename='graph1_precision_joules.png',
        color='blue',
        filter_models=['yolo11n-8.engine'],
        highlight_special=True
    )
    
    plot_cpu_gpu_usage(df_clean, 'graph2_cpu_gpu.png', filter_models=['yolo11n.pt (pc)'])

    plot_swap_usage_stacked(df_clean, 'graph3_swap.png', baseline_pct=baseline_val, filter_models=['yolo11n.pt (pc)'])

    plot_bubble_chart(
        df_clean, 
        title='Précision vs Énergie', 
        filename='graph4_precision_joules_all.png',
        color='orange',
        highlight_special=True
    )

    plot_time_breakdown(
        df_clean, 
        filename='graph5_time_breakdown.png',
    )
