import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def _calculate_ss_tot(datasets_dir: str) -> dict:
    """
    Calculates the Total Sum of Squares (SS_tot) for each dataset.
    SS_tot is the sum of squared differences between the observed data and its mean.

    Args:
        datasets_dir: The directory containing the dataset CSV files.

    Returns:
        A dictionary mapping dataset names to their SS_tot values.
    """
    ss_tot_map = {}
    
    try:
        dataset_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv') and not 'best_params' in f and not 'summary' in f]
    except FileNotFoundError:
        return None

    for file_name in dataset_files:
        dataset_name = file_name.replace('.csv', '')
        file_path = os.path.join(datasets_dir, file_name)
        
        df = pd.read_csv(file_path)
        
        trace_cols = ['replicate_1', 'replicate_2', 'replicate_3']
        mean_trace = df[trace_cols].mean(axis=1)
        
        overall_mean = mean_trace.mean()
        
        ss_tot = np.sum((mean_trace - overall_mean) ** 2)
        ss_tot_map[dataset_name] = ss_tot
        
    return ss_tot_map


def analyze_and_plot_results():
    """
    Loads aggregated results, calculates R-squared and RMSE for each run,
    and generates plots to visualize optimizer performance using both metrics.
    """
    summary_file = "results_summary.csv"
    data_dir = "data"
    plots_dir = "plots"
    n_datapoints = 43  # Based on 420 minutes sampled every 10 minutes (43 points)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")

    # Load summary results
    try:
        df = pd.read_csv(summary_file)
    except FileNotFoundError:
        print(f"Error: The file '{summary_file}' was not found.")
        return

    # --- Calculate Metrics ---
    
    # Calculate RMSE
    df['rmse'] = np.sqrt(df['min_sse'] / n_datapoints)
    
    # Calculate R-squared
    ss_tot_map = _calculate_ss_tot(data_dir)
    if ss_tot_map is None:
        print(f"Error: Could not find the directory '{data_dir}' to calculate R-squared.")
    else:
        df['ss_tot'] = df['dataset'].map(ss_tot_map)
        df['r_squared'] = 1 - (df['min_sse'] / df['ss_tot'])

    # --- Statistical Analysis ---
    
    # RMSE Stats
    print("\n--- Optimizer Performance (Mean & Std Dev of RMSE) ---")
    rmse_stats = df.groupby(['optimizer', 'dataset'])['rmse'].agg(['mean', 'std']).round(4)
    rmse_stats.columns.names = ['Statistic']
    print(rmse_stats.to_string())
    
    # R-squared Stats
    if 'r_squared' in df.columns:
        print("\n--- Optimizer Performance (Mean & Std Dev of R-squared) ---")
        r2_stats = df.groupby(['optimizer', 'dataset'])['r_squared'].agg(['mean', 'std']).round(4)
        r2_stats.columns.names = ['Statistic']
        print(r2_stats.to_string())

    # --- Visualization ---
    
    df['base_dataset'] = df['dataset'].apply(lambda x: x.split('_')[0])

    # Plot 1: RMSE Plots
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='optimizer', y='rmse', palette='viridis')
    plt.title('Overall RMSE Distribution by Optimizer', fontsize=16)
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "overall_performance_rmse_boxplot.png"))
    plt.close()

    g_rmse = sns.catplot(data=df, x='optimizer', y='rmse', col='base_dataset', kind='box', col_wrap=3, palette='plasma', height=5, aspect=1.2)
    g_rmse.fig.suptitle('Optimizer Performance by Dataset Type (RMSE)', y=1.03, fontsize=18)
    for ax in g_rmse.axes.flat:
        ax.set_ylabel("RMSE")
        ax.tick_params(axis='x', labelrotation=45, labelbottom=True)
    g_rmse.set_titles("Dataset: {col_name}")
    g_rmse.set_axis_labels("Optimizer", "")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(plots_dir, "faceted_performance_rmse_boxplot.png"))
    plt.close()
    print(f"\nSaved RMSE plots to '{plots_dir}'")

    # Plot 2: R-squared Plots
    if 'r_squared' in df.columns:
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df, x='optimizer', y='r_squared', palette='viridis')
        plt.title('Overall R-squared Distribution by Optimizer', fontsize=16)
        plt.xlabel('Optimizer', fontsize=12)
        plt.ylabel('R-squared (R²)', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1.05)
        plt.axhline(1.0, color='r', linestyle='--', label='Perfect Fit (R²=1)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "overall_performance_r2_boxplot.png"))
        plt.close()

        g_r2 = sns.catplot(data=df, x='optimizer', y='r_squared', col='base_dataset', kind='box', col_wrap=3, palette='plasma', height=5, aspect=1.2)
        g_r2.fig.suptitle('Optimizer Performance by Dataset Type (R-squared)', y=1.03, fontsize=18)
        for ax in g_r2.axes.flat:
            ax.set_ylabel("R-squared (R²)")
            ax.tick_params(axis='x', labelrotation=45, labelbottom=True)
        g_r2.set_titles("Dataset: {col_name}")
        g_r2.set(ylim=(0, 1.05))
        g_r2.set_axis_labels("Optimizer", "")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(plots_dir, "faceted_performance_r2_boxplot.png"))
        plt.close()
        print(f"Saved R-squared plots to '{plots_dir}'")


if __name__ == "__main__":
    analyze_and_plot_results()