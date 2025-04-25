import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

# Provided dataset sizes.
DATASET_SIZES = {
    'cifar100': 5000,
    'sun397': 8534,
    'imagenet397': 8471,
    'imagenet397-balanced': 15086,
    'svhn_cropped': 7320,
    'svhn_cropped_balanced': 5000,
    'cassava': 5656,
    'patch_camelyon': 5897
}

# ----- Data Loading Functions -----

def load_noise_csv(csv_path):
    """Load the noise multiplier CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    return df

def load_aggregated_data(json_path):
    """Load aggregated experiment data from JSON and return a DataFrame."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    # Convert key columns to numeric.
    for col in ['test_metrics.MulticlassAccuracy', 'hyperparameters.target_epsilon', 'hyperparameters.batch_size']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['test_metrics.MulticlassAccuracy', 'hyperparameters.target_epsilon'], inplace=True)
    if 'configuration.dataset_name' in df.columns:
        df['dataset'] = df['configuration.dataset_name'].apply(
            lambda x: x.split('/', 1)[-1] if isinstance(x, str) and '/' in x else x
        )
    else:
        df['dataset'] = 'unknown'
    return df

def get_best_config_by_batch_size(df):
    """
    For each unique (dataset, hyperparameters.target_epsilon, hyperparameters.batch_size)
    combination, select the row with the highest MulticlassAccuracy.
    """
    grouped = df.groupby(['dataset', 'hyperparameters.target_epsilon', 'hyperparameters.batch_size'], as_index=False)
    best_configs = grouped.apply(lambda g: g.loc[g['test_metrics.MulticlassAccuracy'].idxmax()])
    best_configs.reset_index(drop=True, inplace=True)
    return best_configs

# ----- Plotting Function -----

def plot_noise_curves_with_all_best_markers(noise_csv, aggregated_json, epoch_to_plot=40, delta=1e-5, output_dir='plots'):
    """
    For each dataset, load the noise CSV and aggregated experiment data,
    then for each target epsilon in the best configuration data (grouped by batch size),
    plot one figure showing the cumulative noise curves (σ√T vs. q) from the CSV and overlay markers.
    
    For each dataset and each target epsilon:
      - All best configurations for that (dataset, ε) group (by batch size) are marked.
      - The marker for the overall best (highest accuracy) is drawn in black, the others in red.
      - Marker annotations show the accuracy (to two decimals) and are rotated by 45°.
    """
    # Load data.
    df_noise = load_noise_csv(noise_csv)
    df_exp = load_aggregated_data(aggregated_json)
    
    # Get best config per dataset, target epsilon, and batch size.
    df_best = get_best_config_by_batch_size(df_exp)
    
    # For plotting noise curves, we use the noise CSV which now includes a "Dataset" column.
    datasets = df_noise['Dataset'].unique() if 'Dataset' in df_noise.columns else df_best['dataset'].unique()
    
    for ds in datasets:
        ds_size = DATASET_SIZES.get(ds, None)
        if ds_size is None:
            print(f"Dataset size for {ds} not found; skipping.")
            continue
        
        # Filter noise CSV and best config data for this dataset.
        df_noise_ds = df_noise[df_noise['Dataset'] == ds] if 'Dataset' in df_noise.columns else None
        if df_noise_ds is None or df_noise_ds.empty:
            print(f"No noise data for dataset {ds}; skipping.")
            continue
        df_noise_ep = df_noise_ds[df_noise_ds['Epoch'] == epoch_to_plot]
        if df_noise_ep.empty:
            print(f"No noise data for epoch {epoch_to_plot} for dataset {ds}; skipping.")
            continue
        
        df_best_ds = df_best[df_best['dataset'] == ds]
        if df_best_ds.empty:
            print(f"No best configuration data for dataset {ds}; skipping markers.")
        
        # Create one plot per dataset.
        plt.figure(figsize=(10, 6))
        # Plot cumulative noise curves for all epsilons from the CSV._d
        epsilons_all = sorted(df_noise_ep['Epsilon'].unique())
        for eps in epsilons_all:
            df_curve = df_noise_ep[df_noise_ep['Epsilon'] == eps].sort_values("SampleRate")
            plt.plot(df_curve["SampleRate"], df_curve["CumulativeNoise"],
                     marker='o', label=f'ε={eps}')

        # Overlay markers for best configurations per dataset/ε (grouped by batch size).
        # Group best data by target_epsilon.
        for eps, group in df_best_ds.groupby('hyperparameters.target_epsilon'):
            # Make sure that this epsilon exists in the noise CSV.
            if eps not in epsilons_all:
                continue
            # Determine overall best accuracy for this epsilon group.
            best_idx = group['test_metrics.MulticlassAccuracy'].idxmax()
            best_acc_overall = group.loc[best_idx, 'test_metrics.MulticlassAccuracy']

            # For each row in the group (each unique batch size)
            for _, row in group.iterrows():
                best_bs = row['hyperparameters.batch_size']
                best_acc = row['test_metrics.MulticlassAccuracy']
                # Compute sample rate for this configuration.
                best_q = 1.0 if best_bs == -1 else best_bs / ds_size
                # Find the nearest point in the noise CSV for this epsilon.
                df_noise_eps = df_noise_ep[df_noise_ep['Epsilon'] == eps]
                if df_noise_eps.empty:
                    continue
                closest_idx = (df_noise_eps["SampleRate"] - best_q).abs().idxmin()
                marker_x = df_noise_eps.loc[closest_idx, "SampleRate"]
                marker_y = df_noise_eps.loc[closest_idx, "CumulativeNoise"]
                # If this row has the overall best accuracy, use black; otherwise, red.
                marker_color = 'black' if math.isclose(best_acc, best_acc_overall, rel_tol=1e-6) else 'red'
                plt.scatter(marker_x, marker_y, color=marker_color, s=100, marker='x')
                plt.text(marker_x, marker_y, f' acc={best_acc:.2f}', color=marker_color,
                         fontsize=10, rotation=45, verticalalignment='bottom')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Subsampling Rate (q)')
        plt.ylabel('Cumulative Noise (σ√T)')
        plt.title(f'Cumulative Noise vs q for {ds} (Epoch={epoch_to_plot})')
        plt.legend(loc='best')
        plt.grid(True, which="both", linestyle="--")
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"noise_curve_{ds}_epoch{epoch_to_plot}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved noise plot for {ds} to {filename}")

def main():
    noise_csv = "noise_multiplier_data_prod.csv"
    aggregated_json = "aggregated_data.json"
    epoch_to_plot = 40
    plot_dir = 'plots-prod/all'
    plot_noise_curves_with_all_best_markers(noise_csv, aggregated_json, epoch_to_plot=epoch_to_plot, output_dir=plot_dir)

if __name__ == "__main__":
    main()

