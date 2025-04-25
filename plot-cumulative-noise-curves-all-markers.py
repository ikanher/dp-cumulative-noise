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

def plot_noise_curves_with_best_markers(noise_csv, aggregated_json, epoch_to_plot=40, delta=1e-5, output_dir='plots'):
    """
    For each dataset, load the noise CSV and aggregated experiment data, then create one figure that
    shows all cumulative noise curves (σ√T vs. q) for the given epoch. Then, for each dataset/target ε pair,
    overlay a marker for the best configuration (i.e. the one with the highest accuracy). The marker is drawn
    in blue (to indicate overall best) and is annotated with the accuracy (formatted to two decimals) with a 45° rotation.
    """
    # Load data.
    df_noise = load_noise_csv(noise_csv)
    df_exp = load_aggregated_data(aggregated_json)
    
    # Get best config per dataset, target epsilon, and batch size.
    df_best = get_best_config_by_batch_size(df_exp)
    
    # Determine the datasets: use the "Dataset" column from CSV if it exists;
    # otherwise use the aggregated data 'dataset' column.
    if 'Dataset' in df_noise.columns:
        datasets = df_noise['Dataset'].unique()
    else:
        datasets = df_best['dataset'].unique()
    
    for ds in datasets:
        ds_size = DATASET_SIZES.get(ds, None)
        if ds_size is None:
            print(f"Dataset size for {ds} not found; skipping.")
            continue
        
        # Filter CSV for this dataset.
        if 'Dataset' in df_noise.columns:
            df_noise_ds = df_noise[df_noise['Dataset'] == ds]
        else:
            # If CSV doesn't have a Dataset column, assume it was computed for a fixed size.
            df_noise_ds = df_noise.copy()
        if df_noise_ds.empty:
            print(f"No noise data for dataset {ds}; skipping.")
            continue
        
        # Filter CSV for the chosen epoch.
        df_noise_ep = df_noise_ds[df_noise_ds['Epoch'] == epoch_to_plot]
        if df_noise_ep.empty:
            print(f"No noise data for epoch {epoch_to_plot} for dataset {ds}; skipping.")
            continue
        
        # Filter best config data for this dataset.
        df_best_ds = df_best[df_best['dataset'] == ds]
        if df_best_ds.empty:
            print(f"No best configuration data for dataset {ds}; skipping markers.")
        
        # Create the plot.
        plt.figure(figsize=(10, 6))
        
        # Plot cumulative noise curves for all epsilons (from CSV).
        epsilons_all = sorted(df_noise_ep['Epsilon'].unique())
        for eps in epsilons_all:
            df_curve = df_noise_ep[df_noise_ep['Epsilon'] == eps].sort_values("SampleRate")
            plt.plot(df_curve["SampleRate"], df_curve["CumulativeNoise"],
                     marker='o', label=f'ε={eps}')
        
        # Overlay markers for best configuration per dataset/ε (only the overall best per epsilon).
        for eps, group in df_best_ds.groupby('hyperparameters.target_epsilon'):
            # Only proceed if the epsilon exists in the noise CSV.
            if eps not in epsilons_all:
                continue
            overall_best = group.loc[group['test_metrics.MulticlassAccuracy'].idxmax()]
            best_bs = overall_best['hyperparameters.batch_size']
            best_acc = overall_best['test_metrics.MulticlassAccuracy']
            # Compute sample rate for the best configuration.
            best_q = 1.0 if best_bs == -1 else best_bs / ds_size
            df_noise_eps = df_noise_ep[df_noise_ep['Epsilon'] == eps]
            if df_noise_eps.empty:
                continue
            # Find the closest point on the noise curve.
            closest_idx = (df_noise_eps["SampleRate"] - best_q).abs().idxmin()
            marker_x = df_noise_eps.loc[closest_idx, "SampleRate"]
            marker_y = df_noise_eps.loc[closest_idx, "CumulativeNoise"]
            # Overlay marker (blue star) and annotation (with 45° rotation).
            plt.scatter(marker_x, marker_y, color='blue', s=150, marker='*')
            plt.text(marker_x, marker_y, f'  acc={best_acc:.2f}', color='blue',
                     fontsize=12, rotation=45, verticalalignment='bottom')
        
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
    plot_dir = "plots-prod/best"
    plot_noise_curves_with_best_markers(noise_csv, aggregated_json, epoch_to_plot=epoch_to_plot, output_dir=plot_dir)

if __name__ == "__main__":
    main()
