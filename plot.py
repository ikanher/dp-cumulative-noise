#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot cumulative‑noise curves and mark the best‑accuracy run.
Scans JSON and CSV files automatically and errors out when
an experiment epoch has no matching noise‑multiplier curves.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List

from tueplots import bundles

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='whitegrid')

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

DATASET_SIZES: Dict[str, int] = {
    'cifar100': 5000,
    'sun397': 8534,
    'imagenet397': 8471,
    'imagenet397-balanced': 15086,
    'svhn_cropped': 7320,
    'svhn_cropped_balanced': 5000,
    'cassava': 5656,
    'patch_camelyon': 5897,
}

def _discover_files(pattern: str) -> List[Path]:
    return sorted(Path('.').glob(pattern))

def _index_csv_epochs(csv_paths: List[Path]) -> Dict[int, Path]:
    """Return {epoch: csv_path} – raise if duplicate epoch appears."""
    epoch_map: Dict[int, Path] = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, usecols=['Epoch'])  # cheap, single column
        for ep in df['Epoch'].unique():
            if ep in epoch_map:
                raise RuntimeError(f'Duplicate epoch {ep} in {csv_path} and {epoch_map[ep]}')
            epoch_map[ep] = csv_path
    return epoch_map

def _load_json_records(json_path: Path) -> pd.DataFrame:
    # Load raw JSON list
    with open(json_path, 'r') as fh:
        data = json.load(fh)

    # Drop any ConfusionMatrix from test_metrics to save memory
    for rec in data:
        tm = rec.get('test_metrics')
        if isinstance(tm, dict) and 'ConfusionMatrix' in tm:
            tm.pop('ConfusionMatrix', None)
            tm.pop('MulticlassAccuracyPerClass', None)

    # Now normalize into a flat DataFrame
    df = pd.json_normalize(data)

    # Convert key columns to numeric
    num_cols = [
        'test_metrics.MulticlassAccuracy',
        'hyperparameters.target_epsilon',
        'hyperparameters.batch_size',
        'hyperparameters.epochs',
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['test_metrics.MulticlassAccuracy',
                      'hyperparameters.target_epsilon'],
              inplace=True)

    # Extract dataset name
    if 'configuration.dataset_name' in df.columns:
        df['dataset'] = df['configuration.dataset_name'] \
                          .str.split('/', n=1).str[-1]
    else:
        df['dataset'] = 'unknown'

    return df

def _best_per_bs(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(['dataset',
                    'hyperparameters.target_epsilon',
                    'hyperparameters.batch_size'],
                   as_index=False)
    return g.apply(lambda g_: g_.loc[g_['test_metrics.MulticlassAccuracy'].idxmax()]).reset_index(drop=True)

# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #

def _plot_curves(csv_path: Path, df_best: pd.DataFrame,
                 epoch: int, out_dir: Path) -> None:
    df_noise = pd.read_csv(csv_path)
    df_noise_ep = df_noise[df_noise['Epoch'] == epoch]
    if df_noise_ep.empty:
        # Should not happen – we guaranteed epoch->csv earlier – but check anyway.
        raise ValueError(f'CSV {csv_path} does not contain epoch {epoch}')

    datasets = df_best['dataset'].unique()
    for ds in datasets:
        size = DATASET_SIZES.get(ds)
        if size is None:
            print(f'Skip {ds}: unknown dataset size')
            continue

        df_noise_ds = df_noise_ep[df_noise_ep['Dataset'] == ds]
        if df_noise_ds.empty:
            print(f'Skip {ds}: no noise data in {csv_path}')
            continue

        plt.figure(figsize=(10, 6))

        epsilons = sorted(df_noise_ds['Epsilon'].unique())
        for eps in epsilons:
            curve = (df_noise_ds[df_noise_ds['Epsilon'] == eps]
                     .sort_values('SampleRate'))
            plt.plot(curve['SampleRate'], curve['CumulativeNoise'],
                     marker='o', label=f'ε={eps}')

        # Markers for best config
        for eps, grp in df_best[df_best['dataset'] == ds].groupby('hyperparameters.target_epsilon'):
            if eps not in epsilons:
                continue
            best = grp.loc[grp['test_metrics.MulticlassAccuracy'].idxmax()]
            bs = best['hyperparameters.batch_size']
            acc = best['test_metrics.MulticlassAccuracy']
            q = 1.0 if bs == -1 else bs / size
            curve_eps = df_noise_ds[df_noise_ds['Epsilon'] == eps]
            marker_x = curve_eps.iloc[(curve_eps['SampleRate'] - q).abs().argmin()]['SampleRate']
            marker_y = curve_eps.iloc[(curve_eps['SampleRate'] - q).abs().argmin()]['CumulativeNoise']
            plt.scatter(marker_x, marker_y, color='blue', s=150, marker='*')
            plt.text(marker_x, marker_y, f'  acc={acc:.2f}', color='blue',
                     fontsize=12, rotation=45, va='bottom')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Subsampling rate q')
        plt.ylabel(r'Cumulative noise $\sigma\sqrt{T}$')
        plt.title(f'{ds} – epoch {epoch}')
        plt.legend()
        plt.grid(True, which='both', linestyle='--')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'noise_curve_{ds}_epoch{epoch}.png'
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f'Saved {out_file}')

# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    plt.rcParams.update(
        bundles.neurips2024(family="Dejavu Serif", ncols=2, nrows=1, usetex=False)
    )

    json_paths = _discover_files('aggregated_data_epochs20_tiny_cifar100_and_svhn.json')
    if not json_paths:
        raise FileNotFoundError('No aggregated_data*.json files found')

    csv_paths = _discover_files('*noise_multiplier*.csv')
    if not csv_paths:
        raise FileNotFoundError('No *noise_multiplier*.csv files found')

    epoch_to_csv = _index_csv_epochs(csv_paths)

    combined_records: List[pd.DataFrame] = []
    for jp in json_paths:
        combined_records.append(_load_json_records(jp))
    df_all = pd.concat(combined_records, ignore_index=True)

    # ── loop over each unique epoch ───────────────────────────────────────────
    for epoch in sorted(df_all['hyperparameters.epochs'].dropna().unique().astype(int)):
        if epoch not in epoch_to_csv:
            raise ValueError(f'No CSV file contains epoch {epoch} (required by a JSON record)')
        csv_path = epoch_to_csv[epoch]
        df_epoch = df_all[df_all['hyperparameters.epochs'] == epoch]
        df_best = _best_per_bs(df_epoch)
        _plot_curves(csv_path, df_best, epoch, Path('plots/best'))

if __name__ == '__main__':
    main()
