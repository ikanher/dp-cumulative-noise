#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot cumulative-noise curves and mark the best-accuracy run.
Scans all aggregated_data*.json and *noise_multiplier*.csv files,
drops large ConfusionMatrix fields, and errors if any JSON epoch
has no matching CSV curves.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from tueplots import bundles
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style='whitegrid')

# --------------------------------------------------------------------------- #
# Constants                                                                   #
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

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _discover_files(pattern: str) -> List[Path]:
    return sorted(Path('.').glob(pattern))

def _index_csv_epochs(csv_paths: List[Path]) -> Dict[int, Path]:
    """Build {epoch: csv_path}, error on duplicates."""
    epoch_map: Dict[int, Path] = {}
    for p in csv_paths:
        for ep in pd.read_csv(p, usecols=['Epoch'])['Epoch'].unique():
            if ep in epoch_map:
                raise RuntimeError(f'Duplicate epoch {ep} in {p} and {epoch_map[ep]}')
            epoch_map[ep] = p
    return epoch_map

def _load_json_records(json_path: Path) -> pd.DataFrame:
    """Load one JSON, strip big fields, normalize, convert types."""
    with open(json_path, 'r') as fh:
        data = json.load(fh)
    for rec in data:
        tm = rec.get('test_metrics', {})
        if isinstance(tm, dict):
            tm.pop('ConfusionMatrix', None)
            tm.pop('MulticlassAccuracyPerClass', None)
    df = pd.json_normalize(data)
    # numeric cols
    for c in [
        'test_metrics.MulticlassAccuracy',
        'hyperparameters.target_epsilon',
        'hyperparameters.batch_size',
        'hyperparameters.epochs',
    ]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(
        subset=[
            'test_metrics.MulticlassAccuracy',
            'hyperparameters.target_epsilon',
        ],
        inplace=True
    )

    # Extract dataset name
    if 'configuration.dataset_name' in df.columns:
        df['dataset'] = df['configuration.dataset_name'] \
                          .str.split('/', n=1).str[-1]
    else:
        df['dataset'] = 'unknown'

    return df

def _best_per_bs(df: pd.DataFrame) -> pd.DataFrame:
    """Select highest-accuracy run per (dataset, ε, batch_size)."""
    g = df.groupby(
        ['dataset', 'hyperparameters.target_epsilon', 'hyperparameters.batch_size'],
        as_index=False
    )
    best = g.apply(lambda grp: grp.loc[grp['test_metrics.MulticlassAccuracy'].idxmax()])
    return best.reset_index(drop=True)

# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #

def _plot_curves(
    csv_path: Path,
    df_best: pd.DataFrame,
    epoch: int,
    out_dir: Path
) -> None:
    df_noise = pd.read_csv(csv_path)
    df_ep = df_noise[df_noise['Epoch'] == epoch]
    if df_ep.empty:
        raise ValueError(f'CSV {csv_path} missing epoch {epoch}')

    for ds in df_best['dataset'].unique():
        size = DATASET_SIZES.get(ds)
        if size is None:
            print(f'Skip {ds}: unknown size')
            continue

        df_ds = df_ep[df_ep['Dataset'] == ds]
        if df_ds.empty:
            print(f'Skip {ds}: no noise data')
            continue

        plt.figure(figsize=(10, 6))
        for eps in sorted(df_ds['Epsilon'].unique()):
            curve = df_ds[df_ds['Epsilon'] == eps].sort_values('SampleRate')
            plt.plot(curve['SampleRate'], curve['CumulativeNoise'],
                     marker='o', label=f'ε={eps}')

        for eps, grp in df_best[df_best['dataset'] == ds].groupby('hyperparameters.target_epsilon'):
            if eps not in df_ds['Epsilon'].unique():
                continue
            best = grp.loc[grp['test_metrics.MulticlassAccuracy'].idxmax()]
            bs, acc = best['hyperparameters.batch_size'], best['test_metrics.MulticlassAccuracy']
            q = 1.0 if bs == -1 else bs / size
            subset = df_ds[df_ds['Epsilon'] == eps]
            idx = (subset['SampleRate'] - q).abs().idxmin()
            x, y = subset.loc[idx, ['SampleRate', 'CumulativeNoise']]
            plt.scatter(x, y, marker='*', s=150, color='blue')
            plt.text(x, y, f'  acc={acc:.2f}', rotation=45,
                     va='bottom', color='blue')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Subsampling rate q')
        plt.ylabel(r'Cumulative noise $\sigma\sqrt{T}$')
        plt.title(f'{ds} – epoch {epoch}')
        plt.legend(loc='best')
        plt.grid(True, which='both', linestyle='--')

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f'noise_curve_{ds}_epoch{epoch}.png'
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f'Saved {path}')

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    plt.rcParams.update(
        bundles.neurips2024(family="Dejavu Serif", ncols=2, nrows=1, usetex=False)
    )

    json_paths = _discover_files('aggregated_data_epochs80_tiny.json')
    if not json_paths:
        raise FileNotFoundError('No aggregated_data*.json files found')

    csv_paths = _discover_files('*noise_multiplier*.csv')
    if not csv_paths:
        raise FileNotFoundError('No *noise_multiplier*.csv files found')

    epoch_to_csv = _index_csv_epochs(csv_paths)

    # load and combine all JSONs
    combined: List[pd.DataFrame] = [_load_json_records(jp) for jp in json_paths]
    df_all = pd.concat(combined, ignore_index=True)

    for epoch in sorted(df_all['hyperparameters.epochs'].dropna().unique().astype(int)):
        if epoch not in epoch_to_csv:
            raise ValueError(f'No CSV contains epoch {epoch}')
        df_epoch = df_all[df_all['hyperparameters.epochs'] == epoch]
        df_best  = _best_per_bs(df_epoch)
        _plot_curves(epoch_to_csv[epoch], df_best, epoch, Path('plots-all-makers-auto'))

if __name__ == '__main__':
    main()
