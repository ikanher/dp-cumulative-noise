#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot cumulative-noise curves and mark runs within 3% of the best-accuracy.
Scans JSON and CSV files automatically and errors out when
an experiment epoch has no matching noise-multiplier curves.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tueplots import bundles, figsizes, fontsizes
from matplotlib.colors import ListedColormap

sns.set(style='whitegrid')

# Full datasets
DATASET_SIZES = {
    'sun397': 87002,
    'cifar100': 50000,
    'svhn_cropped': 73257,
}


def _discover_files(pattern: str) -> List[Path]:
    return sorted(Path('.').glob(pattern))


def _index_csv_epochs(csv_paths: List[Path]) -> Dict[int, Path]:
    epoch_map = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, usecols=['Epoch'])
        for ep in df['Epoch'].unique():
            if ep in epoch_map:
                raise RuntimeError(
                    f'Duplicate epoch {ep} in {csv_path} and {epoch_map[ep]}'
                )
            epoch_map[ep] = csv_path
    return epoch_map


def _load_json_records(json_path: Path) -> pd.DataFrame:
    with open(json_path, 'r') as fh:
        data = json.load(fh)
    for rec in data:
        tm = rec.get('test_metrics', {})
        tm.pop('ConfusionMatrix', None)
        tm.pop('MulticlassAccuracyPerClass', None)
    df = pd.json_normalize(data)
    num_cols = [
        'test_metrics.MulticlassAccuracy',
        'hyperparameters.target_epsilon',
        'hyperparameters.batch_size',
        'hyperparameters.epochs',
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(
        subset=['test_metrics.MulticlassAccuracy', 'hyperparameters.target_epsilon'],
        inplace=True,
    )
    if 'configuration.dataset_name' in df.columns:
        df['dataset'] = df['configuration.dataset_name'].str.split('/', n=1).str[-1]
    else:
        df['dataset'] = 'unknown'
    return df


def _best_per_bs(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(
        ['dataset', 'hyperparameters.target_epsilon', 'hyperparameters.batch_size'],
        as_index=False,
    )
    return g.apply(
        lambda grp: grp.loc[grp['test_metrics.MulticlassAccuracy'].idxmax()]
    ).reset_index(drop=True)


def _plot_curves(
    csv_path: Path,
    df_best: pd.DataFrame,
    epoch: int,
    out_dir: Path,
    ymax: float,
    ymin: float,
) -> None:
    df_noise = pd.read_csv(csv_path)
    df_noise_ep = df_noise[df_noise['Epoch'] == epoch]

    if df_noise_ep.empty:
        raise ValueError(f'CSV {csv_path} does not contain epoch {epoch}')

    # Remap batch_size = -1 to dataset size
    df_best['hyperparameters.batch_size'] = df_best.apply(
        lambda row: DATASET_SIZES.get(row['dataset'], -1) if row['hyperparameters.batch_size'] == -1 else row['hyperparameters.batch_size'],
        axis=1,
    )

    all_noises = []

    for ds in df_best['dataset'].unique():
        size = DATASET_SIZES.get(ds)
        if size is None:
            print(f'Skip {ds}: unknown dataset size')
            continue

        df_noise_ds = df_noise_ep[df_noise_ep['Dataset'] == ds]
        if df_noise_ds.empty:
            print(f'Skip {ds}: no noise data in {csv_path}')
            continue

        epsilons = sorted(df_noise_ds['Epsilon'].unique())
        all_noises.append(df_noise_ds['CumulativeNoise'].max())

        plt.figure()
        ax = plt.gca()

        full_palette = sns.color_palette('Blues', as_cmap=True)
        palette = ListedColormap(full_palette(np.linspace(0.95, 0.6, len(epsilons))))

        eps_to_palette = {8: 0, 4: 1, 2: 2, 1: 3}

        for idx, eps in enumerate(epsilons):
            curve = df_noise_ds[df_noise_ds['Epsilon'] == eps].sort_values('SampleRate')
            bs = curve['BatchSize']
            noise = curve['CumulativeNoise']
            ax.plot(
                bs,
                noise,
                alpha=0.9,
                label=f'ε={eps}',
                color=palette(eps_to_palette[eps]),
            )

        # mark all runs for current ds
        for eps, grp in df_best[df_best['dataset'] == ds].groupby('hyperparameters.target_epsilon'):
            if eps not in epsilons:
                continue

            curve_eps = df_noise_ds[df_noise_ds['Epsilon'] == eps]
            grp_sorted = grp.sort_values('test_metrics.MulticlassAccuracy', ascending=False)

            for i, (_, row) in enumerate(grp_sorted.iterrows()):
                bs = row['hyperparameters.batch_size']
                acc = row['test_metrics.MulticlassAccuracy']
                mx = bs

                samp_idx = (curve_eps['SampleRate'] - (bs / size)).abs().idxmin()
                my = curve_eps.loc[samp_idx, 'CumulativeNoise']

                ax.scatter(
                    mx, my,
                    marker='o',
                    s=16,
                    zorder=5,
                    color=palette(eps_to_palette[eps]),
                )

                if i < 3:
                    ax.text(
                        mx, my,
                        f'{acc*100:.2f}%',
                        color='red',
                        rotation=45,
                        va='bottom',
                        zorder=6,
                        fontsize=9,
                    )

        ax.legend(loc='lower left', frameon=True, borderpad=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(bottom=ymin-5, top=ymax+5)

        ax.set_xlabel('Batch size')
        ax.set_ylabel(r'Cumulative noise $\sigma\sqrt{T}$')
        ax.set_title(f'{ds} – epoch {epoch}')

        # Actual batch sizes as xticks
        unique_bs = sorted(b for b in df_noise_ds['BatchSize'].unique() if b != 192 and b != -1)[::-2]

        ax.set_xticks(unique_bs)
        ax.set_xticklabels([str(int(b)) for b in unique_bs], rotation=45)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'noise_curve_{ds}_epoch{epoch}.png'
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f'Saved {out_file}')

    # Return max noise value for global scaling
    return max(all_noises) if all_noises else None


def main() -> None:
    plt.rcParams.update(
        bundles.neurips2024(family="Dejavu Serif", ncols=2, nrows=1, usetex=False)
    )

    plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=1))
    plt.rcParams.update(fontsizes.neurips2024())

    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 10,
        'font.size': 10,  # base font size
    })

    files = [
        #        'aggregated_data_epochs20_tiny.json',
        #        'aggregated_data_epochs40_tiny.json',
        #        'aggregated_data_epochs80_tiny.json',
        #'aggregated_data_vit_tiny_full_data.json',
        'aggregated_data_vit_base_full_data.json',
    ]

    for f in files:
        print(f'Processing: {f}')
        json_paths = _discover_files(f)
        if not json_paths:
            raise FileNotFoundError('No aggregated_data*.json files found')

        csv_paths = _discover_files('full-data-csvs/*noise_multiplier*.csv')
        if not csv_paths:
            raise FileNotFoundError('No *noise_multiplier*.csv files found')

        epoch_to_csv = _index_csv_epochs(csv_paths)

        combined_records = []
        for jp in json_paths:
            combined_records.append(_load_json_records(jp))

        df_all = pd.concat(combined_records, ignore_index=True)

        # First pass to collect all ymax values
        all_ymax = []
        all_ymin = []
        for epoch in sorted(df_all['hyperparameters.epochs'].dropna().unique().astype(int)):
            if epoch not in epoch_to_csv:
                raise ValueError(f'No CSV file contains epoch {epoch}')
            csv_path = epoch_to_csv[epoch]
            df_epoch = df_all[df_all['hyperparameters.epochs'] == epoch]
            df_best = _best_per_bs(df_epoch)

            df_noise = pd.read_csv(csv_path)
            df_noise_ep = df_noise[df_noise['Epoch'] == epoch]
            all_ymax.append(df_noise_ep['CumulativeNoise'].max())
            all_ymin.append(df_noise_ep['CumulativeNoise'].min())

        global_ymax = max(all_ymax)
        global_ymin = max(all_ymin)

        # Second pass to actually plot with fixed ymax
        for epoch in sorted(df_all['hyperparameters.epochs'].dropna().unique().astype(int)):
            csv_path = epoch_to_csv[epoch]
            df_epoch = df_all[df_all['hyperparameters.epochs'] == epoch]
            df_best = _best_per_bs(df_epoch)
            _plot_curves(
                csv_path,
                df_best,
                epoch,
                Path('plots/close-to-best'),
                ymax=global_ymax,
                ymin=global_ymin,
            )


if __name__ == '__main__':
    main()
