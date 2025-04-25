# DP cumulative noise curves

## Files

### `compute-sigmas.py`

Pre-computes the CSV for plotting.

### `compute-sigmas-jax.py`

The same, but handles more epochs than the Opacus one.

NB: This has some weird artifacts in the low ε curves.

### `plot-cumulative-noise-curves-best-marker.py`

Plots the curves from the CSV, adds marker for the best batch size.

### `plot-cumulative-noise-curves-all-markers.py`

Plots the curves from the CSV, adds marker for all the batch sizes.

### `aggregated_data.json` 

The standard aggregated data from the experiments.

### `noise_multiplier_data_prod.csv`

Precomputed CSV for plotting.

NB: Due to the issues mentioned above, this is combined from the JAX _and_ Opacus scripts.
