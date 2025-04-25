import math
import numpy as np
import pandas as pd
import sys
from opacus.accountants.utils import get_noise_multiplier as opacus_get_noise_multiplier_impl

# Cache for noise multipliers.
sigma_cache = {}

def compute_sigma(target_epsilon, sample_rate, epochs, delta=1e-5, accountant='prv', epsilon_tolerance=0.01):
    """
    Computes the noise multiplier σ needed to achieve a privacy budget of
    (target_epsilon, target_delta) and caches the result.
    """
    key = (target_epsilon, sample_rate, epochs, delta, accountant, epsilon_tolerance)
    if key in sigma_cache:
        return sigma_cache[key]
    sigma_val = opacus_get_noise_multiplier_impl(
        target_epsilon=target_epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=epochs,
        accountant=accountant,
        epsilon_tolerance=epsilon_tolerance
    )
    sigma_cache[key] = sigma_val
    return sigma_val

def main():
    # Provided dataset sizes.
    DATASET_SIZES = {
        'cifar100': 5000,
        'sun397': 8534,
        'imagenet397': 8471,
        'imagenet397-balanced': 15086,
        'svhn_cropped': 7320,
#        'svhn_cropped_balanced': 5000,
        'cassava': 5656,
        'patch_camelyon': 5897
    }

    # Fix epochs to 40.
    epoch_list = [40]
    # epsilons from 2^-2 (0.25) to 16 (2^4)
    epsilons = [0.5, 1, 2, 4, 8]

    data = []  # List to store computed data for each configuration.

    for ds, ds_size in DATASET_SIZES.items():
        print(f'------------------- {ds}', file=sys.stderr)
        # For each dataset, compute batch sizes as powers of 2 up to ds_size.
        batch_sizes = [2**x for x in range(0, int(np.log2(ds_size)) + 1)] + [ds_size]
        for ep in epoch_list:
            for epsilon in epsilons:
                for bs in batch_sizes:
                    sample_rate = bs / ds_size
                    steps = ep * (ds_size / bs)
                    sigma = compute_sigma(epsilon, sample_rate, ep, delta=1e-5, epsilon_tolerance=0.01)
                    sigma_over_q = sigma / sample_rate if sample_rate > 0 else float('inf')
                    cumulative_noise = sigma * math.sqrt(steps)
                    total_noise_variance = steps * sigma**2
                    cumulative_effective_noise = cumulative_noise / sample_rate if sample_rate > 0 else float('inf')
                    data.append({
                        "Dataset": ds,
                        "DatasetSize": ds_size,
                        "Epoch": ep,
                        "Epsilon": epsilon,
                        "BatchSize": bs,
                        "SampleRate": sample_rate,
                        "Steps": steps,
                        "Sigma": sigma,
                        "Sigma_over_q": sigma_over_q,
                        "CumulativeNoise": cumulative_noise,
                        "TotalNoiseVariance": total_noise_variance,
                        "CumulativeEffectiveNoise": cumulative_effective_noise
                    })
                    print(f"Dataset: {ds}, ε={epsilon:6.3f}, epochs={ep}, bs={bs}, q={sample_rate:.4f}, σ={sigma:8.6f}", file=sys.stderr)

    df = pd.DataFrame(data)
    output_filename = "opacus_noise_multiplier_data.csv"
    df.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}", file=sys.stderr)

if __name__ == '__main__':
    main()
