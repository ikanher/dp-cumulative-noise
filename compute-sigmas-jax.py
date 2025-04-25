import math
import numpy as np
import pandas as pd
from dp_accounting.pld import pld_privacy_accountant
from dp_accounting import dp_event

DEBUG = False
MAX_SIGMA = 1e6

# Cache for noise multipliers.
sigma_cache = {}

def get_noise_multiplier_jax(*, target_epsilon, target_delta, sample_rate, epochs=None, steps=None, epsilon_tolerance=0.01):
    """
    Computes the noise multiplier σ needed to achieve a privacy budget of
    (target_epsilon, target_delta) using dp_accounting's PLDAccountant with the
    ADD_OR_REMOVE_ONE relation. Composition uses a PoissonSampledDpEvent to
    incorporate the sample_rate.

    Exactly one of `epochs` or `steps` must be provided.
    """
    if (steps is None) == (epochs is None):
        raise ValueError("Provide exactly one of 'epochs' or 'steps'.")
    if steps is None:
        steps = int(epochs / sample_rate)

    sigma_low = 0.0
    sigma_high = 10.0

    # Expansion phase.
    while True:
        accountant = pld_privacy_accountant.PLDAccountant(
            pld_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE
        )
        event = dp_event.PoissonSampledDpEvent(
            sample_rate, dp_event.GaussianDpEvent(noise_multiplier=sigma_high)
        )
        accountant.compose(event, steps)
        eps_high = accountant.get_epsilon(target_delta)
        if DEBUG:
            print(f"[JAX Expansion] sigma_high: {sigma_high:.6f}, eps_high: {eps_high:.6f}")
        sigma_high *= 2.0
        if eps_high <= target_epsilon:
            break
        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    # Binary search phase.
    while target_epsilon - eps_high > epsilon_tolerance:
        sigma_mid = (sigma_low + sigma_high) / 2.0
        accountant = pld_privacy_accountant.PLDAccountant(
            pld_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE
        )
        event = dp_event.PoissonSampledDpEvent(
            sample_rate, dp_event.GaussianDpEvent(noise_multiplier=sigma_mid)
        )
        accountant.compose(event, steps)
        eps_mid = accountant.get_epsilon(target_delta)
        if DEBUG:
            print(f"[JAX Binary Search] sigma_mid: {sigma_mid:.6f}, eps_mid: {eps_mid:.6f}, sigma_low: {sigma_low:.6f}, sigma_high: {sigma_high:.6f}")
        if eps_mid < target_epsilon:
            sigma_high = sigma_mid
            eps_high = eps_mid
        else:
            sigma_low = sigma_mid

    return sigma_high

def compute_sigma(target_epsilon, sample_rate, epochs, delta=1e-5, epsilon_tolerance=0.01):
    """
    Caches and returns the noise multiplier computed by get_noise_multiplier_jax.
    The cache key is based on (target_epsilon, sample_rate, epochs, delta, epsilon_tolerance).
    """
    key = (target_epsilon, sample_rate, epochs, delta, epsilon_tolerance)
    if key in sigma_cache:
        return sigma_cache[key]
    sigma_val = get_noise_multiplier_jax(
        target_epsilon=target_epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=epochs,
        epsilon_tolerance=epsilon_tolerance
    )
    sigma_cache[key] = sigma_val
    return sigma_val

def main():
    dataset_size = 2**13
    max_epsilon = 16
    # epsilons from 2^-2 (0.25) to 16 (2^4)
    epsilons = [2**x for x in range(-2, int(np.log2(max_epsilon)) + 1)]
    batch_sizes = [2**x for x in range(0, int(np.log2(dataset_size)) + 1)]
    # Modify epoch_list as desired.
    #epoch_list = [1, 40, 1000]
    epoch_list = [10000]
    epsilon_tolerance = 0.01
    delta = 1e-5

    # List to collect data.
    data = []

    for ep in epoch_list:
        for epsilon in epsilons:
            for bs in batch_sizes:
                sample_rate = bs / dataset_size
                steps = ep * (dataset_size / bs)
                sigma = compute_sigma(epsilon, sample_rate, ep, delta=delta, epsilon_tolerance=epsilon_tolerance)
                sigma_over_q = sigma / sample_rate if sample_rate > 0 else float('inf')
                cumulative_noise = sigma * math.sqrt(steps)
                cumulative_effective_noise = cumulative_noise / sample_rate if sample_rate > 0 else float('inf')
                data.append({
                    "Epoch": ep,
                    "Epsilon": epsilon,
                    "BatchSize": bs,
                    "SampleRate": sample_rate,
                    "Steps": steps,
                    "Sigma": sigma,
                    "Sigma_over_q": sigma_over_q,
                    "CumulativeNoise": cumulative_noise,
                    "CumulativeEffectiveNoise": cumulative_effective_noise
                })
                print(f"ε={epsilon:6.3f}, epochs={ep}, bs={bs}, q={sample_rate:.4f}, σ={sigma:8.6f}")

    df = pd.DataFrame(data)
    df.to_csv("noise_multiplier_data.csv", index=False)
    print("Data saved to noise_multiplier_data.csv")

if __name__ == '__main__':
    main()
