import typer
from opacus.accountants.utils import get_noise_multiplier


def compute_sigma(target_epsilon, sample_rate, epochs, delta=1e-5, accountant='prv', epsilon_tolerance=0.01):

    sigma = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=epochs,
        accountant=accountant,
        epsilon_tolerance=epsilon_tolerance
    )

    return sigma


def main(epsilon: int, epochs: int, dataset_size: int, batch_size: int):

    sample_rate = batch_size/dataset_size
    sigma = compute_sigma(epsilon, sample_rate, epochs)

    print(f'Sigma is: {sigma}')


if __name__ == '__main__':
    typer.run(main)
