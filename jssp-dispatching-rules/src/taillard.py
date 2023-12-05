import numpy as np


def generate_taillard(
    n_jobs: int, n_machines: int, seed: int | None = None
) -> np.ndarray:
    """Generate a Taillard instance of `n_jobs` and `n_machines`.

    ---
    Args:
        n_jobs: Number of jobs.
        n_machines: Number of machines.
        seed: Random seed.

    ---
    Returns:
        durations: The processing times of the tasks.
            Shape of [n_jobs, n_machines].
        affectations: The affectations of the tasks.
            Shape of [n_jobs, n_machines].
    """
    rng = np.random.default_rng(seed)

    # Generate the processing times.
    durations = rng.integers(1, 100, size=(n_jobs, n_machines))

    # Generate the machine order.
    affectations = []
    for _ in range(n_jobs):
        # Machines are numbered from 0 to n_machines - 1.
        order = rng.permutation(n_machines)
        affectations.append(order)
    affectations = np.array(affectations)

    return durations, affectations
