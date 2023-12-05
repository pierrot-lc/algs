import numpy as np
import pytest

from .reschedule import compute_occupancy, reschedule
from ..solver import Solver
from ..taillard import generate_taillard
from ..validate import validate_solution


@pytest.mark.parametrize(
    "n_jobs, n_machines, seed",
    [
        (6, 6, 0),
        (10, 10, 0),
        (15, 20, 0),
        (20, 20, 1),
        (20, 20, 0),
        (50, 20, 95),
    ],
)
def test_reschedule(n_jobs: int, n_machines: int, seed: int):
    durations, affectations = generate_taillard(n_jobs, n_machines, seed)
    solver = Solver(durations, affectations, "SPT")
    schedule = solver.solve()
    same_schedule = reschedule(durations, affectations, schedule)
    validate_solution(durations, affectations, same_schedule)
    assert np.all(
        same_schedule == schedule
    ), "The schedule changed even though durations were the same"

    rng = np.random.default_rng(seed)
    new_durations = durations + rng.integers(-10, 10, size=(n_jobs, n_machines))
    new_durations[new_durations <= 0] = 1
    new_schedule = reschedule(new_durations, affectations, schedule)

    validate_solution(new_durations, affectations, new_schedule)
    occupancy_1 = compute_occupancy(affectations, schedule)
    occupancy_2 = compute_occupancy(affectations, new_schedule)
    for machine_id in range(len(occupancy_1)):
        assert np.all(
            occupancy_1[machine_id] == occupancy_2[machine_id]
        ), "The new schedule have swapped machine-tasks priorities"
