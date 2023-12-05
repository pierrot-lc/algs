import numpy as np
import pytest

from .solver import Solver
from .rules import HEURISTICS
from ..taillard import generate_taillard
from ..validate import validate_solution


@pytest.mark.parametrize(
    "durations, affectations, heuristic, expected_schedule",
    [
        (
            np.array([[12, 20], [5, 13]]),
            np.array([[1, 0], [0, 1]]),
            "SPT",
            np.array([[0, 12], [0, 12]]),
        ),
        (
            np.array([[12, 20], [5, 13]]),
            np.array([[0, 1], [0, 1]]),
            "SPT",
            np.array([[5, 18], [0, 5]]),
        ),
        (
            np.array([[12, 20, 6], [5, 13, 20]]),
            np.array([[0, 2, 1], [0, 1, 2]]),
            "SPT",
            np.array([[5, 17, 37], [0, 5, 37]]),
        ),
        (
            np.array([[12, 15, 8], [3, 11, 17], [8, 9, 10]]),
            np.array([[1, 0, 2], [1, 2, 0], [2, 0, 1]]),
            "SPT",
            np.array([[3, 17, 32], [0, 8, 32], [0, 8, 17]]),
        ),
    ],
)
def test_solver(
    durations: np.ndarray,
    affectations: np.ndarray,
    heuristic: str,
    expected_schedule: np.ndarray,
):
    solver = Solver(durations, affectations, heuristic)
    schedule = solver.solve()
    assert np.all(schedule == expected_schedule)

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
def test_rules(n_jobs: int, n_machines: int, seed: int):
    durations, affectations = generate_taillard(n_jobs, n_machines, seed)
    for heuristic_name in HEURISTICS.keys():
        solver = Solver(durations, affectations, heuristic_name)
        schedule = solver.solve()
        validate_solution(durations, affectations, schedule)
