import numpy as np

from src.solver import Solver


def solve_makespan(
    durations: np.ndarray, affectations: np.ndarray, heuristic_name: str
) -> int:
    """Solve the given JSSP instance with the dispatching rule and return
    the makespan of the schedule.

    ---
    Args:
        durations: Processing times of the jobs.
            Shape of [n_jobs, n_machines].
        affectations: Machine specification of the jobs.
            Shape of [n_jobs, n_machines].
        heuristic_name: Name of the dispatching rule.

    ---
    Returns:
        Makespan of the schedule.
    """
    solver = Solver(durations, affectations, heuristic_name)
    schedule = solver.solve()
    ending_times = schedule + durations
    return ending_times.max()


if __name__ == "__main__":
    from src.solver import HEURISTICS
    from src.taillard import generate_taillard

    n_jobs = 100
    n_machines = 20
    durations, affectations = generate_taillard(n_jobs, n_machines, seed=0)

    print(f"Solving a random {n_jobs}x{n_machines} instance.")
    for heuristic_name in HEURISTICS.keys():
        makespan = solve_makespan(durations, affectations, heuristic_name)
        print(f"{heuristic_name}: {makespan}")
