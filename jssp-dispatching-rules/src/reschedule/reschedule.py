import numpy as np


def reschedule(
    durations: np.ndarray, affectations: np.ndarray, schedule: np.ndarray
) -> np.ndarray:
    """Adapt the current schedule to take into account the given durations.

    Does not modify the order of the schedule. It just makes sure that each
    task is not starting before its precedency and that each machine is not
    working on two tasks at the same time.

    The new schedule is optimal w.r.t. the given occupancy defined by the
    original schedule.

    ---
    Args:
        durations: The new durations of the tasks.
            Shape of [n_jobs, n_machines].
        affectations: The affectations of the tasks.
            Shape of [n_jobs, n_machines].
        schedule: The current schedule.
            Shape of [n_jobs, n_machines].

    ---
    Returns:
        The new schedule.
            Shape of [n_jobs, n_machines].
    """
    # Save the original tasks order of each machine.
    original_occupancy = compute_occupancy(affectations, schedule)

    # Start by a trivial schedule and make sure that each task job is at least
    # starting right after its precedency.
    new_schedule = init_schedule(durations)

    # Iterate until we have a fixed point solution.
    # During each iteration, we separately fix the job constraints and the
    # machines constraints.
    while not np.all(schedule == new_schedule):
        schedule = new_schedule
        new_schedule = reschedule_jobs(durations, new_schedule)
        new_schedule = reschedule_machines(
            durations, affectations, original_occupancy, new_schedule
        )

    return new_schedule


def reschedule_jobs(durations: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """Modify the schedule to make sure that the job constraints
    are respected.
    """
    n_tasks = durations.shape[1]
    schedule = schedule.copy()

    for task_id in range(1, n_tasks):
        starting_times = np.stack(
            (
                schedule[:, task_id],
                schedule[:, task_id - 1] + durations[:, task_id - 1],
            ),
            axis=1,
        )
        schedule[:, task_id] = np.max(starting_times, axis=1)

    return schedule


def reschedule_machines(
    durations: np.ndarray,
    affectations: np.ndarray,
    occupancy: np.ndarray,
    schedule: np.ndarray,
) -> np.ndarray:
    """Modify the schedule to make sure that the machine constraints
    are respected.
    """
    n_machines = affectations.shape[1]
    schedule = schedule.copy()

    for machine_id in range(n_machines):
        schedule_machine = schedule[affectations == machine_id]
        durations_machine = durations[affectations == machine_id]
        occupancy_machine = occupancy[machine_id]

        schedule_machine = schedule_machine[occupancy_machine]
        durations_machine = durations_machine[occupancy_machine]

        ending_times = schedule_machine + durations_machine
        previous_ending_times = ending_times.copy()
        previous_ending_times[1:] = ending_times[:-1]
        previous_ending_times[0] = 0

        starting_times_candidates = np.stack(
            (schedule_machine, previous_ending_times), axis=1
        )
        schedule_machine = np.max(starting_times_candidates, axis=1)

        schedule_machine = schedule_machine[np.argsort(occupancy_machine)]
        schedule[affectations == machine_id] = schedule_machine

    return schedule


def init_schedule(durations: np.ndarray) -> np.ndarray:
    """Initialize a schedule by starting each job task when its precedency
    is finished. This is important to make sure that no gap between two tasks
    is let unfilled.
    """
    schedule = np.zeros_like(durations)
    for task_id in range(1, durations.shape[1]):
        schedule[:, task_id] = schedule[:, task_id - 1] + durations[:, task_id - 1]

    return schedule


def compute_occupancy(affectations: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """Compute the occupancy of each machine of the given schedule.
    The occupancy of a machine is the order a machine treat each job.

    ---
    Returns:
        The occupancy of each machine.
            Shape of [n_machines, n_jobs].
    """
    # Order the tasks by the machine they are affected to.
    sort_by_machines = np.argsort(affectations, axis=1)
    schedule = np.take_along_axis(schedule, sort_by_machines, axis=1)

    schedule = schedule.transpose()

    # Chronological order of the jobs for each machine.
    occupancy = np.argsort(schedule, axis=1)
    return occupancy
