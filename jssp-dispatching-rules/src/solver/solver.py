from queue import PriorityQueue

import numpy as np

from .rules import HEURISTICS
from ..validate import validate_instance, validate_solution


class Solver:
    """Solver using a simulation rollout with dispatching rules.

    ---
    Args:
        processing_times: Processing times of the jobs.
            Shape of [n_jobs, n_machines].
        machines: Machine specification of the jobs.
            Shape of [n_jobs, n_machines].
    """

    def __init__(
        self,
        durations: np.ndarray,
        affectations: np.ndarray,
        heuristic: str,
    ):
        validate_instance(durations, affectations)
        assert heuristic in HEURISTICS, f"Unknown heuristic {heuristic}"

        self.durations = durations
        self.affectations = affectations
        self.n_jobs, self.n_machines = durations.shape
        self.heuristic = HEURISTICS[heuristic]

        # Schedule is the starting time of each task.
        self.schedule = np.zeros(
            (self.n_jobs, self.n_machines + 1),  # We add a fictive ending task.
            dtype=self.durations.dtype,
        )
        self.schedule.fill(-1)  # -1 for unknown starting times.
        self.priority_queue = PriorityQueue(maxsize=0)

        # The first events are empty.
        self.priority_queue.put((0, np.unique(self.affectations)))

    def solve(self) -> np.ndarray:
        while np.any(self.schedule[:, :-1] == -1):
            current_time, machine_ids = self.priority_queue.get()
            failed_steps = [
                machine_id
                for machine_id in machine_ids
                if not self.step(machine_id, current_time)
            ]

            if 0 < len(failed_steps) < len(machine_ids):
                # Some tasks have been scheduled. We can try again.
                self.priority_queue.put((current_time, failed_steps))
            elif len(failed_steps) == len(machine_ids):
                # All tasks have failed, we can safely wait for more tasks to end.
                next_ending_time, next_machine_ids = self.priority_queue.get()
                next_machine_ids.extend(failed_steps)
                self.priority_queue.put((next_ending_time, next_machine_ids))

        validate_solution(
            self.durations,
            self.affectations,
            self.schedule[:, :-1],
        )
        return self.schedule[:, :-1]

    def step(self, machine_id: int, current_time: int) -> bool:
        """Update the priority queue and the current solution by adding
        a job for the given machine.
        If no candidates are available, returns False and do nothing.
        """
        valid_candidates = self.candidates(machine_id, current_time)
        if len(valid_candidates) == 0:
            return False

        selected_job = self.priority_rule(valid_candidates)
        starting_time = self.canditate_starting_time(selected_job, current_time)

        task_id = self.schedule[selected_job].argmin()
        ending_time = starting_time + self.durations[selected_job, task_id]
        self.priority_queue.put((ending_time, [machine_id]))
        self.schedule[selected_job, task_id] = starting_time

        return True

    def candidates(self, machine_id: int, current_time: int) -> np.ndarray:
        """Select the valid job candidates.
        A candidate is valid if:
            - It is the next unplaced task in its job and that task is to be done
            on the given `machine_id`.
            - Its previous placed task is finished.

        The returned candidates can be an empty array if there is no
        valid candidate.

        ---
        Args:
            machine_id: The machine onto which we filter the valid candidates.
            current_time: The current time of the simulation.

        ---
        Returns:
            The indices of the jobs for which we consider their next task as
            valid candidates.
                Shape of [n_valid_candidates,]
        """
        job_ids = np.arange(self.n_jobs)

        # If a job is fully done, its frontier candidate will have an id of `n_machines`.
        frontier_candidates = self.schedule.argmin(axis=1)  # Shape of [n_jobs,].

        # Add the fictive machine '-1' for finished jobs.
        affectations = np.concatenate(
            (
                self.affectations,
                np.zeros((self.n_jobs, 1), dtype=self.affectations.dtype),
            ),
            axis=1,
        )
        affectations[:, -1] = -1
        candidates_machine_id = affectations[job_ids, frontier_candidates]

        # Ignore frontier candidates that do not concern the given `machine_id`.
        valid_mask = candidates_machine_id == machine_id

        # Find the ending time of each precedent frontier candidate.
        # In case of a starting candidate, its precedent ending time will be 0.
        ending_times = self.schedule[:, :-1] + self.durations
        ending_times = np.concatenate(
            (np.zeros((self.n_jobs, 1), dtype=self.durations.dtype), ending_times),
            axis=1,
        )
        precedences_ending_times = ending_times[job_ids, frontier_candidates]

        # Ignore tasks that have unfinished precedences.
        finished_precedences = precedences_ending_times <= current_time
        valid_mask = valid_mask & finished_precedences

        valid_jobs = job_ids[valid_mask]
        return valid_jobs

    def priority_rule(self, candidates: np.ndarray) -> int:
        """Choose a candidate among the selected ones."""
        return self.heuristic(
            self.durations,
            self.affectations,
            self.schedule,
            candidates,
        )

    def canditate_starting_time(self, job_id: int, current_time: int) -> int:
        """Determine the candidate starting time, which is either
        the current time or the time it takes for its previous task to finish.
        """
        task_id = self.schedule[job_id].argmin()

        if task_id == 0:
            return current_time

        previous_starting_time = self.schedule[job_id, task_id - 1]
        previous_process_time = self.durations[job_id, task_id - 1]
        previous_ending_time = previous_starting_time + previous_process_time

        return max(previous_ending_time, current_time)
