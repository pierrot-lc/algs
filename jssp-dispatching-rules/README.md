# Job-Shop Scheduling Problems with Dispatching Rules

This repo provides an implementation of a JSSP solver using dispatching rules.

![Job Shop Scheduling Problems Illustration](.images/jssp-illustration.png)

It uses a greedy approach with a priority queue simulating the tasks being solved
incrementally. Every time a new task has to be processed by a machine, a pool of
candidates is taken and the dispatching rule is used to select one of the candidates.

When looking for such a solver online, I didn't find any. I hope this repo will be
useful to someone

## How to use

You simply provide the durations, affectations and the rule you want to use.
The solver will return the schedule, which is an array of shape `[n_jobs, n_machines]`
where each entry `(i, j)` is the starting time of the task `(i, j)`.

Example :

```py
from src.solver import Solver
from src.taillard import generate_taillard

n_jobs, n_machines = 100, 20
durations, affectations = generate_taillard(n_jobs, n_machines, seed=0)

solver = Solver(durations, affectations, heuristic="MOPNR")
schedule = solver.solve()
```

You can find all the supported heuristics in `src/solver/rules.py`.
Look in the `main.py` file for a complete example.

The repo also provides a rescheduling method. This allows you to compute a new schedule
based on an already defined schedule with respect to new durations. For example, you may
compute a schedule based on some estimated durations, and you want now to evaluate this
schedule under the real durations.

Example :

```py
import numpy as np

from src.reschedule import reschedule
from src.solver import Solver
from src.taillard import generate_taillard


n_jobs, n_machines = 100, 20
durations, affectations = generate_taillard(n_jobs, n_machines, seed=0)

solver = Solver(durations, affectations, heuristic="MOPNR")
schedule = solver.solve()

rng = np.random.default_rng(0)
new_durations = durations + rng.integers(-10, 10, size=(n_jobs, n_machines))
new_durations[new_durations <= 0] = 1
new_schedule = reschedule(new_durations, affectations, schedule)
```

This new computed schedule does not modify the order in which tasks are processed for
each machine. It simply makes sure that no time is wasted, so that the new schedule
is optimal for the given tasks order.

## Sources

The code was originally written by me for [Wheatley](https://github.com/jolibrain/wheatley),
a RL planning algorithm using GNNs to solve JSSP and other related problems.

Dispatching rules based solver is presented in the paper
[Priority dispatching rules in a fabrication/assembly shop (1990)](https://www.sciencedirect.com/science/article/pii/089571779090372T).

The dispatching rules implemented in this repo are based on the definition
given in the paper
[Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning (2020)](https://arxiv.org/abs/2010.12367).
