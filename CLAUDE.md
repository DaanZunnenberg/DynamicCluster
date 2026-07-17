# CLAUDE.md

## Goal

This repository implements a GAS (Generalized Autoregressive Score) filter
driving an HMM-style dynamic mixture model for clustering multivariate panel
data, with simulation and estimation in a Python package (`dynamiccluster/`)
and post-processing/plotting in R (`R/prepareDataForPlotting.R`,
`R/plottingRoutines.R`).

We are turning this into a proper, installable/reusable **package** based on
the theory and methods implemented here, rather than a one-off simulation
script. When working in this repo, act like a software engineer/developer
building a maintainable library, not a researcher running a one-time
experiment:

- Favor clear package structure (proper module layout, `__init__.py`,
  clean public API) — already in place under `dynamiccluster/`; keep it
  that way as the package grows.
- Preserve the underlying statistical/mathematical methodology exactly
  (the GAS filter recursion, likelihood computation, cluster estimation
  logic) — refactor the engineering, not the theory, unless explicitly
  asked to change the method.
- Add the usual engineering scaffolding as needed: packaging metadata,
  dependency management, tests, docstrings/typing, and separation of
  configuration (the study parameters in `scripts/run_simulation.py`)
  from library code.
- Keep changes traceable to the original paper's algorithm described in
  the README (GAS filter steps 1-6) so correctness can be checked against
  the source material.

## Repository layout (current)

- `dynamiccluster/` — the installable Python package:
  - `state.py` — `SimulationState`, the container for simulated data,
    true/estimated parameters, and cluster probabilities.
  - `initialization.py` — allocation of time-varying parameter/data
    structures (`initialize_time_varying_parameter_structure`,
    `initialize_simulation_matrices`).
  - `simulation.py` — the data-generating process (`simulate_data` and its
    helpers).
  - `estimation.py` — maximum-likelihood estimation, including the
    HMM/GAS filter recursion (`run_hmm_gas_filter`,
    `estimate_maximum_likelihood`).
  - `utils.py` — generic numerical helpers, not specific to the method.
- `scripts/run_simulation.py` — entry point script; sets the study
  configuration (`random_seed`, `n_simulations`, `run_in_parallel`,
  `simulation_type`, ...), runs simulation + estimation, writes results
  to CSV.
- `R/prepareDataForPlotting.R` — reads simulation CSV, produces `.RData`.
- `R/plottingRoutines.R` — reads `.RData`, produces plots into `Results/`.
- `Results/` — output plots/results.

See README.md for full details on usage and the GAS filter algorithm.
