# Reproducibility scope

| Component | Executable supplied | Inputs |
|---|---|---|
| Profiled joint estimator and comparators | `diabetes_step_change.core` | Prepared in-memory arrays |
| Solver special cases | `pytest -q` | Independently generated test arrays |
| Complete synthetic study | `dscm benchmark --profile manuscript` | Fixed synthetic generator and seeds |
| Prepared-array application | `dscm fit` | NPZ design blocks, no pickle |
| ELSA and NHANES aggregate figure views | `examples/plot_aggregate_results.py` | Anonymous CSV/JSON estimates in `data/aggregate` |
| Cohort access, eligibility, assay harmonisation and record linkage | Manuscript methods and source-provider documentation | Authorised participant data; not distributed |

## Cohort roles

ELSA supplies longitudinal intervals for population process reconstruction. CHARLS supplies actual repeat-measurement contrasts over its observed interval. NHANES supplies independent survey-weighted population states and concurrent abnormality prevalence. NHANES cycles are not linked as personal follow-up, and the aggregate figure script does not estimate individual transition times.

Source-provider access information:

- ELSA: https://www.elsa-project.ac.uk/accessing-elsa-data
- CHARLS: https://charls.pku.edu.cn/en/
- NHANES: https://www.cdc.gov/nchs/nhanes/

The software release contains no raw cohort records. The public figures can be reproduced from the supplied summaries. Repeating the cohort fits requires approved source data and a design constructed according to the manuscript. The synthetic study provides a fully executable example of that design and the model fitting steps.

## Aggregate files

- `elsa_nuclear_3d_points.csv`: fixed-stage coordinates of the population cumulative excess trajectory.
- `elsa_nuclear_3d_covariance.csv`: covariance of those estimated coordinates.
- `elsa_nuclear_curve_uncertainty.csv`: model-based case, reference and excess curves with conditional multiplier intervals.
- `elsa_nuclear_phase_uncertainty.csv`: marker-specific broad-stage change estimates and conditional intervals.
- `nhanes_joint_centroids.csv`: age- and sex-adjusted state differences in native and reference-standard-deviation units.
- `nhanes_joint_covariance.json`: survey-design covariance of the three state coordinates.
- `nhanes_joint_burden.csv`: survey-weighted prevalence of at least two concurrently observed abnormalities, with sample coverage.
- `evaluation_summary.csv` and `selected_parameters.json`: frozen reference outputs for the full synthetic experiment.

Small NHANES coverage cells remain suppressed as `<10` where provided in the source summaries. No individual rows or trajectories are included. The figure uncertainty regions describe estimated population means, not the distribution of individuals.

## Computational notes

The solver uses NumPy linear algebra. Exact floating-point agreement can vary with BLAS implementation, but the implementation tests use tolerances well below the reported precision. The complete benchmark uses a single-thread BLAS configuration and its original seeds, candidate grids and parameter-choice rule. Generated files record both source hashes and convergence diagnostics without recording workstation directories.

Every run requires a new or empty output directory. Preserve published results and run repetitions in a new directory.
