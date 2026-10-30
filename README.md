# Diabetes Step Change Model

Joint interval reconstruction of population multisystem change around diabetes recognition.

The software estimates how measured change in incident cases exceeds change in a contemporaneous reference population. Each indicator contributes its own actual observation intervals. A nuclear-norm penalty shares information across the time-by-indicator excess-rate matrix during fitting. This repository contains the estimator, comparators, the independent synthetic benchmark, and anonymous aggregate data for the accompanying population figures.

## Install

Python 3.10 or later is required. From the repository root:

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install ".[plots,test]"
pytest -q
```

The command-line entry point is `dscm`. Every command also works as `python -m diabetes_step_change` after installation.

## Run a small demonstration

```bash
dscm demo --output outputs/demo
```

The quick profile uses one calibration and two evaluation datasets per scenario, with the same data generator and candidate grids as the full experiment. Its estimates illustrate the workflow; the manuscript numbers come from the full profile.

## Reproduce the complete synthetic experiment

```bash
dscm benchmark --profile manuscript --output outputs/manuscript
```

This generates 108 synthetic datasets: three scenarios, each with 12 calibration and 24 independent evaluation datasets. It fits 828 calibration candidates and 360 evaluation models. Parameter choice uses the calibration datasets; all 72 evaluation datasets use the chosen parameters. The fixed seed base is `131030000`.

The three scenarios are smooth-only change, shared change with a marker-specific exception, and unshared processes. Each contains 240 matched sets, 20 indicators, three assessments about four years apart, reporting-time error, visit-time perturbation and 15% indicator-visit missingness. The source explicitly keeps spans across a missed middle assessment.

Output files include the design record, selected parameters, all calibration and evaluation results, paired method differences, representative coefficients and completion diagnostics. An existing non-empty output directory is never overwritten.

The full experiment compares independent fitting, strong smooth fitting, post-estimation rank truncation, direct nuclear-norm estimation, and a low-rank-plus-sparse extension. The pure-sparse comparator was a separate targeted comparison in the cohort analysis and is available in the estimator API. It is not retroactively added to the original synthetic experiment.

## Fit prepared interval arrays

```bash
python examples/create_synthetic_input.py --output outputs/input/synthetic.npz
dscm fit --input outputs/input/synthetic.npz --output outputs/example_fit --method nuclear --parameter 0.03
```

The input format is documented in [docs/input_contract.md](docs/input_contract.md). The output contains model coefficients and diagnostics, with no participant observations. For a Python workflow:

```python
from diabetes_step_change import build_problem, fit, predict

# One dictionary per indicator. X has 22 columns; y and w have matching rows.
data = [{"X": X0, "y": y0, "w": w0}, {"X": X1, "y": y1, "w": w1}]
problem = build_problem(data)
fitted = fit(problem, method="nuclear", param=0.03)
expected_change = predict(fitted, X0, marker_index=0)
```

The number and membership of observations can differ by indicator. The seven-bin design is the design evaluated in the study; this release does not silently generalise it to an arbitrary number of temporal bins.

## Reproduce aggregate figure views

```bash
python examples/plot_aggregate_results.py --data data/aggregate --output outputs/aggregate_figures
```

This writes ELSA excess-change geometry and NHANES state/co-occurrence views from the supplied anonymous estimates. Labels explicitly distinguish a **self-reported diabetes diagnosis** from current HbA1c. Projections support visual comparison of the three coordinates. The script is an aggregate-results example, not a refit of participant data.

## What the method adds

The method estimates a joint excess-change object directly from marker-specific interval observations. Background change and adjustment terms are profiled out, while nuclear-norm shrinkage acts on the excess-rate matrix. Information sharing occurs during estimation rather than through dimension reduction applied after independent fitting.

In the fixed synthetic benchmark, shared-change phase-amplitude RMSE was approximately 0.04946 for nuclear-norm reconstruction and 0.05389 for strong smoothing. The reduction was 8.23%; performance was similar in the smooth-only and unshared scenarios. The cohort analysis used joint structure for process interpretation and retained the independent and sparse comparisons.

Integral matching, nuclear-norm regularisation and low-rank-plus-sparse decomposition are established methods. [docs/methods.md](docs/methods.md) explains their original applications and the specific observation-model adaptation used here. The package does not reproduce the full change-point search of the cited low-rank paper.

## Data and reproducibility

The simulation and solver checks are reproducible without participant data. The `data/aggregate` directory contains population estimates and covariance summaries only. ELSA, CHARLS and NHANES retain different evidence roles: longitudinal process estimation, observed-interval replication, and survey-weighted population states. The repository does not redistribute restricted participant records or claim that the supplied aggregates reproduce cohort eligibility or data cleaning. Access instructions and the scope of each executable are in [docs/reproducibility.md](docs/reproducibility.md).

## Authors and support

Jianyi Zhang and Zhongwei Jia, Department of Global Health, School of Public Health, Peking University, Beijing, China.

Supported by the Prevention and Control of Emerging and Major Infectious Diseases-National Science and Technology Major Project (2025ZD01901000).

Software citation metadata are provided in [CITATION.cff](CITATION.cff). No software DOI or manuscript publication identifier is assigned in this release.

## Licence

The software is released under the [MIT License](LICENSE). Source cohort access conditions remain applicable to participant data. Methodological references retain their original attribution.
