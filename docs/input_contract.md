# Interval input contract

## One observation block per indicator

`build_problem` accepts a non-empty list of dictionaries. Each dictionary has:

| Key | Shape | Meaning |
|---|---|---|
| `X` | n × 22 | Interval-integrated design |
| `y` | n | Standardised observed within-person differences |
| `w` | n | Finite non-negative weights; optional, default one |

All arrays must be finite. Weights must have positive total weight within each indicator. Zero-weight rows are permitted. Indicator-specific missing observations should be handled while constructing each block, rather than taking a complete intersection over every indicator. The caller must preserve valid temporal order, cohort eligibility and risk-set membership.

Within each block, weights are normalised to sum to one. Thus each indicator has equal weight in the joint objective; a large observation count for one marker does not automatically dominate the other markers.

## Column order

Columns 0–6 contain integrated reference-background basis exposures. Columns 7–13 contain the same interval basis multiplied by case status. Columns 14–21 hold the eight existing nuisance terms. The solver profiles columns 0–6 and 14–21 and applies the penalty only to columns 7–13.

In the synthetic benchmark the nuisance terms are interval duration multiplied by centred age, sex and baseline disease status, followed by assessment-period differences and unused zero placeholders. The cohort design uses the study-specific adjustment and measurement-period terms documented in the manuscript. Matching information and reporting-time uncertainty enter through the prepared designs and weights; they are not inferred by the estimator from a participant identifier.

Seven piecewise rate bins have boundaries −8, −6, −4, −2, 0, 2, 4 and 6 years relative to the reporting anchor. The benchmark provides `basis`, `kernel_basis` and `phase_matrix` functions. The basis integrates rates over an actual measurement span. Its reporting-time kernel uses 24-point Gaussian quadrature over a uniform ±1-year offset in the synthetic study.

## NPZ interface

The command-line `fit` input is a non-object NPZ archive containing:

```text
n_markers
X_0, y_0, w_0
X_1, y_1, w_1
...
```

`n_markers` is a scalar integer. Weight arrays are optional. Loading uses `allow_pickle=False`. The package does not require names, dates of birth, addresses or participant identifiers in this file.

## Parameters and outputs

For `nuclear`, `param` is a non-negative fraction of the spectral norm of the profiled score matrix. The pure-sparse route uses the same reference scale for its elementwise L1 penalty. `smooth` multiplies a second-difference penalty by the mean diagonal scale of each indicator's Gram matrix. `posthoc` uses an integer rank. `lowrank_sparse` takes a dictionary with `fraction` and `ratio` through the Python API.

`theta` is the 7 × indicator excess-rate matrix. `beta` is the full 22 × indicator coefficient matrix after nuisance refitting. `lowrank` and `sparse` are decomposition components; only their sum gives the fitted excess rate. Rank and components are properties of the estimated coefficient matrix, not counts of biological systems or person-specific disease stages.

Standardising outcomes is the caller's responsibility. Training-reference scales must be estimated within the training split when assessing out-of-sample performance. Native-unit changes follow by multiplying estimated standardised amplitudes by the corresponding reference standard deviation.
