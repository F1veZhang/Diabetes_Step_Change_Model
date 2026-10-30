# Methodological background and implementation

## Observation target

For indicator m, the input contains observed changes across actual measurement spans. The design separates reference-population change, case-specific excess change and nuisance terms. Profiling the nuisance terms yields an indicator-specific Gram matrix H and score vector b. The direct joint model minimises the sum of profiled quadratic losses plus a nuclear-norm penalty on the seven-by-indicator excess-rate matrix.

The penalty favours a smaller number of temporal directions when supported by the observations. It is applied during fitting, while each indicator retains its own design and observation rows. An accelerated proximal-gradient solver uses singular-value soft thresholding and objective-based restart. The software also provides independent, second-difference smooth, post-estimation rank-truncated, low-rank-plus-sparse and pure-sparse estimates.

## Source ideas and the adaptation

Integral matching is an established approach to dynamical-system estimation. The simode detailed paper contains an explicit SIR example. It supplies the accumulated-change observation idea; it is not claimed as a new import into epidemiology.

Low-rank matrix estimation and trace regression provide the joint reconstruction principle. Shi and colleagues study low-rank matrix estimation in the presence of change-points and include multistation air-quality data. This package adapts the matrix-regularisation principle to marker-specific integrated risk-set contrasts. It does not implement that paper's complete change-point search or claim its theoretical guarantees for the present cohort design.

Low-rank-plus-sparse decomposition has an established dynamic-MRI application. It is included as a comparator. In the cohort application, its shared component was negligible and the pure-sparse comparison gave similar estimates. The component was therefore not interpreted as a shared biological mechanism.

The contribution evaluated here is the observation-level joint estimation of population excess change under incomplete and differently timed marker observations. The synthetic experiment tests where common temporal structure improves recovery and where it provides little gain.

## References

1. Yaari R, Dattner I. simode: R Package for Statistical Inference of Ordinary Differential Equations using Separable Integral-Matching. Journal of Open Source Software. 2019;4:1850. https://doi.org/10.21105/joss.01850. Detailed SIR example: https://arxiv.org/html/1807.04202.
2. Shi L, Wang G, Zou C. Low-Rank Matrix Estimation in the Presence of Change-Points. Journal of Machine Learning Research. 2024;25(220):1–71. https://jmlr.org/papers/v25/22-0852.html.
3. Otazo R, Candès E, Sodickson DK. Low-rank plus sparse matrix decomposition for accelerated dynamic MRI with separation of background and dynamic components. Magnetic Resonance in Medicine. 2015;73:1125–1136. https://doi.org/10.1002/mrm.25240.

The implementation in this repository is a dedicated implementation of the stated objective. No source code from those papers is bundled.
