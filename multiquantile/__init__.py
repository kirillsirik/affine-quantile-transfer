"""Quartile study and slope-test run: several quantile levels fitted to the same responses.

Modules: design (data-generating process), estimation (quantile-regression fits and the slope
Wald test), weighting (density-based slope weights, used by the closed-form values),
run_quartiles and run_slope_test (study runners), validate_quartiles and validate_slope_test
(checks of saved results), common (I/O).
Run the command-line modules from the repository root, for example
``python -m multiquantile.run_quartiles --out build/quartiles``.
"""
