"""
ab_diagnostics
==============
Lightweight A/B experiment diagnostics toolkit.

Quick start
-----------
    from ab_diagnostics import run_ab_diagnostics, print_ab_report

    results = run_ab_diagnostics(
        df,
        continuous_metrics     = ['revenue'],
        binary_metrics         = ['converted'],
        continuous_covariates  = ['age', 'tenure_days'],
        categorical_covariates = ['platform'],
    )
    print_ab_report(results)

Modules
-------
config      CONFIG dict and Finding helpers (ok / warn / fail)
srm         Sample Ratio Mismatch check
balance     Covariate balance diagnostics
effects     Treatment effect estimation
bootstrap   Bootstrap CI and permutation p-value
pipeline    run_ab_diagnostics(), print_ab_report()
"""

from .config   import CONFIG, make_config
from .srm      import check_srm
from .balance  import check_covariate_balance
from .effects  import (
    estimate_effect_continuous,
    estimate_effect_binary,
    estimate_effect_nonparametric,
)
from .bootstrap import bootstrap_ci, bootstrap_pvalue
from .pipeline  import run_ab_diagnostics, print_ab_report

__all__ = [
    # config
    "CONFIG",
    "make_config",
    # modules
    "check_srm",
    "check_covariate_balance",
    "estimate_effect_continuous",
    "estimate_effect_binary",
    "estimate_effect_nonparametric",
    "bootstrap_ci",
    "bootstrap_pvalue",
    # pipeline
    "run_ab_diagnostics",
    "print_ab_report",
]
