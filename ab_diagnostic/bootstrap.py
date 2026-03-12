"""
bootstrap.py
------------
Non-parametric inference via bootstrap and permutation tests.

Functions
---------
bootstrap_ci(df, ...)       Percentile bootstrap CI for any statistic
bootstrap_pvalue(df, ...)   Permutation p-value under the null of no effect

Both accept a `stat_fn` parameter so you can plug in any estimand:
  default              mean difference
  median difference    lambda t, c: np.median(t) - np.median(c)
  relative lift        lambda t, c: t.mean() / c.mean() - 1
  custom KPI           any fn(trt_array, ctrl_array) -> float
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional

import numpy as np

from .config import CONFIG, _f


def bootstrap_ci(
    df,
    metric_col:    str            = "revenue",
    stat_fn:       Optional[Callable] = None,
    treatment_col: str            = CONFIG["treatment_col"],
    treatment_val: str            = CONFIG["treatment_val"],
    control_val:   str            = CONFIG["control_val"],
    alpha:         float          = CONFIG["alpha"],
    n_bootstrap:   int            = CONFIG["n_bootstrap"],
    seed:          int            = CONFIG["bootstrap_seed"],
) -> Dict[str, Any]:
    """
    Percentile bootstrap confidence interval for any treatment effect statistic.

    Parameters
    ----------
    df           : experiment DataFrame
    metric_col   : outcome column to analyse
    stat_fn      : fn(trt_array, ctrl_array) -> float.
                   Defaults to difference in means.
    treatment_col: group assignment column
    treatment_val: treatment arm label
    control_val  : control arm label
    alpha        : significance level (CI covers 1-alpha)
    n_bootstrap  : number of bootstrap resamples
    seed         : random seed for reproducibility

    Returns
    -------
    Finding dict with detail keys:
      observed, ci_low, ci_high, se, n_bootstrap, contains_zero
    """
    if stat_fn is None:
        stat_fn = lambda t, c: float(t.mean() - c.mean())

    trt  = df[df[treatment_col] == treatment_val][metric_col].dropna().values
    ctrl = df[df[treatment_col] == control_val  ][metric_col].dropna().values

    observed = stat_fn(trt, ctrl)

    rng = np.random.default_rng(seed)
    boot_stats = np.array([
        stat_fn(
            rng.choice(trt,  size=len(trt),  replace=True),
            rng.choice(ctrl, size=len(ctrl), replace=True),
        )
        for _ in range(n_bootstrap)
    ])

    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    se    = float(boot_stats.std())

    contains_zero = ci_lo <= 0 <= ci_hi
    sig_str = "not significant" if contains_zero else "significant"

    msg = (
        f"'{metric_col}' bootstrap CI [{ci_lo:.4f}, {ci_hi:.4f}] "
        f"(observed={observed:+.4f}, SE={se:.4f}) — {sig_str}."
    )
    return _f(
        "warn" if contains_zero else "ok",
        f"BOOT_{metric_col.upper()}",
        msg,
        detail=dict(
            observed      = round(float(observed), 4),
            ci_low        = round(ci_lo, 4),
            ci_high       = round(ci_hi, 4),
            se            = round(se, 4),
            n_bootstrap   = n_bootstrap,
            contains_zero = contains_zero,
        ),
    )


def bootstrap_pvalue(
    df,
    metric_col:    str            = "revenue",
    stat_fn:       Optional[Callable] = None,
    treatment_col: str            = CONFIG["treatment_col"],
    treatment_val: str            = CONFIG["treatment_val"],
    control_val:   str            = CONFIG["control_val"],
    alpha:         float          = CONFIG["alpha"],
    n_bootstrap:   int            = CONFIG["n_bootstrap"],
    seed:          int            = CONFIG["bootstrap_seed"],
) -> Dict[str, Any]:
    """
    Permutation-based p-value under the sharp null hypothesis of no effect.

    Randomly relabels treatment/control assignments B times and asks how
    often the permuted statistic is as extreme as the observed one.
    This is a model-free test that works for any metric and estimand.

    Parameters
    ----------
    df           : experiment DataFrame
    metric_col   : outcome column to analyse
    stat_fn      : fn(trt_array, ctrl_array) -> float.
                   Defaults to difference in means.
    treatment_col: group assignment column
    treatment_val: treatment arm label
    control_val  : control arm label
    alpha        : significance level
    n_bootstrap  : number of permutation resamples
    seed         : random seed

    Returns
    -------
    Finding dict with detail keys:
      observed, pvalue, n_permutations
    """
    if stat_fn is None:
        stat_fn = lambda t, c: float(t.mean() - c.mean())

    trt  = df[df[treatment_col] == treatment_val][metric_col].dropna().values
    ctrl = df[df[treatment_col] == control_val  ][metric_col].dropna().values

    observed = stat_fn(trt, ctrl)
    combined = np.concatenate([trt, ctrl])
    n_trt    = len(trt)

    rng = np.random.default_rng(seed)
    perm_stats = []
    for _ in range(n_bootstrap):
        perm = rng.permutation(combined)
        perm_stats.append(stat_fn(perm[:n_trt], perm[n_trt:]))
    perm_stats = np.array(perm_stats)

    pval = float(np.mean(np.abs(perm_stats) >= np.abs(observed)))
    sig  = pval < alpha

    msg = (
        f"'{metric_col}' permutation p-value={pval:.4f} "
        f"({'significant' if sig else 'not significant'} at α={alpha}). "
        f"B={n_bootstrap} permutations."
    )
    return _f(
        "ok" if sig else "warn",
        f"PERM_{metric_col.upper()}",
        msg,
        pvalue=pval,
        detail=dict(
            observed        = round(float(observed), 4),
            n_permutations  = n_bootstrap,
        ),
    )
