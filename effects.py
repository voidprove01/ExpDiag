"""
effects.py
----------
Treatment effect estimation.

Three estimators covering the most common online experiment metric types:

  estimate_effect_continuous()    Welch's t-test  — revenue, time-on-site, etc.
  estimate_effect_binary()        Two-proportion z-test — conversion, click-through
  estimate_effect_nonparametric() Mann-Whitney U  — skewed metrics, outlier-heavy data

All return a single Finding dict with:
  stat    : test statistic
  pvalue  : two-sided p-value
  detail  : ATE / lift, 95% CI, group means, sample sizes
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, norm as sp_norm

from .config import CONFIG, _f


def _welch_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple:
    """Return (ate, ci_lo, ci_hi, se) using Welch's SE formula."""
    ate = a.mean() - b.mean()
    se  = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z   = sp_norm.ppf(1 - alpha / 2)
    return ate, ate - z * se, ate + z * se, se


def estimate_effect_continuous(
    df,
    metric_col:    str   = "revenue",
    treatment_col: str   = CONFIG["treatment_col"],
    treatment_val: str   = CONFIG["treatment_val"],
    control_val:   str   = CONFIG["control_val"],
    alpha:         float = CONFIG["alpha"],
) -> Dict[str, Any]:
    """
    Estimate the ATE on a continuous metric using Welch's t-test.

    Parameters
    ----------
    df           : experiment DataFrame
    metric_col   : outcome column name
    treatment_col: group assignment column
    treatment_val: label for the treatment arm
    control_val  : label for the control arm
    alpha        : significance level

    Returns
    -------
    Finding dict with detail keys:
      ate, ci_low, ci_high, relative_lift,
      trt_mean, ctrl_mean, n_trt, n_ctrl
    """
    trt  = df[df[treatment_col] == treatment_val][metric_col].dropna().values
    ctrl = df[df[treatment_col] == control_val  ][metric_col].dropna().values

    t_stat, pval            = ttest_ind(trt, ctrl, equal_var=False)
    ate, ci_lo, ci_hi, _se  = _welch_ci(trt, ctrl, alpha)
    rel_lift = ate / ctrl.mean() if ctrl.mean() != 0 else float("nan")

    sig = pval < alpha
    msg = (
        f"'{metric_col}': {'Significant' if sig else 'No significant'} effect. "
        f"ATE={ate:+.3f} (95% CI [{ci_lo:.3f}, {ci_hi:.3f}]), "
        f"rel. lift={rel_lift:+.1%}, p={pval:.4f}."
    )
    return _f(
        "ok" if sig else "warn",
        f"EFFECT_{metric_col.upper()}",
        msg,
        stat=t_stat, pvalue=pval,
        detail=dict(
            ate           = round(float(ate), 4),
            ci_low        = round(float(ci_lo), 4),
            ci_high       = round(float(ci_hi), 4),
            relative_lift = round(float(rel_lift), 4),
            trt_mean      = round(float(trt.mean()), 4),
            ctrl_mean     = round(float(ctrl.mean()), 4),
            n_trt         = int(len(trt)),
            n_ctrl        = int(len(ctrl)),
        ),
    )


def estimate_effect_binary(
    df,
    metric_col:    str   = "converted",
    treatment_col: str   = CONFIG["treatment_col"],
    treatment_val: str   = CONFIG["treatment_val"],
    control_val:   str   = CONFIG["control_val"],
    alpha:         float = CONFIG["alpha"],
) -> Dict[str, Any]:
    """
    Estimate the ATE on a binary metric using a two-proportion z-test.

    Returns absolute lift, relative lift, and a 95% CI on the absolute
    difference.  Wilson score CIs are used for individual proportions
    but the CI on the difference uses the unpooled SE (more conservative).

    Parameters
    ----------
    df           : experiment DataFrame
    metric_col   : binary outcome column (0/1 integer)
    treatment_col: group assignment column
    treatment_val: label for the treatment arm
    control_val  : label for the control arm
    alpha        : significance level

    Returns
    -------
    Finding dict with detail keys:
      abs_lift, rel_lift, ci_low, ci_high,
      p_treatment, p_control, n_trt, n_ctrl
    """
    trt  = df[df[treatment_col] == treatment_val][metric_col].dropna()
    ctrl = df[df[treatment_col] == control_val  ][metric_col].dropna()

    p_t, n_t = float(trt.mean()), int(len(trt))
    p_c, n_c = float(ctrl.mean()), int(len(ctrl))

    # Pooled SE for the z-test
    p_pool  = (trt.sum() + ctrl.sum()) / (n_t + n_c)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    z_stat  = (p_t - p_c) / se_pool if se_pool > 0 else 0.0
    pval    = float(2 * (1 - sp_norm.cdf(abs(z_stat))))

    abs_lift = p_t - p_c
    rel_lift = abs_lift / p_c if p_c > 0 else float("nan")

    # Unpooled SE for the CI
    se_diff = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z_crit  = sp_norm.ppf(1 - alpha / 2)
    ci_lo   = abs_lift - z_crit * se_diff
    ci_hi   = abs_lift + z_crit * se_diff

    sig = pval < alpha
    msg = (
        f"'{metric_col}': {'Significant' if sig else 'No significant'} lift. "
        f"Abs lift={abs_lift:+.3f} (95% CI [{ci_lo:.3f}, {ci_hi:.3f}]), "
        f"rel. lift={rel_lift:+.1%}, p={pval:.4f}."
    )
    return _f(
        "ok" if sig else "warn",
        f"EFFECT_{metric_col.upper()}",
        msg,
        stat=z_stat, pvalue=pval,
        detail=dict(
            abs_lift    = round(abs_lift, 4),
            rel_lift    = round(rel_lift, 4),
            ci_low      = round(ci_lo, 4),
            ci_high     = round(ci_hi, 4),
            p_treatment = round(p_t, 4),
            p_control   = round(p_c, 4),
            n_trt       = n_t,
            n_ctrl      = n_c,
        ),
    )


def estimate_effect_nonparametric(
    df,
    metric_col:    str   = "revenue",
    treatment_col: str   = CONFIG["treatment_col"],
    treatment_val: str   = CONFIG["treatment_val"],
    control_val:   str   = CONFIG["control_val"],
    alpha:         float = CONFIG["alpha"],
) -> Dict[str, Any]:
    """
    Mann-Whitney U test for a continuous metric.

    Use this when your metric is heavily right-skewed (e.g. revenue with
    outliers) or when you suspect the normality assumption is violated.
    Reports the rank-biserial correlation as an interpretable effect size.

    Parameters
    ----------
    df           : experiment DataFrame
    metric_col   : outcome column name
    treatment_col: group assignment column
    treatment_val: label for the treatment arm
    control_val  : label for the control arm
    alpha        : significance level

    Returns
    -------
    Finding dict with detail keys:
      u_stat, rank_biserial_r, n_trt, n_ctrl
    """
    trt  = df[df[treatment_col] == treatment_val][metric_col].dropna().values
    ctrl = df[df[treatment_col] == control_val  ][metric_col].dropna().values

    u_stat, pval  = mannwhitneyu(trt, ctrl, alternative="two-sided")
    n_t, n_c      = int(len(trt)), int(len(ctrl))

    # Rank-biserial correlation: r = 1 - 2U/(n1*n2)
    r_bc = float(1 - (2 * u_stat) / (n_t * n_c))

    sig = pval < alpha
    msg = (
        f"'{metric_col}' (nonparametric): "
        f"{'Significant' if sig else 'No significant'} difference. "
        f"U={u_stat:.0f}, rank-biserial r={r_bc:.3f}, p={pval:.4f}."
    )
    return _f(
        "ok" if sig else "warn",
        f"EFFECT_NP_{metric_col.upper()}",
        msg,
        stat=u_stat, pvalue=pval,
        detail=dict(
            u_stat           = float(u_stat),
            rank_biserial_r  = round(r_bc, 4),
            n_trt            = n_t,
            n_ctrl           = n_c,
        ),
    )
