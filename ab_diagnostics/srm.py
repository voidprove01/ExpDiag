"""
srm.py
------
Sample Ratio Mismatch (SRM) diagnostic.

An SRM means the observed allocation between arms doesn't match what was
configured in the experiment system.  It's one of the most common and
dangerous experiment bugs — any downstream effect estimates are invalid
when the split is broken.

Functions
---------
check_srm(df, ...)  ->  list[Finding]
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import chisquare

from .config import CONFIG, ok, warn, fail


def check_srm(
    df,
    treatment_col:  str   = CONFIG["treatment_col"],
    treatment_val:  str   = CONFIG["treatment_val"],
    control_val:    str   = CONFIG["control_val"],
    expected_split: float = CONFIG["expected_split"],
    alpha:          float = CONFIG["alpha"],
    tolerance:      float = CONFIG["srm_tolerance"],
) -> List[Dict[str, Any]]:
    """
    Sample Ratio Mismatch check.

    Tests whether the observed group allocation matches the expected split
    using a chi-square goodness-of-fit test, and separately flags deviations
    that exceed a practical tolerance threshold.

    Parameters
    ----------
    df             : pd.DataFrame  —  experiment data
    treatment_col  : column containing group labels
    treatment_val  : label for the treatment arm
    control_val    : label for the control arm
    expected_split : expected fraction of users in the treatment arm (e.g. 0.5)
    alpha          : significance threshold for the chi-square test
    tolerance      : maximum allowed absolute deviation from expected_split

    Returns
    -------
    list of Finding dicts:
      SRM           — chi-square test result (fail if p < alpha)
      SRM_TOLERANCE — practical tolerance check (warn if deviation > tolerance)
    """
    findings: List[Dict[str, Any]] = []
    counts = df[treatment_col].value_counts()

    # ── Guard: expected group labels present ──────────────────────────────────
    if treatment_val not in counts or control_val not in counts:
        return [fail(
            "SRM_DATA",
            f"Expected groups '{treatment_val}' and '{control_val}' not found "
            f"in column '{treatment_col}'.",
            detail=counts.to_dict(),
        )]

    n_trt   = int(counts[treatment_val])
    n_ctrl  = int(counts[control_val])
    n_total = n_trt + n_ctrl

    observed = np.array([n_trt, n_ctrl], dtype=float)
    expected = np.array([expected_split, 1.0 - expected_split]) * n_total

    chi2, pval   = chisquare(f_obs=observed, f_exp=expected)
    actual_split = n_trt / n_total
    deviation    = abs(actual_split - expected_split)

    # ── Finding 1: chi-square test ────────────────────────────────────────────
    if pval < alpha:
        findings.append(fail(
            "SRM",
            f"Sample Ratio Mismatch detected (χ²={chi2:.2f}, p={pval:.4f}). "
            f"Observed {actual_split:.1%} treatment vs expected {expected_split:.1%}. "
            f"Downstream effect estimates are unreliable.",
            stat=chi2, pvalue=pval,
            detail=dict(
                n_treatment    = n_trt,
                n_control      = n_ctrl,
                observed_split = round(actual_split, 5),
                expected_split = expected_split,
                n_total        = n_total,
            ),
        ))
    else:
        findings.append(ok(
            "SRM",
            f"No SRM detected (χ²={chi2:.2f}, p={pval:.4f}). "
            f"Observed split {actual_split:.1%} vs expected {expected_split:.1%}.",
            stat=chi2, pvalue=pval,
        ))

    # ── Finding 2: practical tolerance ───────────────────────────────────────
    if deviation > tolerance:
        findings.append(warn(
            "SRM_TOLERANCE",
            f"Split deviation {deviation:.2%} exceeds tolerance {tolerance:.2%}. "
            f"Even without statistical significance this warrants investigation.",
            detail=dict(deviation=round(deviation, 5), tolerance=tolerance),
        ))
    else:
        findings.append(ok(
            "SRM_TOLERANCE",
            f"Split deviation {deviation:.2%} is within tolerance {tolerance:.2%}.",
        ))

    return findings
