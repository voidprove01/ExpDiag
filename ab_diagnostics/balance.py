"""
balance.py
----------
Pre-experiment covariate balance diagnostics.

Imbalanced covariates suggest randomisation failure or biased post-hoc
segmentation.  Check this before trusting any effect estimates.

Functions
---------
check_covariate_balance(df, ...)  ->  list[Finding]
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

from .config import CONFIG, ok, warn, fail


def check_covariate_balance(
    df,
    continuous_covariates:  Optional[Sequence[str]] = None,
    categorical_covariates: Optional[Sequence[str]] = None,
    treatment_col: str   = CONFIG["treatment_col"],
    treatment_val: str   = CONFIG["treatment_val"],
    control_val:   str   = CONFIG["control_val"],
    alpha:         float = CONFIG["alpha"],
    smd_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Check pre-experiment covariate balance between treatment and control.

    For each continuous covariate: Welch's t-test + standardised mean
    difference (SMD).  For each categorical covariate: chi-square test of
    independence.

    A finding is set to FAIL when both p < alpha AND |SMD| > smd_threshold.
    A finding is WARN when only the SMD threshold is breached.
    Otherwise OK.

    Parameters
    ----------
    df                     : experiment DataFrame
    continuous_covariates  : list of continuous covariate column names.
                             If None, auto-detected from numeric columns
                             (excludes outcome-like names).
    categorical_covariates : list of categorical covariate column names.
                             If None, auto-detected from object columns.
    treatment_col          : group assignment column
    treatment_val          : treatment arm label
    control_val            : control arm label
    alpha                  : significance level
    smd_threshold          : |SMD| above this value triggers a warning

    Returns
    -------
    list of Finding dicts, one per covariate tested, plus one summary Finding
    """
    trt  = df[df[treatment_col] == treatment_val]
    ctrl = df[df[treatment_col] == control_val]

    exclude = {treatment_col}

    # ── Auto-detect covariates ────────────────────────────────────────────────
    outcome_prefixes = ("convert", "revenue", "outcome", "metric", "click",
                        "purchase", "order", "visit")

    if continuous_covariates is None:
        continuous_covariates = [
            c for c in df.select_dtypes(include="number").columns
            if c not in exclude
            and df[c].nunique() > 10
            and not any(c.lower().startswith(p) for p in outcome_prefixes)
        ]

    if categorical_covariates is None:
        categorical_covariates = [
            c for c in df.select_dtypes(include="object").columns
            if c not in exclude
        ]

    findings: List[Dict[str, Any]] = []
    n_imbalanced = 0

    # ── Continuous covariates ─────────────────────────────────────────────────
    for col in continuous_covariates:
        a = trt[col].dropna().values
        b = ctrl[col].dropna().values

        t_stat, pval = ttest_ind(a, b, equal_var=False)   # Welch's t-test

        pooled_sd = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        smd       = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0

        detail = dict(
            trt_mean  = round(float(a.mean()), 4),
            ctrl_mean = round(float(b.mean()), 4),
            smd       = round(float(smd), 4),
            n_trt     = int(len(a)),
            n_ctrl    = int(len(b)),
        )

        if pval < alpha and abs(smd) > smd_threshold:
            n_imbalanced += 1
            findings.append(fail(
                f"BALANCE_{col.upper()}",
                f"'{col}': significant imbalance "
                f"(t={t_stat:.2f}, p={pval:.4f}, SMD={smd:.3f}).",
                stat=t_stat, pvalue=pval, detail=detail,
            ))
        elif abs(smd) > smd_threshold:
            findings.append(warn(
                f"BALANCE_{col.upper()}",
                f"'{col}': SMD={smd:.3f} exceeds threshold {smd_threshold} "
                f"(p={pval:.4f}).",
                stat=t_stat, pvalue=pval, detail=detail,
            ))
        else:
            findings.append(ok(
                f"BALANCE_{col.upper()}",
                f"'{col}': balanced (SMD={smd:.3f}, p={pval:.4f}).",
                stat=t_stat, pvalue=pval, detail=detail,
            ))

    # ── Categorical covariates ─────────────────────────────────────────────────
    for col in categorical_covariates:
        contingency = pd.crosstab(df[treatment_col], df[col])
        chi2, pval, dof, _ = chi2_contingency(contingency)

        detail = dict(
            crosstab = contingency.to_dict(),
            dof      = int(dof),
        )

        if pval < alpha:
            n_imbalanced += 1
            findings.append(fail(
                f"BALANCE_{col.upper()}",
                f"'{col}': significant imbalance "
                f"(χ²={chi2:.2f}, df={dof}, p={pval:.4f}).",
                stat=chi2, pvalue=pval, detail=detail,
            ))
        else:
            findings.append(ok(
                f"BALANCE_{col.upper()}",
                f"'{col}': balanced (χ²={chi2:.2f}, df={dof}, p={pval:.4f}).",
                stat=chi2, pvalue=pval, detail=detail,
            ))

    # ── Summary finding ───────────────────────────────────────────────────────
    n_tested = len(continuous_covariates) + len(categorical_covariates)
    if n_tested > 0:
        if n_imbalanced == 0:
            findings.append(ok(
                "BALANCE_SUMMARY",
                f"All {n_tested} covariate(s) appear balanced.",
            ))
        else:
            findings.append(fail(
                "BALANCE_SUMMARY",
                f"{n_imbalanced}/{n_tested} covariate(s) show imbalance. "
                f"Investigate before interpreting treatment effects.",
                detail=dict(n_imbalanced=n_imbalanced, n_tested=n_tested),
            ))

    return findings
