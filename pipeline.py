"""
pipeline.py
-----------
Full A/B diagnostic pipeline: one call, structured output.

Functions
---------
run_ab_diagnostics(df, ...)   Run all modules in order; returns dict of findings
print_ab_report(results)      Pretty-print the findings dict to stdout
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config   import CONFIG, warn
from .srm      import check_srm
from .balance  import check_covariate_balance
from .effects  import (
    estimate_effect_continuous,
    estimate_effect_binary,
)
from .bootstrap import bootstrap_ci, bootstrap_pvalue


# ── ANSI colour codes (degrade gracefully in plain terminals) ──────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"

_ICONS = {"ok": "✔", "warn": "⚠", "fail": "✘"}
_COLORS = {"ok": _GREEN, "warn": _YELLOW, "fail": _RED}


def run_ab_diagnostics(
    df,
    continuous_metrics:     Sequence[str] = ("revenue",),
    binary_metrics:         Sequence[str] = ("converted",),
    continuous_covariates:  Sequence[str] = ("age", "tenure_days"),
    categorical_covariates: Sequence[str] = ("platform",),
    run_bootstrap:          bool          = True,
    stop_on_srm:            bool          = True,
    config:                 Dict[str, Any] = CONFIG,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run the full A/B diagnostic pipeline.

    Execution order
    ---------------
    1. SRM check            → if SRM detected and stop_on_srm=True, stop here
    2. Covariate balance    → flag pre-experiment imbalances
    3. Treatment effects    → ATE for each metric
    4. Bootstrap CIs        → non-parametric CIs for each continuous metric

    Parameters
    ----------
    df                     : experiment DataFrame
    continuous_metrics     : continuous outcome columns to estimate effects for
    binary_metrics         : binary (0/1) outcome columns to estimate effects for
    continuous_covariates  : continuous covariate columns for balance check
    categorical_covariates : categorical covariate columns for balance check
    run_bootstrap          : whether to run bootstrap / permutation tests
    stop_on_srm            : if True, skip all downstream checks when SRM detected
    config                 : config dict (use make_config() to customise)

    Returns
    -------
    dict with keys: "srm", "balance", "effects", "bootstrap"
    Each value is a list of Finding dicts.
    """
    results: Dict[str, List] = dict(srm=[], balance=[], effects=[], bootstrap=[])

    tc = config["treatment_col"]
    tv = config["treatment_val"]
    cv = config["control_val"]

    # ── 1. SRM ─────────────────────────────────────────────────────────────────
    results["srm"] = check_srm(
        df,
        treatment_col   = tc,
        treatment_val   = tv,
        control_val     = cv,
        expected_split  = config["expected_split"],
        alpha           = config["alpha"],
        tolerance       = config["srm_tolerance"],
    )

    srm_detected = any(
        f["code"] == "SRM" and f["level"] == "fail"
        for f in results["srm"]
    )
    if srm_detected and stop_on_srm:
        results["srm"].append(warn(
            "SRM_SKIP",
            "All downstream checks skipped — SRM detected. "
            "Fix the randomisation bug before interpreting results.",
        ))
        return results

    # ── 2. Covariate balance ───────────────────────────────────────────────────
    results["balance"] = check_covariate_balance(
        df,
        continuous_covariates  = list(continuous_covariates),
        categorical_covariates = list(categorical_covariates),
        treatment_col = tc,
        treatment_val = tv,
        control_val   = cv,
        alpha         = config["alpha"],
    )

    # ── 3. Treatment effects ───────────────────────────────────────────────────
    shared_kw = dict(treatment_col=tc, treatment_val=tv,
                     control_val=cv, alpha=config["alpha"])
    for col in continuous_metrics:
        results["effects"].append(
            estimate_effect_continuous(df, metric_col=col, **shared_kw))
    for col in binary_metrics:
        results["effects"].append(
            estimate_effect_binary(df, metric_col=col, **shared_kw))

    # ── 4. Bootstrap ──────────────────────────────────────────────────────────
    if run_bootstrap:
        boot_kw = dict(
            treatment_col = tc,
            treatment_val = tv,
            control_val   = cv,
            alpha         = config["alpha"],
            n_bootstrap   = config["n_bootstrap"],
            seed          = config["bootstrap_seed"],
        )
        for col in continuous_metrics:
            results["bootstrap"].append(bootstrap_ci(df, metric_col=col, **boot_kw))
            results["bootstrap"].append(bootstrap_pvalue(df, metric_col=col, **boot_kw))

    return results


def print_ab_report(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Pretty-print findings from run_ab_diagnostics() to stdout.

    Uses ANSI colour codes (✔ green / ⚠ yellow / ✘ red).
    """
    SECTION_LABELS = {
        "srm":       "1 · Sample Ratio Mismatch",
        "balance":   "2 · Covariate Balance",
        "effects":   "3 · Treatment Effects",
        "bootstrap": "4 · Bootstrap Inference",
    }
    sep  = "─" * 66
    wide = "═" * 66

    print(f"\n{_BOLD}{_CYAN}{wide}{_RESET}")
    print(f"{_BOLD}  A/B Experiment Diagnostic Report{_RESET}")
    print(f"{_BOLD}{_CYAN}{wide}{_RESET}")

    total: Dict[str, int] = {"ok": 0, "warn": 0, "fail": 0}

    for section, label in SECTION_LABELS.items():
        findings = results.get(section, [])
        if not findings:
            continue

        pad = 48 - len(label)
        print(f"\n  {_DIM}── {label} {'─'*pad}{_RESET}")

        for f in findings:
            level  = f["level"]
            icon   = _ICONS[level]
            color  = _COLORS[level]
            code   = f["code"]
            msg    = f["message"]
            print(f"  {color}{icon}{_RESET} {_DIM}[{code}]{_RESET} {msg}")

            # Print key detail fields inline for effect findings
            d = f.get("detail", {})
            if "ate" in d:
                print(
                    f"    {_DIM}↳ ATE={d['ate']:+.4f}, "
                    f"CI=[{d['ci_low']:.4f}, {d['ci_high']:.4f}], "
                    f"rel_lift={d['relative_lift']:+.1%}{_RESET}"
                )
            elif "abs_lift" in d:
                print(
                    f"    {_DIM}↳ abs_lift={d['abs_lift']:+.4f}, "
                    f"CI=[{d['ci_low']:.4f}, {d['ci_high']:.4f}], "
                    f"rel_lift={d['rel_lift']:+.1%}{_RESET}"
                )

            total[level] += 1

    print(f"\n{_CYAN}{sep}{_RESET}")
    print(
        f"  {_BOLD}Summary:{_RESET}  "
        f"{_GREEN}✔ {total['ok']} ok{_RESET}  "
        f"{_YELLOW}⚠ {total['warn']} warn{_RESET}  "
        f"{_RED}✘ {total['fail']} fail{_RESET}"
    )
    print(f"{_BOLD}{_CYAN}{wide}{_RESET}\n")
