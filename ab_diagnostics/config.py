"""
config.py
---------
Global configuration dict and Finding constructor helpers.

All public functions in this package accept a `config` kwarg that
overrides any key in the default CONFIG.  Use make_config() to
create a clean copy with your own values.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

# ── Default config ─────────────────────────────────────────────────────────────
CONFIG: Dict[str, Any] = dict(
    treatment_col   = "group",        # DataFrame column name for group assignment
    treatment_val   = "treatment",    # value indicating the treatment arm
    control_val     = "control",      # value indicating the control arm
    alpha           = 0.05,           # significance threshold
    srm_tolerance   = 0.01,           # max allowed split deviation (1 pp)
    expected_split  = 0.50,           # expected fraction of users in treatment
    n_bootstrap     = 2000,           # bootstrap / permutation resamples
    bootstrap_seed  = 42,
)


def make_config(**overrides: Any) -> Dict[str, Any]:
    """
    Return a copy of CONFIG with any keys overridden.

    Example
    -------
    cfg = make_config(treatment_col='variant', expected_split=0.1)
    results = run_ab_diagnostics(df, config=cfg)
    """
    cfg = dict(CONFIG)
    cfg.update(overrides)
    return cfg


# ── Finding helpers ────────────────────────────────────────────────────────────
# Every diagnostic function returns one or more Finding dicts.
#
# Schema
# ------
# level   : "ok" | "warn" | "fail"
# code    : short identifier string, e.g. "SRM", "BALANCE_AGE"
# message : human-readable description of the finding
# stat    : (optional) test statistic
# pvalue  : (optional) p-value
# detail  : (optional) any extra data (dicts, lists, etc.)

def _f(
    level:   str,
    code:    str,
    message: str,
    stat:    Optional[float] = None,
    pvalue:  Optional[float] = None,
    detail:  Any             = None,
) -> Dict[str, Any]:
    f: Dict[str, Any] = dict(level=level, code=code, message=message)
    if stat   is not None: f["stat"]   = round(float(stat),   6)
    if pvalue is not None: f["pvalue"] = round(float(pvalue), 6)
    if detail is not None: f["detail"] = detail
    return f


def ok(code: str, message: str, **kw: Any) -> Dict[str, Any]:
    """Create an OK-level finding."""
    return _f("ok", code, message, **kw)


def warn(code: str, message: str, **kw: Any) -> Dict[str, Any]:
    """Create a WARN-level finding."""
    return _f("warn", code, message, **kw)


def fail(code: str, message: str, **kw: Any) -> Dict[str, Any]:
    """Create a FAIL-level finding."""
    return _f("fail", code, message, **kw)
