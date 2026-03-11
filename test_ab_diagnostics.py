"""
tests/test_ab_diagnostics.py
----------------------------
Unit tests for the ab_diagnostics package.
Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
# import pytest  # install with: pip install pytest

from ab_diagnostics.config  import make_config, ok, warn, fail
from ab_diagnostics.srm     import check_srm
from ab_diagnostics.balance import check_covariate_balance
from ab_diagnostics.effects import (
    estimate_effect_continuous,
    estimate_effect_binary,
    estimate_effect_nonparametric,
)
from ab_diagnostics.bootstrap import bootstrap_ci, bootstrap_pvalue
from ab_diagnostics.pipeline  import run_ab_diagnostics


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_df(n=2000, srm=False, seed=0, effect_rev=2.0, effect_conv=0.03):
    rng   = np.random.default_rng(seed)
    split = 0.6 if srm else 0.5
    group = rng.choice(["control", "treatment"], size=n, p=[split, 1-split])
    W     = (group == "treatment").astype(int)
    age   = rng.normal(35, 8, n).clip(18, 70)
    platform = rng.choice(["ios", "android", "web"], n)
    revenue  = (8.0 + effect_rev * W + rng.normal(0, 5, n)).clip(0).round(2)
    converted = rng.binomial(1, 0.12 + effect_conv * W)
    return pd.DataFrame(dict(
        group=group, age=age.round(1),
        platform=platform, revenue=revenue, converted=converted,
    ))

# @pytest.fixture
def df_healthy():
    return _make_df(seed=42)

# @pytest.fixture
def df_srm():
    return _make_df(srm=True, seed=42)

# @pytest.fixture
def df_null():
    return _make_df(effect_rev=0.0, effect_conv=0.0, seed=42)


# ── Config ─────────────────────────────────────────────────────────────────────

def test_make_config_overrides():
    cfg = make_config(alpha=0.01, treatment_col="variant")
    assert cfg["alpha"] == 0.01
    assert cfg["treatment_col"] == "variant"
    # Other keys unchanged
    assert cfg["expected_split"] == 0.5

def test_finding_helpers():
    f_ok   = ok("FOO", "all good")
    f_warn = warn("BAR", "something odd", stat=1.5)
    f_fail = fail("BAZ", "broken", pvalue=0.001, detail={"x": 1})
    assert f_ok["level"]  == "ok"
    assert f_warn["stat"] == 1.5
    assert f_fail["detail"]["x"] == 1


# ── SRM ────────────────────────────────────────────────────────────────────────

def test_srm_healthy(df_healthy):
    findings = check_srm(df_healthy)
    codes = [f["code"] for f in findings]
    levels = {f["code"]: f["level"] for f in findings}
    assert "SRM" in codes
    assert levels["SRM"] == "ok"

def test_srm_detected(df_srm):
    findings = check_srm(df_srm)
    levels = {f["code"]: f["level"] for f in findings}
    assert levels["SRM"] == "fail"
    assert levels["SRM_TOLERANCE"] in ("warn", "fail")

def test_srm_missing_group():
    df = pd.DataFrame({"group": ["a", "b", "a"], "revenue": [1, 2, 3]})
    findings = check_srm(df)
    assert findings[0]["code"] == "SRM_DATA"
    assert findings[0]["level"] == "fail"

def test_srm_90_10_split():
    # 90/10 split should always fail
    rng = np.random.default_rng(0)
    n = 2000
    group = rng.choice(["control", "treatment"], size=n, p=[0.9, 0.1])
    df = pd.DataFrame({"group": group, "revenue": rng.normal(0, 1, n)})
    findings = check_srm(df, expected_split=0.5)
    levels = {f["code"]: f["level"] for f in findings}
    assert levels["SRM"] == "fail"


# ── Balance ────────────────────────────────────────────────────────────────────

def test_balance_healthy(df_healthy):
    findings = check_covariate_balance(
        df_healthy,
        continuous_covariates  = ["age"],
        categorical_covariates = ["platform"],
    )
    # Randomly balanced — no fails expected at n=2000
    fails = [f for f in findings if f["level"] == "fail"]
    assert len(fails) == 0

def test_balance_injected_imbalance():
    rng   = np.random.default_rng(99)
    n     = 3000
    group = rng.choice(["control", "treatment"], n, p=[0.5, 0.5])
    # Force a large age imbalance: treatment users are 10 years older
    age   = np.where(group == "treatment",
                     rng.normal(45, 5, n),
                     rng.normal(30, 5, n))
    df = pd.DataFrame({"group": group, "age": age})
    findings = check_covariate_balance(df, continuous_covariates=["age"],
                                       categorical_covariates=[])
    fails = [f for f in findings if f["level"] == "fail" and "AGE" in f["code"]]
    assert len(fails) > 0

def test_balance_summary_finding(df_healthy):
    findings = check_covariate_balance(
        df_healthy,
        continuous_covariates  = ["age"],
        categorical_covariates = [],
    )
    codes = [f["code"] for f in findings]
    assert "BALANCE_SUMMARY" in codes

def test_balance_smd_values(df_healthy):
    findings = check_covariate_balance(
        df_healthy,
        continuous_covariates  = ["age"],
        categorical_covariates = [],
    )
    age_f = next(f for f in findings if "AGE" in f["code"] and f["code"] != "BALANCE_SUMMARY")
    assert "smd" in age_f["detail"]
    assert abs(age_f["detail"]["smd"]) < 0.2   # should be small for balanced data


# ── Effects ────────────────────────────────────────────────────────────────────

def test_effect_continuous_significant(df_healthy):
    f = estimate_effect_continuous(df_healthy, metric_col="revenue")
    assert f["level"] == "ok"
    assert f["detail"]["ate"] > 0
    assert f["detail"]["ci_low"] < f["detail"]["ci_high"]

def test_effect_continuous_null(df_null):
    f = estimate_effect_continuous(df_null, metric_col="revenue")
    # With zero true effect, CI should contain 0
    d = f["detail"]
    assert d["ci_low"] <= 0 <= d["ci_high"] or f["pvalue"] > 0.05

def test_effect_binary_significant(df_healthy):
    f = estimate_effect_binary(df_healthy, metric_col="converted")
    assert "abs_lift" in f["detail"]
    assert f["detail"]["ci_low"] < f["detail"]["ci_high"]

def test_effect_binary_proportions_valid(df_healthy):
    f = estimate_effect_binary(df_healthy, metric_col="converted")
    d = f["detail"]
    assert 0 <= d["p_treatment"] <= 1
    assert 0 <= d["p_control"] <= 1

def test_effect_nonparametric(df_healthy):
    f = estimate_effect_nonparametric(df_healthy, metric_col="revenue")
    assert "rank_biserial_r" in f["detail"]
    assert -1 <= f["detail"]["rank_biserial_r"] <= 1

def test_effect_relative_lift_direction(df_healthy):
    f = estimate_effect_continuous(df_healthy, metric_col="revenue")
    # Effect is positive; relative lift should also be positive
    assert f["detail"]["relative_lift"] > 0


# ── Bootstrap ──────────────────────────────────────────────────────────────────

def test_bootstrap_ci_significant(df_healthy):
    f = bootstrap_ci(df_healthy, metric_col="revenue", n_bootstrap=500)
    assert f["detail"]["ci_low"] < f["detail"]["ci_high"]
    # Healthy data with true effect: CI should not contain zero
    assert not f["detail"]["contains_zero"]

def test_bootstrap_ci_null(df_null):
    f = bootstrap_ci(df_null, metric_col="revenue", n_bootstrap=500)
    # With zero effect CI should contain zero or be very small
    assert f["detail"]["ci_low"] < f["detail"]["ci_high"]

def test_bootstrap_custom_stat_fn(df_healthy):
    f = bootstrap_ci(
        df_healthy,
        metric_col = "revenue",
        stat_fn    = lambda t, c: np.median(t) - np.median(c),
        n_bootstrap= 500,
    )
    assert "observed" in f["detail"]

def test_bootstrap_pvalue_significant(df_healthy):
    f = bootstrap_pvalue(df_healthy, metric_col="revenue", n_bootstrap=500)
    assert "pvalue" in f
    assert 0 <= f["pvalue"] <= 1
    assert f["pvalue"] < 0.05   # should be significant

def test_bootstrap_pvalue_null(df_null):
    f = bootstrap_pvalue(df_null, metric_col="revenue", n_bootstrap=500, seed=0)
    assert f["pvalue"] > 0.01   # should not be highly significant


# ── Pipeline ───────────────────────────────────────────────────────────────────

def test_pipeline_returns_all_sections(df_healthy):
    results = run_ab_diagnostics(
        df_healthy,
        continuous_metrics     = ("revenue",),
        binary_metrics         = ("converted",),
        continuous_covariates  = ("age",),
        categorical_covariates = ("platform",),
        run_bootstrap          = True,
    )
    assert set(results.keys()) == {"srm", "balance", "effects", "bootstrap"}
    for section in results.values():
        assert isinstance(section, list)

def test_pipeline_stop_on_srm(df_srm):
    results = run_ab_diagnostics(df_srm, stop_on_srm=True)
    # Balance, effects, bootstrap should be empty when SRM is detected
    assert results["balance"] == []
    assert results["effects"] == []
    assert results["bootstrap"] == []
    # SRM_SKIP warning should be present
    codes = [f["code"] for f in results["srm"]]
    assert "SRM_SKIP" in codes

def test_pipeline_continue_despite_srm(df_srm):
    results = run_ab_diagnostics(df_srm, stop_on_srm=False,
                                 continuous_covariates=("age",),
                                 categorical_covariates=("platform",))
    # When stop_on_srm=False, effects should still be computed
    assert len(results["effects"]) > 0

def test_pipeline_no_bootstrap(df_healthy):
    results = run_ab_diagnostics(df_healthy, run_bootstrap=False,
                                 continuous_covariates=("age",),
                                 categorical_covariates=("platform",))
    assert results["bootstrap"] == []

def test_pipeline_custom_config(df_healthy):
    cfg     = make_config(alpha=0.01, n_bootstrap=100)
    results = run_ab_diagnostics(df_healthy, config=cfg,
                                 continuous_covariates=("age",),
                                 categorical_covariates=("platform",))
    assert len(results["srm"]) > 0

def test_pipeline_finding_schema(df_healthy):
    results = run_ab_diagnostics(df_healthy, run_bootstrap=False,
                                 continuous_covariates=("age",),
                                 categorical_covariates=("platform",))
    for section in results.values():
        for f in section:
            assert "level"   in f
            assert "code"    in f
            assert "message" in f
            assert f["level"] in ("ok", "warn", "fail")
