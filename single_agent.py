"""
single_agent.py
---------------
Single-agent A/B diagnostics — one Claude instance, all 6 tools.

The agent runs one agentic loop with access to every diagnostic tool.
It decides autonomously:
  - which tools to call and in what order
  - whether to skip effects if it detects SRM
  - how many times to call each tool
  - when it has enough information to write the final report

Compare with agents.py (multi-agent) to understand the tradeoffs.
"""

from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import anthropic

from ab_diagnostics import (
    check_srm,
    check_covariate_balance,
    estimate_effect_continuous,
    estimate_effect_binary,
    bootstrap_ci,
    bootstrap_pvalue,
    make_config,
)

client = anthropic.Anthropic()
MODEL  = "claude-sonnet-4-20250514"


# ═════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SingleAgentReport:
    """Output from the single-agent run."""
    experiment_name: str
    verdict:         str          # "PASS" | "WARN" | "FAIL"
    markdown:        str          # full Markdown report
    findings:        list[dict]   # all raw Finding dicts accumulated
    tool_calls:      list[dict]   # log of every tool call made
    n_api_calls:     int = 1      # always 1 for single agent
    total_ok:        int = 0
    total_warn:      int = 0
    total_fail:      int = 0


# ═════════════════════════════════════════════════════════════════════════════
# Tool definitions — all 6 available to the single agent
# ═════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    {
        "name": "check_srm",
        "description": (
            "Check for Sample Ratio Mismatch. Run this FIRST before any other check. "
            "If SRM is detected (level=fail), do NOT call effect or bootstrap tools — "
            "the estimates would be meaningless. Return your report immediately after "
            "noting the SRM."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expected_split": {"type": "number", "description": "Expected treatment fraction (e.g. 0.5)."},
                "tolerance":      {"type": "number", "description": "Allowed split deviation (e.g. 0.01)."},
            },
            "required": [],
        },
    },
    {
        "name": "check_covariate_balance",
        "description": (
            "Check pre-experiment covariate balance between arms. "
            "Run after SRM check passes. Pass empty lists for auto-detection."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "continuous_covariates":  {"type": "array", "items": {"type": "string"},
                                           "description": "Continuous covariate columns. Empty = auto-detect."},
                "categorical_covariates": {"type": "array", "items": {"type": "string"},
                                           "description": "Categorical covariate columns. Empty = auto-detect."},
                "smd_threshold":          {"type": "number", "description": "SMD threshold (default 0.1)."},
            },
            "required": [],
        },
    },
    {
        "name": "estimate_effect_continuous",
        "description": "Estimate ATE on a continuous metric (Welch's t-test). Returns ATE, 95% CI, relative lift.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_col": {"type": "string", "description": "Column name of the continuous outcome."},
            },
            "required": ["metric_col"],
        },
    },
    {
        "name": "estimate_effect_binary",
        "description": "Estimate ATE on a binary metric (two-proportion z-test). Returns abs/rel lift, 95% CI.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_col": {"type": "string", "description": "Column name of the binary outcome (0/1)."},
            },
            "required": ["metric_col"],
        },
    },
    {
        "name": "bootstrap_ci",
        "description": (
            "Percentile bootstrap CI for mean difference. "
            "Use to cross-validate parametric CIs, especially for skewed metrics like revenue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_col":  {"type": "string"},
                "n_bootstrap": {"type": "integer", "description": "Resamples (default 2000)."},
                "seed":        {"type": "integer"},
            },
            "required": ["metric_col"],
        },
    },
    {
        "name": "bootstrap_pvalue",
        "description": "Permutation p-value under the null of no effect. Model-free cross-check.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_col":  {"type": "string"},
                "n_bootstrap": {"type": "integer"},
                "seed":        {"type": "integer"},
            },
            "required": ["metric_col"],
        },
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# Tool executor
# ═════════════════════════════════════════════════════════════════════════════

def _make_executor(df: pd.DataFrame, cfg: dict) -> Callable:
    kw = dict(
        treatment_col = cfg["treatment_col"],
        treatment_val = cfg["treatment_val"],
        control_val   = cfg["control_val"],
        alpha         = cfg["alpha"],
    )
    def execute(name: str, inp: dict) -> str:
        try:
            if name == "check_srm":
                result = check_srm(df,
                    expected_split = float(inp.get("expected_split", cfg["expected_split"])),
                    tolerance      = float(inp.get("tolerance", cfg["srm_tolerance"])),
                    **kw)
            elif name == "check_covariate_balance":
                result = check_covariate_balance(df,
                    continuous_covariates  = inp.get("continuous_covariates") or None,
                    categorical_covariates = inp.get("categorical_covariates") or None,
                    smd_threshold          = float(inp.get("smd_threshold", 0.1)),
                    **kw)
            elif name == "estimate_effect_continuous":
                result = [estimate_effect_continuous(df, metric_col=inp["metric_col"], **kw)]
            elif name == "estimate_effect_binary":
                result = [estimate_effect_binary(df, metric_col=inp["metric_col"], **kw)]
            elif name == "bootstrap_ci":
                result = [bootstrap_ci(df,
                    metric_col  = inp["metric_col"],
                    n_bootstrap = int(inp.get("n_bootstrap", cfg["n_bootstrap"])),
                    seed        = int(inp.get("seed", cfg["bootstrap_seed"])),
                    **kw)]
            elif name == "bootstrap_pvalue":
                result = [bootstrap_pvalue(df,
                    metric_col  = inp["metric_col"],
                    n_bootstrap = int(inp.get("n_bootstrap", cfg["n_bootstrap"])),
                    seed        = int(inp.get("seed", cfg["bootstrap_seed"])),
                    **kw)]
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})
    return execute


# ═════════════════════════════════════════════════════════════════════════════
# Single-agent runner
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert A/B experiment analyst.
    You have access to six diagnostic tools. Use them to fully diagnose the experiment.

    Mandatory sequence:
    1. check_srm FIRST — if SRM is detected (any finding with level=fail and code=SRM),
       stop immediately and write a FAIL report. Do NOT call any other tools.
    2. check_covariate_balance — verify randomisation worked.
    3. estimate_effect_continuous and/or estimate_effect_binary — for each outcome metric.
    4. bootstrap_ci and bootstrap_pvalue — for each continuous metric, to validate
       parametric results.

    After running all appropriate tools, write a Markdown report with this structure:

    # A/B Experiment Diagnostic Report: {experiment_name}

    ## Overall Verdict
    **PASS** | **WARN** | **FAIL**
    (one sentence reason)

    ## Executive Summary
    3-5 sentences on what happened and whether to ship.

    ## Findings by Module
    ### SRM Check
    ### Covariate Balance
    ### Treatment Effects
    ### Bootstrap Validation

    ## Recommendation

    Rules:
    - FAIL: any SRM or balance failure, or no metric is significant
    - WARN: bootstrap disagrees with parametric, or warn-level findings
    - PASS: all checks pass and at least one metric significant
    - Cite actual numbers (ATEs, p-values, CIs) — do not fabricate
""")


def run_single_agent(
    df:              pd.DataFrame,
    experiment_name: str   = "Experiment",
    continuous_metrics: list = None,
    binary_metrics:     list = None,
    config_overrides:   dict = None,
    verbose:            bool = False,
) -> SingleAgentReport:
    """
    Run the single-agent A/B diagnostic pipeline.

    One Claude instance with all 6 tools. The agent decides the order,
    handles the SRM gate, and writes the final report — all in one loop.

    Parameters
    ----------
    df                 : experiment DataFrame
    experiment_name    : name shown in the report
    continuous_metrics : continuous outcome columns (None = auto-detect)
    binary_metrics     : binary outcome columns (None = auto-detect)
    config_overrides   : dict passed to make_config()
    verbose            : print tool calls as they happen

    Returns
    -------
    SingleAgentReport
    """
    cfg = make_config(**(config_overrides or {}))

    # ── Auto-detect metrics ────────────────────────────────────────────────────
    hints = ("revenue","spend","gmv","converted","clicked","activated",
             "churned","sessions","orders")
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if continuous_metrics is None:
        continuous_metrics = [
            c for c in num_cols
            if any(h in c.lower() for h in hints) and df[c].nunique() > 2
        ]
    if binary_metrics is None:
        binary_metrics = [
            c for c in num_cols
            if any(h in c.lower() for h in hints)
            and df[c].nunique() <= 2
            and set(df[c].dropna().unique()).issubset({0, 1})
        ]

    # ── Build column info for context ─────────────────────────────────────────
    col_info = "\n".join(
        f"  {c}: {df[c].dtype}, {df[c].nunique()} unique values"
        for c in df.columns
    )
    n_trt  = (df[cfg["treatment_col"]] == cfg["treatment_val"]).sum()
    n_ctrl = (df[cfg["treatment_col"]] == cfg["control_val"]).sum()

    user_message = (
        f"Experiment: {experiment_name}\n"
        f"N={len(df):,} users: {n_trt} treatment, {n_ctrl} control\n"
        f"Expected split: {cfg['expected_split']:.0%} / {1-cfg['expected_split']:.0%}\n"
        f"Continuous metrics: {continuous_metrics}\n"
        f"Binary metrics: {binary_metrics}\n\n"
        f"Column schema:\n{col_info}\n\n"
        f"Run a full diagnostic and write the Markdown report."
    )

    execute     = _make_executor(df, cfg)
    messages    = [{"role": "user", "content": user_message}]
    all_findings: list[dict] = []
    tool_log:     list[dict] = []

    print(f"\n🤖 Single-agent diagnostics: {experiment_name}")
    print(f"   N={len(df):,}  continuous={continuous_metrics}  binary={binary_metrics}\n")

    # ── Agentic loop ──────────────────────────────────────────────────────────
    while True:
        response = client.messages.create(
            model      = MODEL,
            max_tokens = 2048,
            system     = SYSTEM_PROMPT.replace("{experiment_name}", experiment_name),
            tools      = ALL_TOOLS,
            messages   = messages,
        )

        if verbose:
            print(f"  stop_reason={response.stop_reason}")

        # Collect text
        text_blocks = [b.text for b in response.content if b.type == "text"]

        # Done — extract final report
        if response.stop_reason == "end_turn" or not any(
            b.type == "tool_use" for b in response.content
        ):
            final_text = "\n".join(text_blocks)
            break

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if verbose:
                print(f"  → {block.name}({json.dumps(block.input)[:80]})")
            else:
                print(f"  → {block.name}({', '.join(f'{k}={v}' for k,v in block.input.items())})")

            result_str = execute(block.name, block.input)
            tool_log.append({"tool": block.name, "input": block.input, "result": result_str})

            try:
                parsed = json.loads(result_str)
                if isinstance(parsed, list):
                    all_findings.extend(parsed)
            except json.JSONDecodeError:
                pass

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result_str,
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})

    # ── Count finding levels ───────────────────────────────────────────────────
    level_counts = {"ok": 0, "warn": 0, "fail": 0}
    for f in all_findings:
        lvl = f.get("level", "")
        if lvl in level_counts:
            level_counts[lvl] += 1

    # ── Extract verdict ────────────────────────────────────────────────────────
    verdict = "WARN"
    for line in final_text.splitlines():
        if "**PASS**" in line: verdict = "PASS"; break
        if "**FAIL**" in line: verdict = "FAIL"; break
        if "**WARN**" in line: verdict = "WARN"; break

    print(f"  Done. {len(tool_log)} tool calls. Verdict: {verdict}\n")

    return SingleAgentReport(
        experiment_name = experiment_name,
        verdict         = verdict,
        markdown        = final_text,
        findings        = all_findings,
        tool_calls      = tool_log,
        n_api_calls     = 1,
        total_ok        = level_counts["ok"],
        total_warn      = level_counts["warn"],
        total_fail      = level_counts["fail"],
    )
