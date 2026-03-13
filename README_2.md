# A/B Experiment Diagnostics Agent

An agentic workflow that autonomously diagnoses A/B experiments using the Anthropic API. Point it at a CSV, get back a structured Markdown report with a `PASS / WARN / FAIL` verdict.

---

## How it works

The agent runs a single agentic loop with access to six diagnostic tools. It decides which tools to call, in what order, and when it has enough information to write the final report.

```
User
 │
 ▼
Agent (one Claude instance, one loop)
 │
 ├── check_srm()                    ← always runs first
 │     if SRM detected → stop, write FAIL report
 │
 ├── check_covariate_balance()      ← verify randomisation
 │
 ├── estimate_effect_continuous()   ← one call per continuous metric
 ├── estimate_effect_binary()       ← one call per binary metric
 │
 ├── bootstrap_ci()                 ← cross-validate parametric CIs
 └── bootstrap_pvalue()             ← permutation test
          │
          ▼
     Markdown report (PASS / WARN / FAIL)
```

The loop runs until the agent has called all relevant tools and written the report. On average: **6 tool calls, 1 API call, ~10 seconds**.

---

## Quickstart

```bash
git clone https://github.com/yourname/ab-diagnostics-agent
cd ab-diagnostics-agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python run_agent.py --csv your_experiment.csv
```

---

## Usage

```
python run_agent.py --csv <path> [options]

Required
  --csv PATH              Path to experiment CSV

Experiment setup
  --name TEXT             Experiment name shown in report (default: filename)
  --treatment-col TEXT    Group assignment column         (default: group)
  --treatment-val TEXT    Treatment arm label             (default: treatment)
  --control-val TEXT      Control arm label               (default: control)
  --expected-split FLOAT  Expected treatment fraction     (default: 0.5)
  --alpha FLOAT           Significance level              (default: 0.05)

Metrics (auto-detected if omitted)
  --continuous COL [COL …]   Continuous outcome columns  (e.g. revenue)
  --binary     COL [COL …]   Binary outcome columns      (e.g. converted)

Output
  --output PATH           Save Markdown report to file
  --verbose               Print each tool call as it happens
```

**Examples**

```bash
# Minimal — auto-detects metrics from column names
python run_agent.py --csv data.csv

# Explicit metrics and custom group labels
python run_agent.py \
  --csv data.csv \
  --name "Checkout Redesign Q3" \
  --treatment-col variant \
  --treatment-val B \
  --control-val A \
  --continuous revenue \
  --binary converted activated \
  --output report.md

# Watch every tool call in real time
python run_agent.py --csv data.csv --verbose
```

---

## CSV format

At minimum your CSV needs a group column and one or more outcome columns:

```
user_id, group,     converted, revenue, age, tenure_days, device
0,       control,   0,         7.42,    34,  180,         mobile
1,       treatment, 1,         12.30,   28,  90,          desktop
```

**Metric auto-detection** picks up columns whose names contain any of:
`revenue`, `spend`, `gmv`, `converted`, `clicked`, `activated`, `churned`, `sessions`, `orders`

Continuous vs binary is inferred from cardinality — columns with ≤ 2 unique values in {0, 1} are treated as binary.

---

## Output

```
🤖 Running diagnostics: Checkout Redesign Q3
   N=10,000  continuous=['revenue']  binary=['converted']

  → check_srm()
  → check_covariate_balance()
  → estimate_effect_continuous(metric_col=revenue)
  → estimate_effect_binary(metric_col=converted)
  → bootstrap_ci(metric_col=revenue)
  → bootstrap_pvalue(metric_col=revenue)

# A/B Experiment Diagnostic Report: Checkout Redesign Q3

## Overall Verdict
**PASS** — All checks passed and both metrics show significant lift.

## Executive Summary
The experiment ran cleanly with no SRM (χ²=0.12, p=0.73) and balanced
pre-experiment covariates across all arms. Revenue showed a significant
ATE of +$2.33 (95% CI [1.50, 3.17], p<0.001, +57.7% relative lift).
Conversion lifted by +3.7pp (p<0.001). Bootstrap CIs agree with
parametric results. Recommend shipping the change.

...
```

---

## Project structure

```
ab_diagnostics/        Diagnostics package — the tools the agent calls
│  config.py           CONFIG dict, make_config(), Finding helpers
│  srm.py              check_srm()
│  balance.py          check_covariate_balance()
│  effects.py          estimate_effect_continuous/binary/nonparametric()
│  bootstrap.py        bootstrap_ci(), bootstrap_pvalue()
│  pipeline.py         run_ab_diagnostics(), print_ab_report()

single_agent.py        Agent: agentic loop, tool schemas, tool executor
run_agent.py           CLI entry point
agent_walkthrough.ipynb    Portfolio notebook — step-by-step walkthrough
comparison_notebook.ipynb  Single-agent vs multi-agent comparison
tests/                 Unit tests for the diagnostics package
requirements.txt
```

---

## The agentic loop

The core pattern in `single_agent.py`:

```python
messages = [{"role": "user", "content": user_message}]

while True:
    response = client.messages.create(
        model=MODEL, tools=ALL_TOOLS, messages=messages
    )

    # Agent finished — return the report
    if response.stop_reason == "end_turn":
        return response.content[0].text

    # Agent wants to call a tool
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result,
            })

    # Feed results back and loop
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user",      "content": tool_results})
```

The agent calls tools until it decides it has enough information, then returns a text response. The loop terminates on `stop_reason == "end_turn"`.

---

## Requirements

- Python 3.8+
- `anthropic >= 0.20`
- `numpy`, `pandas`, `scipy`

```bash
pip install -r requirements.txt
```
