# 🤖 Multi-Agent A/B Experiment Diagnostics

A production-style agentic workflow built on the [Anthropic API](https://docs.anthropic.com/) that autonomously diagnoses A/B experiments using a **coordinator + specialist sub-agents** pattern.

---

## Architecture

```
CoordinatorAgent
    │
    ├──► SRMAgent         tool: check_srm
    ├──► BalanceAgent     tool: check_covariate_balance
    ├──► EffectsAgent     tools: estimate_effect_continuous
    │                            estimate_effect_binary
    └──► BootstrapAgent   tools: bootstrap_ci
                                 bootstrap_pvalue
```

Each sub-agent runs an **agentic loop**: it calls its tools, receives structured findings, and returns a natural-language interpretation. The coordinator synthesises all outputs into a `PASS / WARN / FAIL` verdict and a full Markdown report.

---

## Key Concepts Demonstrated

| Concept | Where |
|---|---|
| **Tool use** | Each sub-agent has a schema-defined tool set; Claude decides when and how to call them |
| **Agentic loop** | `_run_agent_loop()` in `agents.py` — runs until `end_turn` |
| **Multi-agent orchestration** | `run_diagnostics_agent()` coordinates 4 specialists |
| **Gate logic** | SRM failure skips EffectsAgent and BootstrapAgent |
| **Structured output** | `DiagnosticReport` dataclass; findings follow a consistent schema |
| **Separation of concerns** | Each agent has one system prompt, one responsibility |

---

## Project Structure

```
ab_diagnostics/            Diagnostics package (tools the agents call)
│   config.py              CONFIG, make_config(), Finding helpers
│   srm.py                 check_srm()
│   balance.py             check_covariate_balance()
│   effects.py             estimate_effect_continuous/binary/nonparametric()
│   bootstrap.py           bootstrap_ci(), bootstrap_pvalue()
│   pipeline.py            run_ab_diagnostics(), print_ab_report()

agents.py                  Multi-agent orchestration engine
single_agent.py            Single-agent implementation (same tools, one loop)
compare.py                 Runs both architectures and prints a comparison table
run_agent.py               CLI entry point (multi-agent)
agent_walkthrough.ipynb    Portfolio notebook — multi-agent walkthrough
comparison_notebook.ipynb  Side-by-side architecture comparison
tests/                     Unit tests for the diagnostics package
requirements.txt
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/yourname/ab-diagnostics-agent
cd ab-diagnostics-agent
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run on your data
python run_agent.py --csv your_experiment.csv

# 4. Full options
python run_agent.py \
    --csv data.csv \
    --name "Checkout Redesign Q3" \
    --continuous revenue \
    --binary converted \
    --treatment-col group \
    --treatment-val treatment \
    --output report.md \
    --verbose
```

Expected output:
```
Loading data.csv...
Loaded 10,000 rows, 8 columns

🤖 Starting multi-agent diagnostics for: Checkout Redesign Q3

→ Running SRMAgent...
  Done. 2 findings.
→ Running BalanceAgent...
  Done. 5 findings.
→ Running EffectsAgent...
  Done. 2 findings.
→ Running BootstrapAgent...
  Done. 2 findings.
→ Running CoordinatorAgent (synthesising report)...
  Done. Verdict: PASS

══════════════════════════════════════════════════
# A/B Experiment Diagnostic Report: Checkout Redesign Q3
...
```

---

## CSV Format

Your CSV needs at minimum:
- A **group column** with treatment/control labels (default: `group`)
- One or more **outcome columns**

```csv
user_id,group,converted,revenue,age,tenure_days,device
0,control,0,7.42,34.1,180,mobile
1,treatment,1,12.30,28.5,90,desktop
...
```

Auto-detection picks up columns with names containing: `revenue`, `spend`, `converted`, `clicked`, `activated`, `sessions`, `orders`.

---

## Single Agent vs Multi-Agent

Both implementations are included. Run a direct comparison:

```bash
python compare.py                  # clean synthetic data
python compare.py --srm            # with SRM injected — tests gate behaviour
python compare.py --csv data.csv   # your own data
```

Sample output:
```
════════════════════════════════════════════════════════════════════
  Metric                           Single Agent     Multi-Agent
  ────────────────────────────────────────────────────────────────
  Claude API calls                            1               5
  Tool calls made                             6               6
  Wall-clock time (s)                       8.2            22.1
  Findings: ok                               10              10
  Verdict                                  PASS            PASS
```

| | Single Agent | Multi-Agent |
|---|---|---|
| API calls | 1 | 5 (coordinator + 4 sub-agents) |
| Code size | ~120 lines | ~350 lines |
| Gate logic | Prompt instruction | Enforced in Python |
| Best for | Sequential tasks, one context window | Parallelism, fault isolation |

**When to pick single agent:** task is sequential, fits in one context window, you want simpler code.

**When to pick multi-agent:** sub-tasks can run in parallel, gate logic must be guaranteed, or you need fault isolation (one agent failing ≠ whole pipeline failing).

---

## Agentic Loop — How It Works

```python
# Simplified from agents.py
def _run_agent_loop(system, user, tools, execute_tool):
    messages = [{"role": "user", "content": user}]

    while True:
        response = client.messages.create(
            model=MODEL, tools=tools, messages=messages
        )

        # Agent is done — return its interpretation
        if response.stop_reason == "end_turn":
            return response.content[0].text

        # Agent wants to call a tool
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                # Feed result back and loop
                messages += [assistant_turn, tool_result_turn]
```

---

## Requirements

- Python 3.8+
- `anthropic >= 0.20`
- `numpy`, `pandas`, `scipy`

---

## Running the Portfolio Notebook

```bash
jupyter notebook agent_walkthrough.ipynb
```

The notebook walks through each agent individually, shows tool call logs, and explains every design decision — designed to be readable during a technical interview.
