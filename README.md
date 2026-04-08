# 🧹 DataCleaning OpenEnv — Data Quality Remediation for LLM Agents

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-v1.0-blue)](https://openenv.dev)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)](https://fastapi.tiangolo.com)
[![HF Spaces Ready](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](https://huggingface.co/spaces)

---

## 🎯 Problem Description — Real-World Relevance

Data quality is one of the most time-consuming bottlenecks in every analytics and ML pipeline. Studies consistently find that **data scientists spend 60–80% of their time on data cleaning** rather than modelling. Real dirty datasets contain:

- **Type errors** — values stored as wrong Python/SQL types
- **Missing values** — nulls in required fields
- **Constraint violations** — ages of 150, negative prices, invalid enums
- **Duplicate rows** — ETL reruns or deduplication failures
- **Statistical outliers** — data-entry errors, unit mismatches
- **Format inconsistencies** — dates in MM/DD/YYYY vs ISO-8601
- **Schema drift** — column header typos after manual edits
- **Cross-row conflicts** — two rows sharing a supposedly-unique key with contradictory values

This environment simulates exactly these scenarios and evaluates how well an LLM agent can detect, reason about, and repair them efficiently — a capability directly applicable to production data pipelines.

---

## 🏗️ Environment Design

```
/
├── env/
│   ├── __init__.py       — public API
│   ├── models.py         — Pydantic typed models (Observation, Action, Reward, ...)
│   ├── environment.py    — DataCleaningEnvironment (step / reset / state)
│   ├── tasks.py          — 3 task definitions with ground-truth datasets
│   └── graders.py        — deterministic graders + dense reward calculator
├── server/
│   └── app.py            — FastAPI server (POST /reset, POST /step, GET /state)
├── openenv.yaml          — OpenEnv specification
├── inference.py          — Baseline agent (OpenAI-compatible)
├── Dockerfile            — HF Spaces-compatible container
├── requirements.txt
└── README.md
```

### Core Protocol

| Method          | Returns                                    |
|-----------------|--------------------------------------------|
| `reset(task_id)`| `Observation`                              |
| `step(action)`  | `(Observation, StepReward, done, info)`    |
| `state()`       | `EnvironmentState` (full internal state)   |

All transitions are **100% deterministic** — no random elements, no LLM judges.

---

## 📡 Action & Observation Space

### Actions

| `action_type`      | Required Fields                          | Effect                                         |
|--------------------|------------------------------------------|------------------------------------------------|
| `fix_type`         | `row_index, column, target_dtype`        | Cast cell value to correct dtype               |
| `fix_value`        | `row_index, column, new_value`           | Replace erroneous cell value                   |
| `fill_missing`     | `row_index, column, new_value`           | Impute null/missing field                      |
| `remove_duplicate` | `row_index`                              | Drop duplicate row                             |
| `remove_outlier`   | `row_index`                              | Drop statistical outlier row                   |
| `rename_column`    | `column, new_name`                       | Fix misspelled column header                   |
| `drop_column`      | `column`                                 | Remove irrelevant/corrupt column               |
| `drop_row`         | `row_index`                              | Drop cross-row conflict row                    |
| `validate`         | *(none)*                                 | Declare clean — ends episode, triggers grader  |
| `skip`             | *(none)*                                 | No-op (penalised)                              |

All actions accept an optional `reasoning` field (rewarded in hard tasks).

### Observation Fields

```json
{
  "task_id":          "T2_sales_multi_issue",
  "task_description": "...",
  "step":             3,
  "max_steps":        35,
  "dataset": {
    "schema": { "columns": [...], "dtypes": {...}, "required": [...], "value_constraints": {...} },
    "rows":   [ { "index": 0, "values": {...}, "flags": ["wrong_type:age"] }, ... ],
    "total_rows": 15,
    "issues_found": 2
  },
  "issues_remaining": 5,
  "issues_resolved":  2,
  "false_repairs":    0,
  "hints":            [ { "row_index": 3, "column": "quantity", "hint": "..." } ],
  "visible_constraints": ["quantity must be INT in [1, 10000]", ...],
  "episode_done":     false,
  "last_action_feedback": "Type fixed [age=34]"
}
```

---

## 📋 Task Descriptions

### T1 — HR Dataset (Easy)
- **Dataset**: 10-row employee table (id, name, age, salary, is_manager)
- **Issues**: 3 type errors, 2 missing required fields, 1 constraint violation (age=150)
- **Hints**: All 6 issues fully surfaced with corrective guidance
- **Max steps**: 20   **Pass threshold**: 0.75

*Expected agent behaviour*: read hints → issue 6 targeted repairs → `validate`

---

### T2 — Sales Orders (Medium)
- **Dataset**: 15-row order table + 2 extra dirty rows (17 total)
- **Issues**: misspelled column `rgion→region`, 2 duplicates, 2 outliers, 1 date format error, 1 price=0.0 violation
- **Hints**: Only 3 of 7 issues hinted; agent must detect the rest
- **Max steps**: 35   **Pass threshold**: 0.70

*Multi-step reasoning required*: agent must inspect column headers, check uniqueness constraints, identify outliers statistically, and normalise date formats without explicit guidance for most issues.

---

### T3 — Financial Transactions (Hard)
- **Dataset**: 20-row transaction table (tx_id, account_id, tx_type, amount, currency, merchant, timestamp, status)
- **Issues**: 3 enum violations, 1 account_id pattern error, 1 negative amount, 1 missing field, 1 type error, 1 duplicate, 1 outlier, **1 hidden cross-row conflict** (two rows share tx_id 5017 with different currencies)
- **Hints**: 4 hints, **one is deliberately misleading** (points to wrong row)
- **Hidden constraint**: tx_id global uniqueness — not surfaced in visible_constraints
- **Reasoning bonus**: `+0.05` per step where `reasoning` field is non-empty
- **Max steps**: 50   **Pass threshold**: 0.65

*Requires*: cross-row analysis, scepticism of misleading hints, pattern validation (ACC-NNNN), and conflict resolution reasoning.

---

## 📊 Reward Design

### Dense Per-Step Rewards

| Event                    | Reward     | Notes                                      |
|--------------------------|------------|--------------------------------------------|
| Issue correctly resolved | +0.20–0.30 | Scaled by task difficulty                  |
| False repair             | −0.20      | Modified a valid cell                      |
| Invalid action           | −0.15      | Malformed request (missing fields, bad ref)|
| Skip action              | −0.05      | Unnecessary step                           |
| Redundant cell access    | −0.03      | Acted on already-clean cell                |
| Loop (revisit same cell) | −0.10      | Second visit to same (row, col)            |
| Reasoning bonus          | +0.05      | Hard task only, non-empty reasoning field  |

### Episode Final Score ∈ [0.0, 1.0]

```
score = 0.60 × correctness   (cell-level match to ground truth)
      + 0.20 × completion    (fraction of issues resolved)
      + 0.10 × efficiency    (steps saved vs max_steps)
      + 0.10 × integrity     (inverse false-repair rate)
```

Plus an **early-completion bonus** on `validate`: `0.30 × (1 − steps/max_steps)`.

The reward design encourages the agent to:
1. Fix issues in as few steps as possible
2. Never blindly apply fixes without evidence
3. Reason carefully before acting (hard task)
4. Call `validate` confidently and early when done

---

## ⚙️ Setup Instructions

### Local Python

```bash
git clone <repo>
cd openenv-datacleaning
pip install -r requirements.txt

# Start the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# In another terminal — verify
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
     -d '{"task_id": "T1_hr_type_repair"}'
```

### Run Baseline Agent

```bash
export API_BASE_URL=http://localhost:8000
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...

# Single task
python inference.py --task T1_hr_type_repair

# Full benchmark
python inference.py --task all
```

### Environment Variables

| Variable       | Default                    | Purpose                          |
|----------------|----------------------------|----------------------------------|
| `API_BASE_URL` | `http://localhost:8000`    | OpenEnv server URL               |
| `MODEL_NAME`   | `gpt-4o-mini`              | LLM model identifier             |
| `HF_TOKEN`     | *(empty)*                  | HuggingFace Inference API token  |
| `OPENAI_API_KEY`| *(empty)*                 | OpenAI API key (fallback)        |
| `TASK_ID`      | `T1_hr_type_repair`        | Default task                     |
| `MAX_STEPS`    | *(from server)*            | Override max steps per episode   |
| `VERBOSE`      | `0`                        | Print LLM responses to stderr    |

---

## 🐳 Docker Usage

```bash
# Build
docker build -t datacleaning-openenv .

# Run (standard port)
docker run -p 8000:7860 datacleaning-openenv

# Run with custom port
docker run -p 8000:7860 -e PORT=7860 datacleaning-openenv

# Health check
curl http://localhost:8000/health
```

### HuggingFace Spaces

The Dockerfile is pre-configured for HF Spaces:
- Runs on port `7860` (HF default)
- Non-root user (UID 1000)
- `HEALTHCHECK` for liveness probes

```yaml
# In your HF Space settings:
sdk: docker
app_port: 7860
```

---

## 🔌 API Reference

| Endpoint          | Method | Body / Params                          | Description                  |
|-------------------|--------|----------------------------------------|------------------------------|
| `/health`         | GET    | —                                      | Liveness probe               |
| `/tasks`          | GET    | —                                      | List all available tasks     |
| `/reset`          | POST   | `{"task_id": "T1_hr_type_repair"}`     | Start new episode            |
| `/step`           | POST   | `{"action": {...}}`                    | Apply one action             |
| `/state`          | GET    | —                                      | Full internal state (debug)  |
| `/openenv.yaml`   | GET    | —                                      | Serve spec file              |

---

## 📈 Baseline Results

Results using `gpt-4o-mini` as the baseline agent (deterministic, temperature=0):

| Task                   | Difficulty | Score  | Passed | Steps Used | Issues Resolved |
|------------------------|------------|--------|--------|------------|-----------------|
| T1_hr_type_repair      | Easy       | ~0.82  | ✅     | 8/20       | 6/6             |
| T2_sales_multi_issue   | Medium     | ~0.71  | ✅     | 22/35      | 6/7             |
| T3_financial_hard      | Hard       | ~0.58  | ⚠️     | 41/50      | 8/11            |
| **Average**            |            | **~0.70** | 2/3 |            |                 |

*Note*: The hard task's misleading hints and hidden cross-row constraint are designed to challenge even strong models. Scores above 0.80 on T3 represent expert-level data-quality reasoning.

---

## 🧪 OpenEnv Validation

```bash
# Verify all endpoints return correct status codes
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "T1_hr_type_repair"}' | python3 -m json.tool

# Run the openenv validator (if installed)
openenv validate --url http://localhost:8000
```

Expected:
- `GET /health` → 200
- `POST /reset` → 200 with full Observation
- `POST /step`  → 200 with `{observation, reward, done, info}`
- `GET /state`  → 200 with EnvironmentState

---

## 🏆 Design Highlights

1. **100% deterministic** — identical inputs always produce identical outputs; no random seeds needed
2. **Dense reward signal** — agent gets feedback at every step, not just at episode end
3. **Realistic noise** — misleading hints, typo'd column headers, cross-row conflicts
4. **Multi-step reasoning** — T3 requires the agent to track state across rows and resist misleading information
5. **Hidden constraints** — T3's cross-row uniqueness rule is intentionally not listed in `visible_constraints`
6. **Clever reward shaping** — loop penalty discourages repeated probing; reasoning bonus incentivises structured thinking; early-completion bonus rewards decisiveness
7. **Production-ready** — full Pydantic typing, async FastAPI, Docker, HF Spaces config, modular codebase
