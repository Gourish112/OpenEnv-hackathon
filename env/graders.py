"""
Deterministic graders for the Data Cleaning & Anomaly Remediation environment.

Each grader compares the agent's current dataset state against ground truth
and returns a continuous score in [0.0, 1.0] plus a detailed breakdown.

Grading is fully deterministic — no LLM judges, no random elements.
"""
from __future__ import annotations

import re
import math
from typing import Any, Dict, List, Optional, Tuple

from env.models import DataRow, DatasetSchema, DType, EpisodeResult



def _coerce(val: Any, dtype: DType) -> Tuple[bool, Any]:
    """Try to coerce a value to the expected dtype. Returns (success, coerced)."""
    if val is None:
        return False, None
    try:
        if dtype == DType.INT:
            return True, int(val)
        if dtype == DType.FLOAT:
            return True, float(val)
        if dtype == DType.STRING:
            return True, str(val)
        if dtype == DType.BOOLEAN:
            if isinstance(val, bool):
                return True, val
            if str(val).lower() in ("true", "1", "yes"):
                return True, True
            if str(val).lower() in ("false", "0", "no"):
                return True, False
            return False, val
        if dtype == DType.DATE:
            s = str(val)
            if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                return True, s
            return False, val
    except (ValueError, TypeError):
        pass
    return False, val


def _values_match(a: Any, b: Any, dtype: DType) -> bool:
    """Return True if two values are semantically equal under dtype."""
    ok_a, ca = _coerce(a, dtype)
    ok_b, cb = _coerce(b, dtype)
    if not ok_a or not ok_b:
        return False
    if dtype == DType.FLOAT:
        return math.isclose(float(ca), float(cb), rel_tol=1e-6)
    return ca == cb


def _check_constraint(val: Any, constraint: Dict[str, Any], dtype: DType) -> bool:
    """Check a single value against its constraints. Returns True if valid."""
    if val is None:
        return False
    ok, cv = _coerce(val, dtype)
    if not ok:
        return False
    if "min" in constraint and float(cv) < constraint["min"]:
        return False
    if "max" in constraint and float(cv) > constraint["max"]:
        return False
    if "enum" in constraint and str(cv).lower() not in [e.lower() for e in constraint["enum"]]:
        return False
    if "pattern" in constraint:
        if not re.match(constraint["pattern"], str(cv)):
            return False
    return True



def _row_matches_truth(
    current: DataRow,
    truth: DataRow,
    schema: DatasetSchema,
) -> Tuple[float, List[str]]:
    """
    Compare a current row to its ground-truth counterpart.
    Returns (fraction_correct ∈ [0,1], list_of_mismatches).
    Columns missing from schema are ignored.
    """
    cols = schema.columns
    correct = 0
    mismatches = []
    for col in cols:
        dtype = schema.dtypes.get(col, DType.STRING)
        cur_val = current.values.get(col)
        tru_val = truth.values.get(col)
        if _values_match(cur_val, tru_val, dtype):
            correct += 1
        else:
            mismatches.append(f"col={col}: got={cur_val!r}, want={tru_val!r}")
    return correct / len(cols) if cols else 1.0, mismatches



def grade_dataset(
    current_rows: List[DataRow],
    ground_truth: List[DataRow],
    schema: DatasetSchema,
    task_id: str,
    total_steps: int,
    max_steps: int,
    issues_total: int,
    issues_resolved: int,
    false_repairs: int,
    cumulative_reward: float,
) -> EpisodeResult:
    """
    Compute the final episode score by comparing current state to ground truth.

    Scoring breakdown:
      - correctness_score  (60%): how many cells match ground truth
      - completion_score   (20%): bonus if ALL issues resolved
      - efficiency_score   (10%): steps used vs max_steps
      - integrity_score    (10%): penalty for false repairs and constraint violations
    """
    # ---- Correctness -------------------------------------------------------
    # Match current rows to ground truth rows by primary key (first unique_key)
    pk_col = schema.unique_keys[0] if schema.unique_keys else None

    truth_by_key: Dict[Any, DataRow] = {}
    if pk_col:
        for r in ground_truth:
            k = r.values.get(pk_col)
            if k is not None:
                truth_by_key[k] = r

    total_cell_score = 0.0
    matched_rows = 0
    unmatched_rows = 0
    mismatch_details: List[str] = []

    for row in current_rows:
        if pk_col:
            key = row.values.get(pk_col)
            truth_row = truth_by_key.get(key)
        else:
            # Fallback: match by index
            truth_row = next((r for r in ground_truth if r.index == row.index), None)

        if truth_row is None:
            # Extra row not in ground truth
            unmatched_rows += 1
            continue

        score, mismatches = _row_matches_truth(row, truth_row, schema)
        total_cell_score += score
        matched_rows += 1
        mismatch_details.extend(mismatches[:3])  # cap verbosity

    # Rows in ground truth that are missing from current
    missing_keys: set = set()
    if pk_col:
        current_keys = {r.values.get(pk_col) for r in current_rows}
        missing_keys = set(truth_by_key.keys()) - current_keys
        missing_row_penalty = len(missing_keys) * 0.5  # per missing row
    else:
        missing_row_penalty = 0.0

    expected_rows = len(ground_truth)
    if expected_rows > 0:
        row_coverage = matched_rows / expected_rows
        avg_cell_score = total_cell_score / expected_rows
    else:
        row_coverage = 1.0
        avg_cell_score = 1.0

    # Extra rows penalty
    extra_rows = max(0, len(current_rows) - expected_rows)
    extra_row_penalty = extra_rows * 0.1

    correctness_score = max(0.0, avg_cell_score * row_coverage
                            - missing_row_penalty * 0.05
                            - extra_row_penalty)
    correctness_score = min(1.0, correctness_score)

    # ---- Completion --------------------------------------------------------
    issue_resolution_rate = issues_resolved / max(issues_total, 1)
    completion_score = issue_resolution_rate

    # ---- Efficiency --------------------------------------------------------
    step_ratio = total_steps / max(max_steps, 1)
    # Full score for using ≤50% steps, linear decay to 0 at 100%
    efficiency_score = max(0.0, 1.0 - step_ratio)

    # ---- Integrity ---------------------------------------------------------
    false_repair_rate = false_repairs / max(issues_total, 1)
    integrity_score = max(0.0, 1.0 - false_repair_rate * 2)

    # ---- Weighted final score ----------------------------------------------
    weights = {
        "correctness":  0.60,
        "completion":   0.20,
        "efficiency":   0.10,
        "integrity":    0.10,
    }

    final_score = (
        weights["correctness"]  * correctness_score +
        weights["completion"]   * completion_score  +
        weights["efficiency"]   * efficiency_score  +
        weights["integrity"]    * integrity_score
    )
    final_score = round(min(1.0, max(0.0, final_score)), 4)

    # Hard-task bonus: reasoning provided (cumulative_reward already captures this)
    pass_threshold_map = {
        "T1_hr_type_repair":    0.75,
        "T2_sales_multi_issue": 0.70,
        "T3_financial_hard":    0.65,
    }
    threshold = pass_threshold_map.get(task_id, 0.70)

    return EpisodeResult(
        task_id=task_id,
        total_steps=total_steps,
        final_score=final_score,
        issues_total=issues_total,
        issues_resolved=issues_resolved,
        false_repairs=false_repairs,
        cumulative_reward=round(cumulative_reward, 4),
        grade_breakdown={
            "correctness_score": round(correctness_score, 4),
            "completion_score":  round(completion_score, 4),
            "efficiency_score":  round(efficiency_score, 4),
            "integrity_score":   round(integrity_score, 4),
            "row_coverage":      round(row_coverage, 4),
            "step_ratio":        round(step_ratio, 4),
            "extra_rows":        float(extra_rows),
            "missing_rows":      float(len(missing_keys) if pk_col else 0),
        },
        passed=final_score >= threshold,
    )

def compute_step_reward(
    action_type: str,
    issue_resolved: bool,
    false_repair: bool,
    invalid_action: bool,
    already_visited: bool,
    reasoning_provided: bool,
    task_difficulty: str,
) -> Dict[str, float]:
    """
    Dense per-step reward shaping.

    Returns a dict with reward components; caller sums them.
    All values in [-1.0, +1.0] range.
    """
    rewards: Dict[str, float] = {
        "issue_resolved":     0.0,
        "false_repair":       0.0,
        "invalid_action":     0.0,
        "efficiency_penalty": 0.0,
        "loop_penalty":       0.0,
        "reasoning_bonus":    0.0,
        "early_completion":   0.0,
    }

    if invalid_action:
        rewards["invalid_action"] = -0.15
        return rewards

    if action_type == "skip":
        rewards["efficiency_penalty"] = -0.05
        return rewards

    if action_type == "validate":
        # Reward/penalty handled at episode end by grader
        return rewards

    if already_visited:
        rewards["loop_penalty"] = -0.10

    if false_repair:
        rewards["false_repair"] = -0.20

    if issue_resolved:
        # Scale reward by difficulty
        base = {"easy": 0.20, "medium": 0.25, "hard": 0.30}.get(task_difficulty, 0.20)
        rewards["issue_resolved"] = base
    else:
        # Acted on a cell that had no issue (or wrong fix) — small penalty
        if not false_repair and not already_visited:
            rewards["efficiency_penalty"] = -0.03

    # Reasoning bonus in hard task
    if task_difficulty == "hard" and reasoning_provided:
        rewards["reasoning_bonus"] = 0.05

    return rewards
