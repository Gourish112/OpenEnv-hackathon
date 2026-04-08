from __future__ import annotations

import re
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    Action, ActionType, CellIssue, DataRow, DatasetSchema, DatasetSnapshot,
    DType, EnvironmentState, EpisodeResult, Observation, StepReward,
)
from env.tasks import get_task
from env.graders import compute_step_reward, grade_dataset, _coerce, _values_match


class DataCleaningEnvironment:
    """
    Simulates a data-quality remediation workflow.

    The agent receives a dirty dataset and must issue targeted repair actions
    until the dataset is clean, then call VALIDATE to end the episode.
    """

    def __init__(self, task_id: str = "T1_hr_type_repair") -> None:
        self._task_id = task_id
        self._task: Optional[dict] = None
        self._step_count: int = 0
        self._current_rows: List[DataRow] = []
        self._ground_truth: List[DataRow] = []
        self._schema: Optional[DatasetSchema] = None
        self._done: bool = False
        self._issues_resolved: int = 0
        self._false_repairs: int = 0
        self._visited_cells: Dict[str, int] = {}
        self._cumulative_reward: float = 0.0
        self._episode_result: Optional[EpisodeResult] = None
        self._last_feedback: str = ""
        self._column_aliases: Dict[str, str] = {}  

    async def reset(self, task_id: Optional[str] = None) -> Observation:
        """Reset environment to a fresh episode."""
        if task_id:
            self._task_id = task_id
        self._task = get_task(self._task_id)

        self._step_count = 0
        self._current_rows = deepcopy(self._task["dirty_rows"])
        self._ground_truth = deepcopy(self._task["ground_truth"])
        self._schema = deepcopy(self._task["schema"])
        self._done = False
        self._issues_resolved = 0
        self._false_repairs = 0
        self._visited_cells = {}
        self._cumulative_reward = 0.0
        self._episode_result = None
        self._last_feedback = "Episode started. Analyse the dataset and begin repairs."

        self._column_aliases = self._detect_column_aliases()

        return self._build_observation()

    async def step(self, action: Action) -> Tuple[Observation, StepReward, bool, dict]:
        """
        Apply an action and return (observation, reward, done, info).
        Raises ValueError if called after episode is done.
        """
        if self._done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        reward, feedback = await self._apply_action(action)
        self._cumulative_reward += reward.total
        self._last_feedback = feedback

        obs = self._build_observation()
        info = {
            "step": self._step_count,
            "issues_resolved": self._issues_resolved,
            "false_repairs": self._false_repairs,
            "cumulative_reward": self._cumulative_reward,
            "episode_result": self._episode_result.model_dump() if self._episode_result else None,
        }
        return obs, reward, self._done, info

    async def state(self) -> EnvironmentState:
        """Return full internal state (used by GET /state)."""
        return EnvironmentState(
            **{
                "task_id":           self._task_id,
                "step":              self._step_count,
                "max_steps":         self._task["max_steps"] if self._task else 0,
                "is_done":           self._done,
                "ground_truth_rows": self._ground_truth,
                "current_rows":      self._current_rows,
                "schema":            self._schema,
                "issues_total":      self._task["total_issues"] if self._task else 0,
                "issues_resolved":   self._issues_resolved,
                "false_repairs":     self._false_repairs,
                "visited_cells":     self._visited_cells,
                "cumulative_reward": self._cumulative_reward,
                "episode_result":    self._episode_result,
            }
        )

    async def _apply_action(self, action: Action) -> Tuple[StepReward, str]:
        difficulty = self._task.get("difficulty", "easy")
        max_steps = self._task["max_steps"]

        if action.action_type == ActionType.VALIDATE:
            return await self._handle_validate(difficulty, max_steps)

        if action.action_type == ActionType.SKIP:
            r = self._make_reward(
                "efficiency_penalty", -0.05,
                "SKIP action used — unnecessary step penalised."
            )
            return r, "SKIP recorded. Each skip wastes a step."

        if action.action_type == ActionType.RENAME_COLUMN:
            return await self._handle_rename_column(action, difficulty)

        if action.action_type == ActionType.DROP_COLUMN:
            return await self._handle_drop_column(action, difficulty)

        if action.row_index is None:
            r = self._make_reward("invalid_action", -0.15,
                                  "Action requires row_index but none provided.")
            return r, "Invalid action: row_index is required."

        row = self._get_row(action.row_index)
        if row is None:
            r = self._make_reward("invalid_action", -0.15,
                                  f"Row {action.row_index} not found in dataset.")
            return r, f"Invalid action: row {action.row_index} does not exist."

        if action.action_type == ActionType.REMOVE_DUPLICATE:
            return await self._handle_remove_duplicate(action, row, difficulty)

        if action.action_type == ActionType.REMOVE_OUTLIER:
            return await self._handle_remove_outlier(action, row, difficulty)

        if action.action_type == ActionType.DROP_ROW:
            return await self._handle_drop_row(action, row, difficulty)

        if action.column is None:
            r = self._make_reward("invalid_action", -0.15,
                                  "Cell action requires 'column' to be specified.")
            return r, "Invalid action: column is required for cell-level actions."

        col = self._resolve_column(action.column)
        if col not in (row.values.keys()):
            r = self._make_reward("invalid_action", -0.15,
                                  f"Column '{action.column}' not found in row {action.row_index}.")
            return r, f"Invalid action: column '{action.column}' not in row."

        cell_key = f"{action.row_index}:{col}"
        already_visited = self._visited_cells.get(cell_key, 0) > 0
        self._visited_cells[cell_key] = self._visited_cells.get(cell_key, 0) + 1

        if action.action_type == ActionType.FIX_TYPE:
            return await self._handle_fix_type(action, row, col, already_visited, difficulty)

        if action.action_type in (ActionType.FIX_VALUE, ActionType.FILL_MISSING):
            return await self._handle_fix_value(action, row, col, already_visited, difficulty)

        r = self._make_reward("invalid_action", -0.15, f"Unknown action type: {action.action_type}")
        return r, "Invalid action type."

    

    async def _handle_validate(self, difficulty: str, max_steps: int) -> Tuple[StepReward, str]:
        """Close the episode and compute final score."""
        self._done = True
        remaining = self._task["total_issues"] - self._issues_resolved
        early_bonus = max(0.0, 0.30 * (1.0 - self._step_count / max_steps))

        self._episode_result = grade_dataset(
            current_rows=self._current_rows,
            ground_truth=self._ground_truth,
            schema=self._schema,
            task_id=self._task_id,
            total_steps=self._step_count,
            max_steps=max_steps,
            issues_total=self._task["total_issues"],
            issues_resolved=self._issues_resolved,
            false_repairs=self._false_repairs,
            cumulative_reward=self._cumulative_reward,
        )

        completion_bonus = self._episode_result.final_score * 0.5
        total_reward = completion_bonus + early_bonus
        if remaining > 0:
            total_reward -= remaining * 0.05  # penalty for premature validate

        components = StepReward(
            total=round(total_reward, 4),
            issue_resolved=0.0,
            false_repair=0.0,
            invalid_action=0.0,
            efficiency_penalty=0.0,
            loop_penalty=0.0,
            reasoning_bonus=0.0,
            early_completion=round(early_bonus, 4),
            explanation=f"Episode ended. Score={self._episode_result.final_score:.3f}. "
                        f"Resolved {self._issues_resolved}/{self._task['total_issues']} issues. "
                        f"Passed={self._episode_result.passed}",
        )
        return components, f"VALIDATE called. Final score: {self._episode_result.final_score:.3f}"

    async def _handle_rename_column(self, action: Action, difficulty: str) -> Tuple[StepReward, str]:
        if not action.column or not action.new_name:
            r = self._make_reward("invalid_action", -0.15, "RENAME_COLUMN needs 'column' and 'new_name'.")
            return r, "Invalid RENAME_COLUMN: missing column or new_name."

        old_col = action.column
        new_col = action.new_name

        # Check old_col exists in any row
        found = any(old_col in row.values for row in self._current_rows)
        if not found:
            r = self._make_reward("invalid_action", -0.15, f"Column '{old_col}' not found.")
            return r, f"Column '{old_col}' not found."

        # Check new_col is expected by schema
        if new_col not in self._schema.columns:
            r = self._make_reward("false_repair", -0.20, f"'{new_col}' is not a schema column.")
            self._false_repairs += 1
            return r, f"'{new_col}' is not a valid schema column."

        # Apply rename
        for row in self._current_rows:
            if old_col in row.values:
                row.values[new_col] = row.values.pop(old_col)
                row.flags = [f for f in row.flags if old_col not in f]

        self._column_aliases[old_col] = new_col
        self._issues_resolved += 1

        components = self._build_step_reward(
            compute_step_reward("rename_column", True, False, False, False,
                                bool(action.reasoning), difficulty)
        )
        return components, f"Column '{old_col}' renamed to '{new_col}'. Issue resolved."

    async def _handle_drop_column(self, action: Action, difficulty: str) -> Tuple[StepReward, str]:
        col = action.column
        if not col:
            r = self._make_reward("invalid_action", -0.15, "DROP_COLUMN requires 'column'.")
            return r, "DROP_COLUMN requires 'column'."
        if col in self._schema.columns:
            r = self._make_reward("false_repair", -0.20, f"'{col}' is a valid schema column.")
            self._false_repairs += 1
            return r, f"Cannot drop required column '{col}'."
        for row in self._current_rows:
            row.values.pop(col, None)
        return self._make_reward("issue_resolved", 0.10, f"Column '{col}' dropped."), f"Dropped '{col}'."

    async def _handle_remove_duplicate(
        self, action: Action, row: DataRow, difficulty: str
    ) -> Tuple[StepReward, str]:
        is_dup = "duplicate" in row.flags
        if not is_dup:
            r = self._make_reward("false_repair", -0.20, f"Row {action.row_index} is not a duplicate.")
            self._false_repairs += 1
            return r, f"Row {action.row_index} is not flagged as duplicate — false repair."

        self._current_rows = [r for r in self._current_rows if r.index != action.row_index]
        self._issues_resolved += 1
        comps = self._build_step_reward(
            compute_step_reward("remove_duplicate", True, False, False, False,
                                bool(action.reasoning), difficulty)
        )
        return comps, f"Duplicate row {action.row_index} removed."

    async def _handle_remove_outlier(
        self, action: Action, row: DataRow, difficulty: str
    ) -> Tuple[StepReward, str]:
        is_outlier = any("outlier" in f for f in row.flags)
        if not is_outlier:
            r = self._make_reward("false_repair", -0.20, f"Row {action.row_index} is not an outlier.")
            self._false_repairs += 1
            return r, f"Row {action.row_index} is not an outlier — false repair."

        self._current_rows = [r for r in self._current_rows if r.index != action.row_index]
        self._issues_resolved += 1
        comps = self._build_step_reward(
            compute_step_reward("remove_outlier", True, False, False, False,
                                bool(action.reasoning), difficulty)
        )
        return comps, f"Outlier row {action.row_index} removed."

    async def _handle_drop_row(
        self, action: Action, row: DataRow, difficulty: str
    ) -> Tuple[StepReward, str]:
        # Only valid for cross-row-conflict resolution (T3 hidden constraint)
        is_conflict = "cross_row_conflict" in row.flags
        if not is_conflict:
            r = self._make_reward("false_repair", -0.20,
                                  f"Row {action.row_index} cannot be dropped (not a conflict row).")
            self._false_repairs += 1
            return r, "DROP_ROW rejected: row is not a cross-row conflict."
        self._current_rows = [r for r in self._current_rows if r.index != action.row_index]
        self._issues_resolved += 1
        comps = self._build_step_reward(
            compute_step_reward("drop_row", True, False, False, False,
                                bool(action.reasoning), difficulty)
        )
        return comps, f"Conflict row {action.row_index} dropped."

    async def _handle_fix_type(
        self, action: Action, row: DataRow, col: str,
        already_visited: bool, difficulty: str
    ) -> Tuple[StepReward, str]:
        if action.target_dtype is None:
            r = self._make_reward("invalid_action", -0.15, "FIX_TYPE requires 'target_dtype'.")
            return r, "FIX_TYPE requires 'target_dtype'."

        cur_val = row.values.get(col)
        expected_dtype = self._schema.dtypes.get(col, DType.STRING)
        ok, coerced = _coerce(cur_val, action.target_dtype)

        if not ok:
            r = self._make_reward("false_repair", -0.20,
                                  f"Cannot coerce {cur_val!r} to {action.target_dtype}.")
            self._false_repairs += 1
            return r, f"Cannot coerce '{cur_val}' to {action.target_dtype}."

        # Is this the right dtype?
        if action.target_dtype != expected_dtype:
            r = self._make_reward("false_repair", -0.20,
                                  f"Wrong target dtype. Expected {expected_dtype} for col '{col}'.")
            self._false_repairs += 1
            return r, f"Wrong target dtype for '{col}'."

        # Check if cell actually had a type issue
        issue_resolved = self._is_type_issue(col, cur_val, expected_dtype)
        row.values[col] = coerced
        row.flags = [f for f in row.flags if f"wrong_type:{col}" not in f]

        if issue_resolved:
            self._issues_resolved += 1

        comps = self._build_step_reward(
            compute_step_reward("fix_type", issue_resolved, not issue_resolved,
                                False, already_visited, bool(action.reasoning), difficulty)
        )
        if not issue_resolved:
            self._false_repairs += 1
        msg = ("Type fixed" if issue_resolved else "No type issue existed here — false repair.")
        return comps, f"{msg} [{col}={coerced!r}]"

    async def _handle_fix_value(
        self, action: Action, row: DataRow, col: str,
        already_visited: bool, difficulty: str
    ) -> Tuple[StepReward, str]:
        if action.new_value is None and action.action_type != ActionType.FILL_MISSING:
            r = self._make_reward("invalid_action", -0.15, "FIX_VALUE requires 'new_value'.")
            return r, "FIX_VALUE requires 'new_value'."

        cur_val = row.values.get(col)
        dtype = self._schema.dtypes.get(col, DType.STRING)
        new_val = action.new_value

        # Validate new value against constraints
        constraint = self._schema.value_constraints.get(col, {})
        if constraint and new_val is not None:
            ok, coerced = _coerce(new_val, dtype)
            if not ok:
                r = self._make_reward("false_repair", -0.20, f"new_value {new_val!r} wrong dtype.")
                self._false_repairs += 1
                return r, f"new_value cannot be coerced to {dtype}."
            if not self._check_constraint(coerced, constraint):
                r = self._make_reward("false_repair", -0.20,
                                      f"new_value {new_val!r} violates constraint {constraint}.")
                self._false_repairs += 1
                return r, f"new_value violates constraint for '{col}'."
            new_val = coerced

        # Was there actually an issue here?
        issue_resolved = self._is_value_issue(col, row.index, cur_val, dtype)
        row.values[col] = new_val

        # Remove issue flags
        row.flags = [f for f in row.flags
                     if f"wrong_value:{col}" not in f
                     and f"missing:{col}" not in f]

        if issue_resolved:
            self._issues_resolved += 1
        else:
            self._false_repairs += 1

        comps = self._build_step_reward(
            compute_step_reward("fix_value", issue_resolved, not issue_resolved,
                                False, already_visited, bool(action.reasoning), difficulty)
        )
        msg = "Value repaired." if issue_resolved else "No issue existed here — false repair."
        return comps, f"{msg} [{col}: {cur_val!r} → {new_val!r}]"


    def _get_row(self, idx: int) -> Optional[DataRow]:
        return next((r for r in self._current_rows if r.index == idx), None)

    def _resolve_column(self, col: str) -> str:
        return self._column_aliases.get(col, col)

    def _detect_column_aliases(self) -> Dict[str, str]:
        """Detect typo'd column names by comparing row keys to schema columns."""
        if not self._current_rows or not self._schema:
            return {}
        sample_cols = set(self._current_rows[0].values.keys()) if self._current_rows else set()
        schema_cols = set(self._schema.columns)
        aliases = {}
        for sc in sample_cols:
            if sc not in schema_cols:
                # Find closest schema column by edit distance
                best = min(schema_cols, key=lambda x: _edit_distance(sc, x))
                if _edit_distance(sc, best) <= 3:
                    aliases[sc] = best
        return aliases

    def _is_type_issue(self, col: str, val: Any, expected_dtype: DType) -> bool:
        if val is None:
            return False
        ok, _ = _coerce(val, expected_dtype)
        if not ok:
            return True
        # Check if value is stored as wrong Python type
        if expected_dtype == DType.INT and not isinstance(val, int):
            return True
        if expected_dtype == DType.FLOAT and not isinstance(val, float):
            return True
        if expected_dtype == DType.BOOLEAN and not isinstance(val, bool):
            return True
        return False

    def _is_value_issue(self, col: str, row_idx: int, val: Any, dtype: DType) -> bool:
        """Check whether this cell has a genuine value issue vs ground truth or constraints."""
        if val is None:
            return col in self._schema.required
        # Check constraint
        constraint = self._schema.value_constraints.get(col, {})
        if constraint:
            ok, cv = _coerce(val, dtype)
            if not ok:
                return True
            if not self._check_constraint(cv, constraint):
                return True
        # Check against ground truth
        truth_row = next((r for r in self._ground_truth if r.index == row_idx), None)
        if truth_row:
            expected = truth_row.values.get(col)
            return not _values_match(val, expected, dtype)
        return False

    @staticmethod
    def _check_constraint(val: Any, constraint: Dict[str, Any]) -> bool:
        if "min" in constraint and float(val) < constraint["min"]:
            return False
        if "max" in constraint and float(val) > constraint["max"]:
            return False
        if "enum" in constraint and str(val).lower() not in [e.lower() for e in constraint["enum"]]:
            return False
        if "pattern" in constraint and not re.match(constraint["pattern"], str(val)):
            return False
        return True

    def _build_observation(self) -> Observation:
        task = self._task
        snapshot = DatasetSnapshot(
            **{
                "schema": self._schema,
                "rows": self._current_rows,
                "total_rows": len(self._current_rows),
                "visible_rows": len(self._current_rows),
                "issues_found": self._issues_resolved + self._false_repairs,
            }
        )
        return Observation(
            task_id=self._task_id,
            task_description=task["description"],
            step=self._step_count,
            max_steps=task["max_steps"],
            dataset=snapshot,
            issues_remaining=task["total_issues"] - self._issues_resolved,
            issues_resolved=self._issues_resolved,
            false_repairs=self._false_repairs,
            hints=task.get("hints", []) if task.get("difficulty") != "hard" else task.get("hints", [])[:2],
            visible_constraints=task.get("visible_constraints", []),
            episode_done=self._done,
            last_action_feedback=self._last_feedback,
        )

    def _make_reward(self, component: str, value: float, explanation: str) -> StepReward:
        kwargs = {
            "total": value,
            "issue_resolved": 0.0,
            "false_repair": 0.0,
            "invalid_action": 0.0,
            "efficiency_penalty": 0.0,
            "loop_penalty": 0.0,
            "reasoning_bonus": 0.0,
            "early_completion": 0.0,
            "explanation": explanation,
            component: value,
        }
        return StepReward(**{k: v for k, v in kwargs.items()
                             if k in StepReward.model_fields or k == "total"})

    def _build_step_reward(self, components: Dict[str, float]) -> StepReward:
        total = sum(components.values())
        return StepReward(
            total=round(total, 4),
            issue_resolved=components.get("issue_resolved", 0.0),
            false_repair=components.get("false_repair", 0.0),
            invalid_action=components.get("invalid_action", 0.0),
            efficiency_penalty=components.get("efficiency_penalty", 0.0),
            loop_penalty=components.get("loop_penalty", 0.0),
            reasoning_bonus=components.get("reasoning_bonus", 0.0),
            early_completion=components.get("early_completion", 0.0),
            explanation="",
        )


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]
