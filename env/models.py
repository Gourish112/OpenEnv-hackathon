"""
Pydantic typed models for the Data Cleaning & Anomaly Remediation environment.
All models are strictly typed, serialisable, and deterministic.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator



class ActionType(str, Enum):
    # Cell-level repairs
    FIX_TYPE        = "fix_type"          # cast a cell to the correct dtype
    FIX_VALUE       = "fix_value"         # replace an erroneous cell value
    FILL_MISSING    = "fill_missing"      # impute a missing / null cell
    REMOVE_DUPLICATE = "remove_duplicate" # drop a duplicate row by index
    REMOVE_OUTLIER  = "remove_outlier"    # remove a statistical outlier row
    # Column-level operations
    RENAME_COLUMN   = "rename_column"     # fix a misspelled column header
    DROP_COLUMN     = "drop_column"       # remove an irrelevant / corrupt column
    # Row-level operations
    DROP_ROW        = "drop_row"          # drop a structurally broken row
    # Dataset-level
    VALIDATE        = "validate"          # declare dataset clean; ends episode
    SKIP            = "skip"              # no-op (penalised for inefficiency)


class Severity(str, Enum):
    CRITICAL = "critical"  # blocks downstream pipelines
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


class DType(str, Enum):
    INT     = "int"
    FLOAT   = "float"
    STRING  = "string"
    BOOLEAN = "boolean"
    DATE    = "date"     # ISO-8601 string expected



class CellIssue(BaseModel):
    row_index:   int
    column:      str
    issue_type:  str          # "wrong_type" | "wrong_value" | "missing" | "outlier"
    current_val: Any
    hint:        Optional[str] = None   # surfaced in medium/hard tasks

class DatasetSchema(BaseModel):
    """Expected schema for the dataset in this episode."""
    columns:        List[str]
    dtypes:         Dict[str, DType]
    required:       List[str]           # columns that must not be null
    unique_keys:    List[str] = []      # columns that must be unique
    value_constraints: Dict[str, Any] = {}  # e.g. {"age": {"min": 0, "max": 130}}

class DataRow(BaseModel):
    index:  int
    values: Dict[str, Any]
    flags:  List[str] = []  # e.g. ["duplicate", "outlier", "missing:age"]

class DatasetSnapshot(BaseModel):
    """Partial view of the dataset exposed in an observation."""
    schema_:      DatasetSchema        = Field(alias="schema")
    rows:         List[DataRow]
    total_rows:   int
    visible_rows: int                  # may be less than total in hard tasks
    issues_found: int                  # agent-visible running count

    model_config = {"populate_by_name": True}



class Action(BaseModel):
    """
    Fully typed action submitted by the agent.
    Only relevant fields need to be populated per action_type.
    """
    action_type:   ActionType

    # Target coordinates
    row_index:     Optional[int]   = None
    column:        Optional[str]   = None

    # Payloads
    new_value:     Optional[Any]   = None   # for FIX_VALUE / FILL_MISSING
    target_dtype:  Optional[DType] = None   # for FIX_TYPE
    new_name:      Optional[str]   = None   # for RENAME_COLUMN

    # Justification (rewarded in hard task)
    reasoning:     Optional[str]   = None

    @field_validator("action_type", mode="before")
    @classmethod
    def coerce_action_type(cls, v: Any) -> ActionType:
        if isinstance(v, str):
            return ActionType(v)
        return v


class Observation(BaseModel):
    """Everything the agent can see at a given step."""
    task_id:          str
    task_description: str
    step:             int
    max_steps:        int

    dataset:          DatasetSnapshot

    # Running scoreboard (visible to agent)
    issues_remaining: int
    issues_resolved:  int
    false_repairs:    int      # repairs that broke valid cells

    # Structured hints (empty in easy, partial in medium, misleading in hard)
    hints:            List[CellIssue] = []

    # Constraints the agent must respect (hidden subset revealed in hard task)
    visible_constraints: List[str] = []

    episode_done:     bool = False
    last_action_feedback: Optional[str] = None



class StepReward(BaseModel):
    """Dense per-step reward breakdown."""
    total:              float

    # Components
    issue_resolved:     float = 0.0   # +ve for correct fix
    false_repair:       float = 0.0   # -ve for breaking valid cell
    invalid_action:     float = 0.0   # -ve for malformed / no-op action
    efficiency_penalty: float = 0.0   # -ve for skips / redundant actions
    loop_penalty:       float = 0.0   # -ve for repeating same (row,col) pair
    reasoning_bonus:    float = 0.0   # +ve for justified action in hard task
    early_completion:   float = 0.0   # +ve for finishing under step budget

    explanation: str = ""


class EpisodeResult(BaseModel):
    task_id:          str
    total_steps:      int
    final_score:      float           # ∈ [0.0, 1.0]
    issues_total:     int
    issues_resolved:  int
    false_repairs:    int
    cumulative_reward: float
    grade_breakdown:  Dict[str, float]
    passed:           bool            # score >= task pass threshold



class EnvironmentState(BaseModel):
    task_id:           str
    step:              int
    max_steps:         int
    is_done:           bool
    ground_truth_rows: List[DataRow]     # clean ground-truth (not exposed to agent)
    current_rows:      List[DataRow]     # live mutable state
    schema_:           DatasetSchema     = Field(alias="schema")
    issues_total:      int
    issues_resolved:   int
    false_repairs:     int
    visited_cells:     Dict[str, int]    # "(row,col)" -> visit_count for loop detection
    cumulative_reward: float
    episode_result:    Optional[EpisodeResult] = None

    model_config = {"populate_by_name": True}
