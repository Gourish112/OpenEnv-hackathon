#!/usr/bin/env python3
"""
inference.py — Baseline agent for the DataCleaning OpenEnv environment.

Protocol
--------
Output strictly follows the OpenEnv stdout format:
  [START]   task_id=<id>  max_steps=<n>
  [STEP]    step=<n>  action=<json>  reward=<f>  done=<bool>
  [END]     score=<f>  passed=<bool>  steps=<n>  resolved=<n>/<total>

Environment variables read:
  API_BASE_URL   — base URL of the OpenEnv server  (default: http://localhost:8000)
  MODEL_NAME     — LLM to use via OpenAI-compatible API
  HF_TOKEN       — Bearer token for HF Inference API (if using HF endpoint)
  TASK_ID        — which task to run  (default: T1_hr_type_repair)
  MAX_STEPS      — override max_steps from server
  OPENAI_API_KEY — fallback if HF_TOKEN not set
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv() 
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME:   str = os.environ.get("MODEL_NAME","Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
TASK_ID:      str = os.environ.get("TASK_ID", "T1_hr_type_repair")
OPENAI_KEY:   str = os.environ.get("OPENAI_API_KEY")
VERBOSE:      bool = os.environ.get("VERBOSE", "0").lower() in ("1", "true", "yes")
MAX_STEPS_OVERRIDE = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1



api_key = os.getenv("HF_TOKEN")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
)


SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})
if HF_TOKEN:
    SESSION.headers.update({"Authorization": f"Bearer {HF_TOKEN}"})


def _post(endpoint: str, payload: dict) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    resp = SESSION.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _get(endpoint: str) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()

SYSTEM_PROMPT = """You are a data-quality agent. Your job is to clean a dirty tabular dataset
by issuing precise repair actions one at a time.

You must respond with ONLY a valid JSON object representing your next action.
Do not include any prose, markdown, or code fences — pure JSON only.

Available action_types:
  fix_type        — cast cell to correct dtype; requires row_index, column, target_dtype
  fix_value       — replace cell value; requires row_index, column, new_value
  fill_missing    — impute null cell; requires row_index, column, new_value
  remove_duplicate — drop duplicate row; requires row_index
  remove_outlier  — drop outlier row; requires row_index
  rename_column   — fix column header; requires column (current), new_name (correct)
  drop_column     — remove invalid column; requires column
  drop_row        — remove conflict row; requires row_index
  validate        — declare dataset clean and end episode
  skip            — no-op (avoid using this)

target_dtype options: int, float, string, boolean, date

Always include a "reasoning" field explaining why you are taking this action.
Be precise and efficient. Every unnecessary action reduces your score.

Respond with exactly one action JSON per message. Example:
{"action_type": "fix_type", "row_index": 0, "column": "age", "target_dtype": "int", "reasoning": "age is stored as string '34' but schema requires INT"}
"""


def build_user_prompt(obs: dict) -> str:
    dataset = obs.get("dataset", {})
    rows = dataset.get("rows", [])
    schema = dataset.get("schema", {})
    hints = obs.get("hints", [])
    constraints = obs.get("visible_constraints", [])

    # Format rows compactly
    rows_str = json.dumps(rows, indent=2)[:6000]  # truncate for context window

    return f"""
=== TASK ===
{obs.get("task_description", "")}

=== STEP {obs.get("step")} / {obs.get("max_steps")} ===
Issues remaining: {obs.get("issues_remaining")}
Issues resolved:  {obs.get("issues_resolved")}
False repairs:    {obs.get("false_repairs")}
Last feedback:    {obs.get("last_action_feedback", "")}

=== SCHEMA ===
Columns: {schema.get("columns", [])}
Dtypes:  {schema.get("dtypes", {})}
Required: {schema.get("required", [])}
Unique keys: {schema.get("unique_keys", [])}
Value constraints: {schema.get("value_constraints", {})}

=== CONSTRAINTS ===
{chr(10).join(f"  • {c}" for c in constraints)}

=== HINTS ===
{json.dumps(hints, indent=2) if hints else "No hints provided."}

=== DATASET (current state) ===
{rows_str}

What is your next action? Respond with a single JSON object only.
""".strip()

def call_llm(conversation: List[Dict[str, str]]) -> str:
    """Call the LLM and return the raw text response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation,
        temperature=0.0,   # deterministic
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str) -> Optional[Dict[str, Any]]:
    """Extract JSON action from LLM response."""
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find the first JSON object
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return None

rewards: List[float] = []
def run_episode(task_id: str) -> dict:
    """Run a single episode and return the result dict."""

    # ── RESET ──────────────────────────────────────────────────────────────
    obs_data = _post("/reset", {"task_id": task_id})
    max_steps = MAX_STEPS_OVERRIDE or obs_data.get("max_steps", 30)

    print(f"[START] task={task_id} env=data_cleaning model={MODEL_NAME}", flush=True)
    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    cumulative_reward = 0.0
    step_count = 0
    done = False
    final_info: dict = {}

    # ── STEP LOOP ──────────────────────────────────────────────────────────
    for step_num in range(1, max_steps + 1):
        if done:
            break

        # Build prompt from current observation
        user_msg = build_user_prompt(obs_data)
        conversation.append({"role": "user", "content": user_msg})

        # Get LLM action
        try:
            raw_response = call_llm(conversation)
        except Exception as e:
            step_reward = 0.0
            rewards.append(step_reward)

            print(
                f"[STEP] step={step_num} action=null reward=0.00 done=false error=LLM_CALL_FAILED",
                flush=True,
            )

            done = False  # allow next step to proceed
            continue
        if VERBOSE:
            print(f"  [LLM]  {raw_response[:200]}", file=sys.stderr)

        action_dict = parse_action(raw_response)
        if action_dict is None:
            # Fallback: skip action to avoid hard crash
            action_dict = {"action_type": "skip", "reasoning": "LLM response was not valid JSON"}

        # Add assistant message to conversation
        conversation.append({"role": "assistant", "content": raw_response})

        # ── STEP ───────────────────────────────────────────────────────────
        try:
            step_result = _post("/step", {"action": action_dict})
        except Exception as e:
            print(
                f"[STEP] step={step_num} action=null reward=0.00 done=false error=LLM_CALL_FAILED",
                flush=True,
            )
            break

        obs_data     = step_result.get("observation", obs_data)
        reward_data  = step_result.get("reward", {})
        done         = step_result.get("done", False)
        info         = step_result.get("info", {})
        final_info   = info

        step_reward  = reward_data.get("total", 0.0)
        cumulative_reward += step_reward
        step_count = step_num

        action_json = json.dumps(action_dict)
        rewards.append(step_reward)

        print(
            f"[STEP] step={step_num} "
            f"action={action_json} "
            f"reward={step_reward:.2f} "
            f"done={str(done).lower()} "
            f"error=null",
            flush=True,
        )

        if done:
            break

    # ── If not done yet, force VALIDATE ───────────────────────────────────
    if not done:
        try:
            step_result = _post("/step", {"action": {"action_type": "validate",
                                                     "reasoning": "max_steps reached"}})
            obs_data  = step_result.get("observation", obs_data)
            done      = True
            final_info = step_result.get("info", final_info)
            step_count += 1
            rewards.append(step_result.get('reward', {}).get('total', 0.0))

            print(
                f"[STEP] step={step_count} "
                f"action={{\"action_type\":\"validate\"}} "
                f"reward={step_result.get('reward', {}).get('total', 0.0):.2f} "
                f"done=true "
                f"error=null",
                flush=True,
            )
        except Exception:
            pass

    # ── END ────────────────────────────────────────────────────────────────
    episode_result = final_info.get("episode_result") or {}
    score          = episode_result.get("final_score", 0.0)
    passed         = episode_result.get("passed", False)
    resolved       = episode_result.get("issues_resolved", 0)
    total_issues   = episode_result.get("issues_total", "?")

    score = min(max(score, 0.0), 1.0)
    success = score >= SUCCESS_SCORE_THRESHOLD

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={step_count} "
        f"score={score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )

    return {
        "task_id":          task_id,
        "score":            score,
        "passed":           passed,
        "steps":            step_count,
        "issues_resolved":  resolved,
        "issues_total":     total_issues,
        "cumulative_reward": cumulative_reward,
        "episode_result":   episode_result,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DataCleaning OpenEnv baseline agent")
    parser.add_argument("--task", default=TASK_ID,
                        choices=["T1_hr_type_repair", "T2_sales_multi_issue", "T3_financial_hard", "all"],
                        help="Task to run (or 'all' for full benchmark)")
    parser.add_argument("--wait", type=float, default=1.0,
                        help="Seconds to wait between episodes when running all tasks")
    args = parser.parse_args()

    tasks_to_run = (
        ["T1_hr_type_repair", "T2_sales_multi_issue", "T3_financial_hard"]
        if args.task == "all"
        else [args.task]
    )

    all_results = []
    for tid in tasks_to_run:
        try:
            result = run_episode(tid)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] task={tid}  error={e}", flush=True)
            traceback.print_exc()
        if len(tasks_to_run) > 1:
            time.sleep(args.wait)

    if len(all_results) > 1:
        avg_score = sum(r["score"] for r in all_results) / len(all_results)
        n_passed  = sum(1 for r in all_results if r["passed"])
        print(
            f"\n[BENCHMARK]  tasks={len(all_results)}"
            f"  avg_score={avg_score:.4f}"
            f"  passed={n_passed}/{len(all_results)}",
            flush=True,
        )
