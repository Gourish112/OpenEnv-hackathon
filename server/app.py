"""
FastAPI server exposing the OpenEnv HTTP protocol.

Endpoints:
  POST /reset          — start a new episode
  POST /step           — apply one action
  GET  /state          — full internal environment state
  GET  /health         — liveness probe
  GET  /tasks          — list available tasks
  GET  /openenv.yaml   — serve the spec file
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ValidationError

from env import DataCleaningEnvironment, Action, TASK_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger("openenv.server")

app = FastAPI(
    title="DataCleaning OpenEnv",
    description="Production-grade data cleaning & anomaly remediation environment for LLM agents.",
    version="1.0.0",
)

# Single shared environment instance (stateful per session)
_env = DataCleaningEnvironment()



class ResetRequest(BaseModel):
    task_id: Optional[str] = "T1_hr_type_repair"


class StepRequest(BaseModel):
    action: Dict[str, Any]



@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    log.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )



@app.get("/health")
async def health():
    return {"status": "ok", "environment": "DataCleaning-OpenEnv", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "task_id":       tid,
                "difficulty":    info["difficulty"],
                "total_issues":  info["total_issues"],
                "max_steps":     info["max_steps"],
                "pass_threshold": info["pass_threshold"],
                "description":   info["description"][:120] + "...",
            }
            for tid, info in TASK_REGISTRY.items()
        ]
    }


@app.post("/reset")
async def reset(req: ResetRequest):
    """
    Reset the environment.
    Returns the initial observation conforming to OpenEnv spec.
    """
    task_id = req.task_id or "T1_hr_type_repair"
    if task_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Available: {list(TASK_REGISTRY)}",
        )
    log.info("RESET  task_id=%s", task_id)
    obs = await _env.reset(task_id=task_id)
    return obs.model_dump()


@app.post("/step")
async def step(req: StepRequest):
    """
    Apply one action.
    Returns {observation, reward, done, info} conforming to OpenEnv spec.
    """
    try:
        action = Action(**req.action)
    except (ValidationError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    log.info("STEP   action_type=%s  row=%s  col=%s",
             action.action_type, action.row_index, action.column)

    try:
        obs, reward, done, info = await _env.step(action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
async def get_state():
    """Return full internal environment state (for debugging / validators)."""
    state = await _env.state()
    return state.model_dump()


@app.get("/openenv.yaml", response_class=PlainTextResponse)
async def serve_yaml():
    """Serve the openenv.yaml spec file."""
    yaml_path = Path(__file__).parent.parent / "openenv.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    return yaml_path.read_text()




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    log.info("Starting DataCleaning OpenEnv server on %s:%d", host, port)
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="info",
    )
