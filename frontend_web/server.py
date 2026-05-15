from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if load_dotenv:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / "dev" / ".env")

from open_coscientist import HypothesisGenerator

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUN_QUEUES: Dict[str, asyncio.Queue] = {}
RUN_TASKS: Dict[str, asyncio.Task] = {}


class RunRequest(BaseModel):
    research_goal: str = Field(..., min_length=1)
    model_name: str = Field("claude-haiku-4-5-20251001")
    max_iterations: int = Field(1, ge=0, le=5)
    initial_hypotheses_count: int = Field(3, ge=1, le=12)
    evolution_max_count: int = Field(3, ge=0, le=10)
    enable_literature_review_node: bool = True
    enable_tool_calling_generation: bool = False
    cited_papers: list[str] = Field(default_factory=list)


async def publish(run_id: str, event: str, data: Dict[str, Any]) -> None:
    queue = RUN_QUEUES.get(run_id)
    if queue is None:
        return
    await queue.put({"event": event, "data": data})


async def run_workflow(run_id: str, request: RunRequest) -> None:
    queue = RUN_QUEUES.get(run_id)
    if queue is None:
        return

    last_state: Dict[str, Any] = {}

    async def progress_callback(phase: str, data: Dict[str, Any]) -> None:
        await publish(run_id, "log", {"phase": phase, "data": data})

    try:
        generator = HypothesisGenerator(
            model_name=request.model_name,
            max_iterations=request.max_iterations,
            initial_hypotheses_count=request.initial_hypotheses_count,
            evolution_max_count=request.evolution_max_count,
        )

        opts = {
            "enable_literature_review_node": request.enable_literature_review_node,
            "enable_tool_calling_generation": request.enable_tool_calling_generation,
            "user_inputs": {"literature": request.cited_papers},
        }

        async for node_name, state in generator.generate_hypotheses(
            research_goal=request.research_goal,
            progress_callback=progress_callback,
            opts=opts,
            stream=True,
        ):
            last_state = state
            await publish(run_id, "step", {"node": node_name, "state": state})

        await publish(run_id, "done", {"status": "completed", "state": last_state})
    except Exception as exc:
        await publish(run_id, "error", {"message": str(exc)})
    finally:
        await queue.put(None)
        await cleanup_run(run_id)


async def cleanup_run(run_id: str) -> None:
    await asyncio.sleep(1)
    RUN_QUEUES.pop(run_id, None)
    task = RUN_TASKS.pop(run_id, None)
    current = asyncio.current_task()
    if task and task is not current and not task.done():
        task.cancel()


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/run")
async def start_run(request: RunRequest) -> JSONResponse:
    run_id = str(uuid.uuid4())
    RUN_QUEUES[run_id] = asyncio.Queue()
    RUN_TASKS[run_id] = asyncio.create_task(run_workflow(run_id, request))
    return JSONResponse({"run_id": run_id})


@app.get("/api/stream/{run_id}")
async def stream(run_id: str) -> StreamingResponse:
    queue = RUN_QUEUES.get(run_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        while True:
            event = await queue.get()
            if event is None:
                break
            name = event.get("event", "message")
            payload = json.dumps(event.get("data", {}), ensure_ascii=False)
            yield f"event: {name}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


static_dir = Path(__file__).resolve().parent
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("COSCIENTIST_WEB_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
