from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Tactix API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data directory ──
DATA_DIR = Path.home() / ".tactix" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory state (replace with DB later) ──
projects: dict[str, dict[str, Any]] = {}
# Active WebSocket connections per project
progress_sockets: dict[str, list[WebSocket]] = {}


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Project CRUD ──

@app.post("/api/projects")
async def create_project(name: str = "Untitled") -> dict[str, Any]:
    project_id = str(uuid.uuid4())[:8]
    project_dir = DATA_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    project = {
        "id": project_id,
        "name": name,
        "status": "created",
        "video_path": None,
        "tracking_path": None,
    }
    projects[project_id] = project
    return project


@app.get("/api/projects")
async def list_projects() -> list[dict[str, Any]]:
    return list(projects.values())


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    if project_id not in projects:
        return JSONResponse({"error": "not found"}, status_code=404)
    return projects[project_id]


# ── Video upload ──

@app.post("/api/projects/{project_id}/upload")
async def upload_video(project_id: str, file: UploadFile) -> dict[str, Any]:
    if project_id not in projects:
        return JSONResponse({"error": "not found"}, status_code=404)

    project_dir = DATA_DIR / project_id
    video_path = project_dir / (file.filename or "video.mp4")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    projects[project_id]["video_path"] = str(video_path)
    projects[project_id]["status"] = "uploaded"
    return {"video_path": str(video_path)}


# ── Video streaming ──

@app.get("/api/projects/{project_id}/video")
async def stream_video(project_id: str) -> FileResponse:
    if project_id not in projects:
        return JSONResponse({"error": "not found"}, status_code=404)  # type: ignore[return-value]
    video_path = projects[project_id].get("video_path")
    if not video_path or not Path(video_path).exists():
        return JSONResponse({"error": "no video"}, status_code=404)  # type: ignore[return-value]
    return FileResponse(video_path, media_type="video/mp4")


# ── Processing ──

@app.post("/api/projects/{project_id}/process")
async def start_processing(project_id: str) -> dict[str, str]:
    """Start the TactixEngine processing in a background thread."""
    import threading

    if project_id not in projects:
        return JSONResponse({"error": "not found"}, status_code=404)  # type: ignore[return-value]

    project = projects[project_id]
    if not project.get("video_path"):
        return JSONResponse({"error": "no video uploaded"}, status_code=400)  # type: ignore[return-value]

    project["status"] = "processing"

    def run_engine() -> None:
        try:
            from tactix.config import Config
            from tactix.engine.system import TactixEngine

            cfg = Config()
            cfg.INPUT_VIDEO = project["video_path"]

            project_dir = DATA_DIR / project_id
            cfg.OUTPUT_VIDEO = str(project_dir / "result.mp4")
            cfg.EXPORT_VIEWER_JSON = True
            cfg.VIEWER_JSON_PATH = str(project_dir / "viewer_data.json")

            engine = TactixEngine(cfg)

            # Hook into engine progress for WebSocket broadcasting
            original_process = engine._process_frame
            frame_count = [0]

            def hooked_process(*args: Any, **kwargs: Any) -> Any:
                result = original_process(*args, **kwargs)
                frame_count[0] += 1
                _broadcast_progress(project_id, frame_count[0], engine.total_frames)
                return result

            engine._process_frame = hooked_process  # type: ignore[attr-defined]
            engine.run()

            project["tracking_path"] = str(project_dir / "viewer_data.json")
            project["output_video"] = str(project_dir / "result.mp4")
            project["status"] = "completed"
            _broadcast_progress(project_id, engine.total_frames, engine.total_frames, done=True)
        except Exception as e:
            project["status"] = "error"
            project["error"] = str(e)
            _broadcast_progress(project_id, 0, 0, error=str(e))

    thread = threading.Thread(target=run_engine, daemon=True)
    thread.start()

    return {"status": "processing"}


def _broadcast_progress(
    project_id: str, frame: int, total: int, done: bool = False, error: str | None = None
) -> None:
    """Send progress to all connected WebSocket clients for a project."""
    import asyncio

    sockets = progress_sockets.get(project_id, [])
    msg = json.dumps({"frame": frame, "total": total, "done": done, "error": error})
    for ws in sockets:
        try:
            asyncio.run(ws.send_text(msg))
        except Exception:
            pass


# ── WebSocket for progress ──

@app.websocket("/ws/projects/{project_id}/progress")
async def ws_progress(websocket: WebSocket, project_id: str) -> None:
    await websocket.accept()
    progress_sockets.setdefault(project_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        progress_sockets[project_id].remove(websocket)


# ── Tracking data ──

@app.get("/api/projects/{project_id}/tracking")
async def get_tracking(project_id: str) -> Any:
    if project_id not in projects:
        return JSONResponse({"error": "not found"}, status_code=404)

    tracking_path = projects[project_id].get("tracking_path")
    if not tracking_path or not Path(tracking_path).exists():
        return JSONResponse({"error": "no tracking data"}, status_code=404)

    with open(tracking_path) as f:
        return json.load(f)


# ── Serve entry point ──

def serve() -> None:
    import uvicorn
    uvicorn.run("tactix.api.main:app", host="0.0.0.0", port=8000, reload=True)
