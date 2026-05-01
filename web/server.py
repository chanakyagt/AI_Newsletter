import asyncio
import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

# ── Bootstrap ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

app = FastAPI(title="NABDH Pipeline")

# ── Global state ─────────────────────────────────────────────────────────────

pipeline_state: dict = {
    "status": "idle",
    "mode": None,
    "started_at": None,
    "finished_at": None,
    "current_step": None,
    "progress": 0,
    "error": None,
    "newsletter_path": None,
}

log_queue: asyncio.Queue = None
active_process: subprocess.Popen = None
_loop: asyncio.AbstractEventLoop = None

# ── Steps ─────────────────────────────────────────────────────────────────────

STEPS = [
    (1, "uae_ai_news.py",               "Step 1 — Fetching UAE AI News"),
    (2, "uae_ai_translate.py",          "Step 2 — Translating Arabic Articles"),
    (3, "uae_ai_semantic_dedup_v2.py",  "Step 3 — Removing Duplicate Stories"),
    (4, "uae_ai_scorer_v2.py",          "Step 4 — Scoring Articles for Relevance"),
    (5, "uae_firecrawl_v1.py",          "Step 5 — Fetching Full Article Content"),
    (6, "uae_ai_keypoints_v2.py",       "Step 6 — Extracting Intelligence"),
    (7, "nabdh_newsletter_v2.py",       "Step 7 — Generating Newsletter"),
]

STEP_PROGRESS = {1: 5, 2: 20, 3: 35, 4: 50, 5: 65, 6: 80, 7: 95}

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    global log_queue, _loop
    _loop = asyncio.get_running_loop()
    log_queue = asyncio.Queue(maxsize=1000)
    _refresh_newsletter_path()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _push(event: dict) -> None:
    if _loop is None or log_queue is None:
        return
    asyncio.run_coroutine_threadsafe(
        _safe_put(event), _loop
    )


async def _safe_put(event: dict) -> None:
    try:
        log_queue.put_nowait(event)
    except asyncio.QueueFull:
        # Drop oldest to make room
        try:
            log_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            log_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


def _find_newsletter() -> str | None:
    pattern = ROOT / "news_output" / "newsletter" / "nabdh_*.html"
    candidates = sorted(
        Path(ROOT / "news_output" / "newsletter").glob("nabdh_*.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def _refresh_newsletter_path() -> None:
    pipeline_state["newsletter_path"] = _find_newsletter()


# ── Humanise ─────────────────────────────────────────────────────────────────

def _humanise(line: str, step: int) -> str | None:
    l = line.strip()

    # Suppress decorative separator lines
    if ("═" * 5 in l) or ("─" * 5 in l) or l.startswith("===") or l.startswith("---"):
        return None

    # Step 1: News Fetch
    if "[THEME]" in l:
        theme = l.split("[THEME]")[-1].strip()
        names = {
            "gov_policy":      "Government & Policy",
            "investment":      "Investment & Finance",
            "enterprise_ai":   "Enterprise AI",
            "infrastructure":  "Infrastructure & Data Centres",
            "talent":          "Talent & Education",
            "startup":         "Startups & Ventures",
            "geopolitics":     "Geopolitics & Strategy",
            "research":        "Research & Innovation",
        }
        return f"📡 Scanning topic: {names.get(theme, theme)}"
    if "→" in l and "articles" in l:
        digits = "".join(filter(str.isdigit, l.split("→")[1].split("articles")[0]))
        if digits:
            return f"   └─ Found {digits} articles"
    if "total_articles_deduplicated" in l or "WRITING COMBINED" in l:
        return "✅ News fetch complete — combining all sources"
    if "FETCHING ENGLISH" in l:
        return "🌐 Fetching English language news feeds..."
    if "FETCHING ARABIC" in l:
        return "🌐 Fetching Arabic language news feeds..."

    # Step 2: Translation
    if "Translating" in l and "article" in l:
        return f"🔤 {l}"
    if "translated" in l.lower() and "article" in l.lower():
        return f"✅ {l}"

    # Step 3: Deduplication
    if "Loading from" in l:
        return "📂 Loading articles for deduplication..."
    if "Embedding" in l or "embedding" in l:
        return "🧮 Running semantic similarity check (finding duplicate stories)..."
    if "distinct" in l.lower() or "unique" in l.lower():
        nums = [s for s in l.split() if s.replace(",", "").isdigit()]
        if len(nums) >= 2:
            return f"✅ Deduplication done — {nums[0]} total → {nums[-1]} unique stories"
        return f"✅ {l}"

    # Step 4: Scoring
    if "[INPUT]" in l and "articles" in l:
        digits = "".join(filter(str.isdigit, l.split("articles")[0].split()[-1] if l.split("articles") else ""))
        if digits:
            return f"📊 Scoring {digits} articles for C-suite relevance..."
        return "📊 Scoring articles for C-suite relevance..."
    if "[QUEUE]" in l and "unscored" in l:
        return f"⚙️  {l.replace('[QUEUE]', '').strip()}"
    if "newsletter_candidates" in l or "candidates" in l.lower():
        nums = [s for s in l.split() if s.replace(",", "").isdigit()]
        if nums:
            return f"✅ Scoring done — {nums[0]} articles selected for newsletter"
    if "CHECKPOINT" in l:
        return f"💾 Progress saved ({l.split('CHECKPOINT')[-1].strip()})"

    # Step 5: Firecrawl
    if "QUEUE" in l and "to fetch" in l:
        n = l.split("to fetch")[0].split()[-1] if "to fetch" in l else "?"
        return f"🕷️  Fetching full article content for {n} articles..."
    if "FALLBACK" in l:
        return "   └─ ⚠ Used RSS summary (full page unavailable)"
    if "FIRECRAWL COMPLETE" in l:
        return "✅ Content fetch complete"
    if " OK " in l and ("chars" in l or "c  w=" in l):
        title = l.split("OK")[-1].strip()[:50]
        return f"   └─ ✓ Fetched: {title}"

    # Step 6: Keypoints
    if "[INPUT]" in l and "articles loaded" in l:
        return "🧠 Extracting intelligence from articles..."
    if "✓" in l and "pts" in l:
        parts = l.split("✓")
        if len(parts) > 1:
            rest = parts[-1].strip()
            title = rest.split("]")[-1].strip()[:55] if "]" in rest else rest[:55]
            return f"   └─ Processed: {title}"
    if "keypoints" in l.lower() and ("saved" in l.lower() or "written" in l.lower()):
        return "✅ Keypoint extraction complete"

    # Step 7: Newsletter
    if "Generating editorial" in l or "[1/3]" in l:
        return "✍️  Writing editorial..."
    if "EDITORIAL" in l and "chars" in l:
        return "   └─ Editorial drafted"
    if "Generating hooks" in l or "[2/3]" in l:
        return "✍️  Writing story hooks..."
    if "Generating closing" in l or "[3/3]" in l:
        return "✍️  Writing closing section..."
    if "nabdh_" in l and ".html" in l:
        return "✅ Newsletter HTML generated"
    if "SAVED" in l and ".html" in l:
        return "📄 Newsletter file saved"

    # Generic
    if "[ERROR]" in l:
        return f"❌ {l.replace('[ERROR]', '').strip()}"
    if "[WARN]" in l:
        return f"⚠️  {l.replace('[WARN]', '').strip()}"

    return l


# ── Subprocess runner ─────────────────────────────────────────────────────────

def _run_script_streaming(script_name: str, step_num: int, step_label: str) -> int:
    global active_process
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    active_process = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=ROOT,
    )
    for raw_line in active_process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        human = _humanise(line, step_num)
        if human is None:
            continue
        _push({
            "type": "log",
            "text": human,
            "step": step_num,
            "progress": None,
            "timestamp": _ts(),
        })
    active_process.wait()
    return active_process.returncode


# ── Pipeline runners ──────────────────────────────────────────────────────────

def _run_full_pipeline() -> None:
    global pipeline_state
    for step_num, script, label in STEPS:
        pipeline_state["current_step"] = label
        pipeline_state["progress"] = STEP_PROGRESS[step_num]
        _push({
            "type": "step_start",
            "text": label,
            "step": step_num,
            "progress": STEP_PROGRESS[step_num],
            "timestamp": _ts(),
        })

        rc = _run_script_streaming(script, step_num, label)

        if rc != 0:
            pipeline_state["status"] = "error"
            pipeline_state["error"] = f"{label} failed (exit code {rc})"
            pipeline_state["finished_at"] = datetime.now().isoformat()
            _push({
                "type": "error",
                "text": f"❌ {label} failed. Check logs above.",
                "step": step_num,
                "progress": STEP_PROGRESS[step_num],
                "timestamp": _ts(),
            })
            return

        _push({
            "type": "step_done",
            "text": f"✅ {label} complete",
            "step": step_num,
            "progress": STEP_PROGRESS[step_num] + 5,
            "timestamp": _ts(),
        })

    pipeline_state["status"] = "done"
    pipeline_state["progress"] = 100
    pipeline_state["finished_at"] = datetime.now().isoformat()
    _refresh_newsletter_path()
    _push({
        "type": "done",
        "text": "🎉 NABDH newsletter is ready.",
        "step": 7,
        "progress": 100,
        "timestamp": _ts(),
    })


def _run_newsletter_only() -> None:
    global pipeline_state
    step_num, script, label = STEPS[6]  # Step 7
    pipeline_state["current_step"] = label
    pipeline_state["progress"] = STEP_PROGRESS[step_num]
    _push({
        "type": "step_start",
        "text": label,
        "step": step_num,
        "progress": STEP_PROGRESS[step_num],
        "timestamp": _ts(),
    })

    rc = _run_script_streaming(script, step_num, label)

    if rc != 0:
        pipeline_state["status"] = "error"
        pipeline_state["error"] = f"{label} failed (exit code {rc})"
        pipeline_state["finished_at"] = datetime.now().isoformat()
        _push({
            "type": "error",
            "text": f"❌ {label} failed. Check logs above.",
            "step": step_num,
            "progress": STEP_PROGRESS[step_num],
            "timestamp": _ts(),
        })
        return

    pipeline_state["status"] = "done"
    pipeline_state["progress"] = 100
    pipeline_state["finished_at"] = datetime.now().isoformat()
    _refresh_newsletter_path()
    _push({
        "type": "done",
        "text": "🎉 Newsletter regenerated and ready.",
        "step": 7,
        "progress": 100,
        "timestamp": _ts(),
    })


def _run_editorial_only() -> None:
    global pipeline_state, active_process
    pipeline_state["current_step"] = "Editorial Regeneration"
    pipeline_state["progress"] = 50
    _push({
        "type": "step_start",
        "text": "Regenerating editorial section...",
        "step": 7,
        "progress": 50,
        "timestamp": _ts(),
    })

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    active_process = subprocess.Popen(
        [sys.executable, "run_pipeline.py", "--redo-editorial"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=ROOT,
    )
    for raw_line in active_process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        human = _humanise(line, 7)
        if human is None:
            continue
        _push({
            "type": "log",
            "text": human,
            "step": 7,
            "progress": None,
            "timestamp": _ts(),
        })
    active_process.wait()
    rc = active_process.returncode

    if rc != 0:
        pipeline_state["status"] = "error"
        pipeline_state["error"] = f"Editorial regeneration failed (exit code {rc})"
        pipeline_state["finished_at"] = datetime.now().isoformat()
        _push({
            "type": "error",
            "text": "❌ Editorial regeneration failed. Check logs above.",
            "step": 7,
            "progress": 50,
            "timestamp": _ts(),
        })
        return

    pipeline_state["status"] = "done"
    pipeline_state["progress"] = 100
    pipeline_state["finished_at"] = datetime.now().isoformat()
    _refresh_newsletter_path()
    _push({
        "type": "done",
        "text": "🎉 Editorial section regenerated.",
        "step": 7,
        "progress": 100,
        "timestamp": _ts(),
    })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "ui.html")


@app.get("/api/status")
async def status():
    return {**pipeline_state}


def _start_thread(target) -> None:
    threading.Thread(target=target, daemon=True).start()


@app.post("/api/run")
async def run_pipeline():
    if pipeline_state["status"] == "running":
        return JSONResponse({"error": "Pipeline is already running"}, status_code=409)
    pipeline_state.update({
        "status": "running",
        "mode": "full",
        "progress": 0,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "error": None,
        "current_step": None,
    })
    _start_thread(_run_full_pipeline)
    return {"ok": True}


@app.post("/api/run/newsletter")
async def run_newsletter():
    if pipeline_state["status"] == "running":
        return JSONResponse({"error": "Pipeline is already running"}, status_code=409)
    pipeline_state.update({
        "status": "running",
        "mode": "newsletter",
        "progress": 0,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "error": None,
        "current_step": None,
    })
    _start_thread(_run_newsletter_only)
    return {"ok": True}


@app.post("/api/run/editorial")
async def run_editorial():
    if pipeline_state["status"] == "running":
        return JSONResponse({"error": "Pipeline is already running"}, status_code=409)
    pipeline_state.update({
        "status": "running",
        "mode": "editorial",
        "progress": 0,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "error": None,
        "current_step": None,
    })
    _start_thread(_run_editorial_only)
    return {"ok": True}


@app.post("/api/cancel")
async def cancel_pipeline():
    global active_process
    if active_process and active_process.poll() is None:
        active_process.terminate()
        pipeline_state["status"] = "idle"
        pipeline_state["error"] = "Cancelled by user"
        _push({
            "type": "error",
            "text": "⛔ Pipeline cancelled.",
            "step": None,
            "progress": pipeline_state["progress"],
            "timestamp": _ts(),
        })
    return {"ok": True}


@app.get("/api/stream")
async def stream(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                line = await asyncio.wait_for(log_queue.get(), timeout=15.0)
                yield {"data": json.dumps(line)}
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"type": "ping"})}
    return EventSourceResponse(event_generator())


@app.get("/api/newsletter")
async def get_newsletter(download: bool = False):
    path = _find_newsletter()
    if not path:
        return JSONResponse({"error": "No newsletter found"}, status_code=404)
    name = Path(path).name
    if download:
        return FileResponse(
            path,
            media_type="text/html",
            filename=name,
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )
    return FileResponse(path, media_type="text/html")
