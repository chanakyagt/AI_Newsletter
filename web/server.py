import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

# ── Bootstrap ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

app = FastAPI(title="NABDH Pipeline")

LOCK_FILE = ROOT / "news_output" / ".pipeline.lock"

# ── Global state ─────────────────────────────────────────────────────────────

pipeline_state: dict = {
    "status": "idle",
    "mode": None,
    "started_at": None,
    "started_at_ts": None,
    "finished_at": None,
    "current_step": None,
    "progress": 0,
    "error": None,
    "error_hint": None,
    "failed_step": None,
    "summary": {},
    "newsletter_path": None,
    "log_file": None,
}

log_queue: asyncio.Queue = None
active_process: subprocess.Popen = None
_loop: asyncio.AbstractEventLoop = None
_last_error_hint: str = ""
_log_buffer: list = []

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
    # Clear any stale lock from a previous crashed run
    if LOCK_FILE.exists():
        age = time.time() - LOCK_FILE.stat().st_mtime
        if age > 10800:
            LOCK_FILE.unlink()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _push(event: dict) -> None:
    if _loop is None or log_queue is None:
        return
    _log_buffer.append(event)
    asyncio.run_coroutine_threadsafe(_safe_put(event), _loop)


async def _safe_put(event: dict) -> None:
    try:
        log_queue.put_nowait(event)
    except asyncio.QueueFull:
        try:
            log_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            log_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _step_done(step_num: int, date: str) -> bool:
    d = ROOT / "news_output" / date
    checks = {
        1: lambda: any((d / "english").glob("*.json")),
        2: lambda: any((d / "arabic_translated").glob("*.json")),
        3: lambda: (d / "deduped" / "distinct_articles.json").exists(),
        4: lambda: (d / "scored" / "newsletter_candidates.json").exists(),
        5: lambda: (d / "firecrawled" / "firecrawled_articles.json").exists(),
        6: lambda: (d / "keypoints" / "keypoints.json").exists(),
        7: lambda: any((d / "newsletter").glob("nabdh_*.html")),
    }
    try:
        return checks.get(step_num, lambda: False)()
    except Exception:
        return False


def _find_newsletter(file: str = None) -> str | None:
    news_output = ROOT / "news_output"
    # Collect candidate newsletter dirs: dated dirs newest-first, then legacy flat dir
    dirs_to_search = []
    try:
        for d in sorted(news_output.iterdir(), reverse=True):
            if d.is_dir() and len(d.name) == 10 and d.name[4] == "-":
                nl = d / "newsletter"
                if nl.exists():
                    dirs_to_search.append(nl)
    except Exception:
        pass
    legacy = news_output / "newsletter"
    if legacy.exists():
        dirs_to_search.append(legacy)

    for newsletter_dir in dirs_to_search:
        if file:
            p = newsletter_dir / file
            if p.exists():
                return str(p)
        else:
            # Only the primary file (no backups/edits) for the "current newsletter"
            candidates = sorted(
                [p for p in newsletter_dir.glob("nabdh_*.html")
                 if "_backup_" not in p.name and "_edit_" not in p.name],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return str(candidates[0])
    return None


def _refresh_newsletter_path() -> None:
    pipeline_state["newsletter_path"] = _find_newsletter()


# ── Lock ──────────────────────────────────────────────────────────────────────

def _acquire_lock() -> bool:
    if LOCK_FILE.exists():
        age = time.time() - LOCK_FILE.stat().st_mtime
        if age < 10800:
            return False
        LOCK_FILE.unlink()
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def _release_lock() -> None:
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
        except OSError:
            pass


# ── Fresh run cleanup ─────────────────────────────────────────────────────────

def _clean_for_fresh_run(today: str) -> None:
    date_dir = ROOT / "news_output" / today
    subdirs = ["english", "arabic", "arabic_translated", "combined",
               "deduped", "scored", "firecrawled", "keypoints", "newsletter"]
    deleted = []
    for sub in subdirs:
        d = date_dir / sub
        if d.exists():
            shutil.rmtree(d)
            deleted.append(sub)

    if deleted:
        _push({"type": "log",
               "text": f"🧹 Cleared today's data: {', '.join(deleted)}",
               "step": 0, "progress": 0, "timestamp": _ts()})
    _push({"type": "log",
           "text": "✅ Fresh start — all steps will run on today's news",
           "step": 0, "progress": 0, "timestamp": _ts()})


# ── Humanise ─────────────────────────────────────────────────────────────────

def _humanise(line: str, step: int) -> str | None:
    l = line.strip()

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
    if "[EDITION]" in l:
        return f"🔢 {l.replace('[EDITION]', '').strip()}"
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


# ── Summary extraction ────────────────────────────────────────────────────────

def _extract_summary(text: str, step_num: int) -> None:
    s = pipeline_state["summary"]
    if step_num == 1 and "Found" in text and "articles" in text:
        m = re.search(r"Found (\d+) articles", text)
        if m:
            s["articles_fetched"] = s.get("articles_fetched", 0) + int(m.group(1))
    elif step_num == 3 and "Deduplication done" in text:
        m = re.search(r"(\d+) total.*?(\d+) unique", text)
        if m:
            s["articles_total"] = int(m.group(1))
            s["articles_unique"] = int(m.group(2))
    elif step_num == 4 and "Scoring done" in text:
        m = re.search(r"(\d+) articles selected", text)
        if m:
            s["articles_scored"] = int(m.group(1))
    elif step_num == 5:
        if "✓ Fetched:" in text:
            s["full_content"] = s.get("full_content", 0) + 1
        elif "RSS summary" in text:
            s["rss_fallback"] = s.get("rss_fallback", 0) + 1
    elif step_num == 6 and "Processed:" in text:
        s["keypoints"] = s.get("keypoints", 0) + 1


def _save_run_log() -> str | None:
    global _log_buffer
    if not _log_buffer:
        return None
    log_dir = ROOT / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"run_{ts}.txt"
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            for ev in _log_buffer:
                if ev.get("type") == "ping":
                    continue
                f.write(f"[{ev.get('timestamp', '')}] {ev.get('text', '')}\n")
        return str(log_path)
    except Exception:
        return None


# ── Subprocess runner ─────────────────────────────────────────────────────────

def _run_script_streaming(script_name: str, step_num: int, step_label: str, news_date: str) -> int:
    global active_process, _last_error_hint
    _last_error_hint = ""
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "NEWS_DATE": news_date,
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
        event = {
            "type": "log",
            "text": human,
            "step": step_num,
            "progress": None,
            "timestamp": _ts(),
        }
        _push(event)
        _log_buffer.append(event)
        _extract_summary(human, step_num)
        if "❌" in human or "⚠️" in human or "[ERROR]" in line or "error" in line.lower():
            _last_error_hint = human.strip().lstrip("❌⚠️ ").strip()
    active_process.wait()
    return active_process.returncode


# ── Pipeline runners ──────────────────────────────────────────────────────────

def _run_full_pipeline(force_fresh: bool = False, from_step: int = 1) -> None:
    global pipeline_state, _log_buffer
    try:
        today = _today()
        if force_fresh:
            _clean_for_fresh_run(today)
        elif from_step > 1:
            _push({"type": "log",
                   "text": f"📅 Date: {today} — resuming from Step {from_step}",
                   "step": 0, "progress": 0, "timestamp": _ts()})
        else:
            _push({"type": "log",
                   "text": f"📅 Date: {today} — resuming from last completed step",
                   "step": 0, "progress": 0, "timestamp": _ts()})

        for step_num, script, label in STEPS:
            # Skip steps explicitly before our start point
            if step_num < from_step:
                _push({
                    "type": "step_done",
                    "text": f"⏭ {label} — skipped",
                    "step": step_num,
                    "progress": STEP_PROGRESS[step_num] + 5,
                    "timestamp": _ts(),
                })
                pipeline_state["progress"] = STEP_PROGRESS[step_num] + 5
                continue

            # Resume: skip steps that already have output for today
            if not force_fresh and _step_done(step_num, today):
                _push({
                    "type": "step_done",
                    "text": f"⏭ {label} — already done today, skipping",
                    "step": step_num,
                    "progress": STEP_PROGRESS[step_num] + 5,
                    "timestamp": _ts(),
                })
                pipeline_state["progress"] = STEP_PROGRESS[step_num] + 5
                continue

            pipeline_state["current_step"] = label
            pipeline_state["progress"] = STEP_PROGRESS[step_num]
            _push({
                "type": "step_start",
                "text": label,
                "step": step_num,
                "progress": STEP_PROGRESS[step_num],
                "timestamp": _ts(),
            })

            rc = _run_script_streaming(script, step_num, label, today)

            if rc != 0:
                pipeline_state["status"] = "error"
                pipeline_state["failed_step"] = step_num
                pipeline_state["error"] = f"Step {step_num} — {label} failed"
                pipeline_state["error_hint"] = _last_error_hint or f"Process exited with code {rc}"
                pipeline_state["finished_at"] = datetime.now().isoformat()
                pipeline_state["log_file"] = _save_run_log()
                _push({
                    "type": "error",
                    "text": f"❌ {label} failed. Check logs above.",
                    "step": step_num,
                    "progress": STEP_PROGRESS[step_num],
                    "failed_step": step_num,
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
        pipeline_state["log_file"] = _save_run_log()
        _refresh_newsletter_path()
        _push({
            "type": "done",
            "text": "🎉 NABDH newsletter is ready.",
            "step": 7,
            "progress": 100,
            "timestamp": _ts(),
        })
    finally:
        _release_lock()


def _run_newsletter_only() -> None:
    global pipeline_state
    try:
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

        rc = _run_script_streaming(script, step_num, label, _today())

        if rc != 0:
            pipeline_state["status"] = "error"
            pipeline_state["failed_step"] = step_num
            pipeline_state["error"] = f"Step {step_num} — {label} failed"
            pipeline_state["error_hint"] = _last_error_hint or f"Process exited with code {rc}"
            pipeline_state["finished_at"] = datetime.now().isoformat()
            pipeline_state["log_file"] = _save_run_log()
            _push({
                "type": "error",
                "text": f"❌ {label} failed. Check logs above.",
                "step": step_num,
                "progress": STEP_PROGRESS[step_num],
                "failed_step": step_num,
                "timestamp": _ts(),
            })
            return

        pipeline_state["status"] = "done"
        pipeline_state["progress"] = 100
        pipeline_state["finished_at"] = datetime.now().isoformat()
        pipeline_state["log_file"] = _save_run_log()
        _refresh_newsletter_path()
        _push({
            "type": "done",
            "text": "🎉 Newsletter regenerated and ready.",
            "step": 7,
            "progress": 100,
            "timestamp": _ts(),
        })
    finally:
        _release_lock()


def _run_editorial_only() -> None:
    global pipeline_state, active_process
    try:
        pipeline_state["current_step"] = "Editorial Regeneration"
        pipeline_state["progress"] = 50
        _push({
            "type": "step_start",
            "text": "Regenerating editorial section...",
            "step": 7,
            "progress": 50,
            "timestamp": _ts(),
        })

        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "NEWS_DATE": _today()}
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
            pipeline_state["failed_step"] = 7
            pipeline_state["error"] = "Editorial regeneration failed"
            pipeline_state["error_hint"] = _last_error_hint or f"Process exited with code {rc}"
            pipeline_state["finished_at"] = datetime.now().isoformat()
            pipeline_state["log_file"] = _save_run_log()
            _push({
                "type": "error",
                "text": "❌ Editorial regeneration failed. Check logs above.",
                "step": 7,
                "progress": 50,
                "failed_step": 7,
                "timestamp": _ts(),
            })
            return

        pipeline_state["status"] = "done"
        pipeline_state["progress"] = 100
        pipeline_state["finished_at"] = datetime.now().isoformat()
        pipeline_state["log_file"] = _save_run_log()
        _refresh_newsletter_path()
        _push({
            "type": "done",
            "text": "🎉 Editorial section regenerated.",
            "step": 7,
            "progress": 100,
            "timestamp": _ts(),
        })
    finally:
        _release_lock()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "ui.html")


@app.get("/api/debug/env")
async def debug_env():
    """Check which pipeline env vars are present (values hidden). Safe to expose publicly."""
    keys_to_check = [
        "DeepSeek_API_Key_1", "DeepSeek_API_Key_2", "DeepSeek_API_Key_3",
        "Deepseek_API_Key_1", "Deepseek_API_Key_2", "Deepseek_API_Key_3",
        "OPENAI_API_KEY_1", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3",
        "FIRECRAWL_API_KEY",
    ]
    return {
        k: ("SET (len={})".format(len(os.environ[k])) if k in os.environ and os.environ[k] else "MISSING")
        for k in keys_to_check
    }


@app.get("/api/status")
async def get_status():
    newsletter = _find_newsletter()
    return {
        "status":           pipeline_state["status"],
        "mode":             pipeline_state["mode"],
        "progress":         pipeline_state["progress"],
        "current_step":     pipeline_state["current_step"],
        "started_at":       pipeline_state["started_at"],
        "started_at_ts":    pipeline_state["started_at_ts"],
        "finished_at":      pipeline_state["finished_at"],
        "error":            pipeline_state["error"],
        "error_hint":       pipeline_state["error_hint"],
        "failed_step":      pipeline_state["failed_step"],
        "summary":          pipeline_state["summary"],
        "newsletter_ready": newsletter is not None,
        "newsletter_name":  Path(newsletter).name if newsletter else None,
        "newsletter_path":  newsletter,
    }


def _start_thread(target) -> None:
    threading.Thread(target=target, daemon=True).start()


@app.post("/api/run")
async def run_pipeline(force: bool = False):
    global _log_buffer
    if pipeline_state["status"] == "running":
        return JSONResponse({"error": "Pipeline is already running. Wait for it to finish."}, status_code=409)
    if not _acquire_lock():
        return JSONResponse({"error": "Pipeline is already running. Wait for it to finish."}, status_code=409)
    _log_buffer = []
    now = datetime.now()
    pipeline_state.update({
        "status": "running",
        "mode": "full" if not force else "full-fresh",
        "progress": 0,
        "started_at": now.isoformat(),
        "started_at_ts": now.timestamp(),
        "finished_at": None,
        "error": None,
        "error_hint": None,
        "failed_step": None,
        "summary": {},
        "current_step": None,
        "log_file": None,
    })
    _start_thread(lambda: _run_full_pipeline(force_fresh=force))
    return {"ok": True}


@app.post("/api/run/newsletter")
async def run_newsletter():
    global _log_buffer
    if pipeline_state["status"] == "running":
        return JSONResponse({"error": "Pipeline is already running. Wait for it to finish."}, status_code=409)
    if not _acquire_lock():
        return JSONResponse({"error": "Pipeline is already running. Wait for it to finish."}, status_code=409)
    _log_buffer = []
    now = datetime.now()
    pipeline_state.update({
        "status": "running",
        "mode": "newsletter",
        "progress": 0,
        "started_at": now.isoformat(),
        "started_at_ts": now.timestamp(),
        "finished_at": None,
        "error": None,
        "error_hint": None,
        "failed_step": None,
        "summary": {},
        "current_step": None,
        "log_file": None,
    })
    _start_thread(_run_newsletter_only)
    return {"ok": True}


@app.post("/api/run/editorial")
async def run_editorial():
    global _log_buffer
    if pipeline_state["status"] == "running":
        return JSONResponse({"error": "Pipeline is already running. Wait for it to finish."}, status_code=409)
    if not _acquire_lock():
        return JSONResponse({"error": "Pipeline is already running. Wait for it to finish."}, status_code=409)
    _log_buffer = []
    now = datetime.now()
    pipeline_state.update({
        "status": "running",
        "mode": "editorial",
        "progress": 0,
        "started_at": now.isoformat(),
        "started_at_ts": now.timestamp(),
        "finished_at": None,
        "error": None,
        "error_hint": None,
        "failed_step": None,
        "summary": {},
        "current_step": None,
        "log_file": None,
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
    _release_lock()
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
async def get_newsletter(file: str = None, download: bool = False):
    if file:
        if not file.startswith("nabdh_") or "/" in file or "\\" in file:
            return JSONResponse({"error": "Invalid filename"}, status_code=400)
        found = _find_newsletter(file)
        if not found:
            return JSONResponse({"error": "File not found"}, status_code=404)
        path = Path(found)
    else:
        found = _find_newsletter()
        if not found:
            return JSONResponse({"error": "No newsletter found"}, status_code=404)
        path = Path(found)
    name = path.name
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return FileResponse(str(path), media_type="text/html", headers=headers)


@app.post("/api/newsletter/save")
async def save_newsletter_edits(request: Request):
    try:
        body         = await request.json()
        html_content = body.get("html", "")

        if len(html_content) < 10000:
            return JSONResponse({"error": "Content too short — save rejected"}, status_code=400)

        opens  = html_content.count("<div")
        closes = html_content.count("</div>")
        if opens != closes:
            return JSONResponse({
                "error": (f"HTML structure broken: {opens} opening divs, "
                          f"{closes} closing divs. Save rejected to protect your file.")
            }, status_code=400)

        found = _find_newsletter()
        if not found:
            return JSONResponse({"error": "No newsletter file found on server"}, status_code=404)

        p           = Path(found)
        backup_name = f"{p.stem}_edit_{datetime.now().strftime('%H%M%S')}{p.suffix}"
        backup_path = p.parent / backup_name
        p.rename(backup_path)

        p.write_text(html_content, encoding="utf-8")

        return JSONResponse({
            "ok":     True,
            "saved":  p.name,
            "backup": backup_name,
        })
    except Exception as e:
        return JSONResponse({"error": f"Save failed: {str(e)}"}, status_code=500)


@app.get("/api/history")
async def get_history():
    news_output = ROOT / "news_output"
    all_files: list[Path] = []

    # Search dated dirs (news_output/YYYY-MM-DD/newsletter/)
    try:
        for d in news_output.iterdir():
            if d.is_dir() and len(d.name) == 10 and d.name[4] == "-":
                nl_dir = d / "newsletter"
                if nl_dir.exists():
                    all_files.extend(nl_dir.glob("nabdh_*.html"))
    except Exception:
        pass

    # Legacy flat dir fallback (local dev)
    legacy = news_output / "newsletter"
    if legacy.exists():
        all_files.extend(legacy.glob("nabdh_*.html"))

    # Deduplicate by resolved path
    seen: set[str] = set()
    unique: list[Path] = []
    for f in all_files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)

    files = sorted(unique, key=lambda f: f.stat().st_mtime, reverse=True)[:40]

    def _classify(name: str) -> str:
        if "_backup_" in name:
            return "pre-regen"
        elif "_edit_" in name:
            return "pre-edit"
        else:
            return "latest"

    def _base_date(name: str) -> str:
        m = re.match(r"nabdh_(\d{4}-\d{2}-\d{2})", name)
        return m.group(1) if m else "unknown"

    return {"newsletters": [
        {
            "name":     f.name,
            "date":     _base_date(f.name),
            "size_kb":  round(f.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "type":     _classify(f.name),
        }
        for f in files
    ]}


@app.get("/api/health")
async def health_check():
    keys = {
        "deepseek": bool(
            os.environ.get("DeepSeek_API_Key_1") or os.environ.get("Deepseek_API_Key_1")
        ),
        "openai":   bool(os.environ.get("OPENAI_API_KEY_1")),
        "firecrawl": bool(os.environ.get("FIRECRAWL_API_KEY")),
    }
    return {"keys": keys, "all_ok": all(keys.values())}


@app.post("/api/force-unlock")
async def force_unlock():
    if pipeline_state["status"] == "running":
        return JSONResponse(
            {"error": "Pipeline appears to be actively running — cancel it first."},
            status_code=409,
        )
    _release_lock()
    pipeline_state["status"] = "idle"
    pipeline_state["error"] = None
    return {"ok": True}


@app.post("/api/run/resume")
async def run_resume(from_step: int = 1):
    global _log_buffer
    if from_step < 1 or from_step > 7:
        return JSONResponse({"error": "from_step must be between 1 and 7"}, status_code=400)
    if pipeline_state["status"] == "running":
        return JSONResponse({"error": "Pipeline is already running. Wait for it to finish."}, status_code=409)
    if not _acquire_lock():
        return JSONResponse({"error": "Pipeline is already running. Wait for it to finish."}, status_code=409)
    _log_buffer = []
    now = datetime.now()
    pipeline_state.update({
        "status": "running",
        "mode": f"resume-from-{from_step}",
        "progress": 0,
        "started_at": now.isoformat(),
        "started_at_ts": now.timestamp(),
        "finished_at": None,
        "error": None,
        "error_hint": None,
        "failed_step": None,
        "summary": {},
        "current_step": None,
        "log_file": None,
    })
    _start_thread(lambda: _run_full_pipeline(force_fresh=False, from_step=from_step))
    return {"ok": True}


@app.get("/api/logs/latest")
async def get_latest_log():
    log_path = pipeline_state.get("log_file")
    if not log_path or not Path(log_path).exists():
        log_dir = ROOT / "pipeline_logs"
        if not log_dir.exists():
            return JSONResponse({"error": "No log files found yet"}, status_code=404)
        logs = sorted(log_dir.glob("run_*.txt"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not logs:
            return JSONResponse({"error": "No log files found yet"}, status_code=404)
        log_path = str(logs[0])
    name = Path(log_path).name
    return FileResponse(log_path, media_type="text/plain",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})


# RAILWAY DEPLOYMENT CHECKLIST
# ─────────────────────────────
# 1. All .env variables must be added in Railway dashboard → Variables
#    Required: Deepseek_API_Key_1/2/3, OPENAI_API_KEY_1/2/3, FIRECRAWL_API_KEY
#    Optional: NABDH_EDITION_OVERRIDE (set to "1" for first real run, then remove)
#
# 2. news_output/ is created fresh on first run — no need to upload it
#
# 3. edition_counter.json is created automatically on first newsletter
#
# 4. To reset edition number: set NABDH_EDITION_OVERRIDE=1 in Railway Variables
#    for one run, then delete that variable so auto-increment resumes
#
# 5. Logs are streamed live via SSE — no log files needed on Railway
#
# 6. Pipeline takes 18-25 minutes — Railway's HTTP timeout is 5 minutes
#    SSE keepalive pings every 15 seconds prevent the connection being killed
#    The pipeline runs in a background thread — timeout only affects the stream
#    not the pipeline itself. User can refresh and reconnect via /api/status check.
