#!/usr/bin/env python3
"""
NABDH Pipeline Orchestrator
Runs all 7 steps of the UAE AI Intelligence newsletter pipeline.
"""

import sys
import os
import subprocess
import time
import json
import argparse
import traceback
import re
import html
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Bootstrap: ensure dotenv is available before anything else
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
except ImportError:
    print("\n  [ERROR] python-dotenv is not installed.")
    print("  Fix:    pip install python-dotenv\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PIPELINE_DIR = Path(__file__).parent.resolve()
ENV_FILE = PIPELINE_DIR / ".env"
LOG_DIR = PIPELINE_DIR / "pipeline_logs"

REQUIRED_KEYS = [
    "Deepseek_API_Key_1",
    "Deepseek_API_Key_2",
    "Deepseek_API_Key_3",
    "OPENAI_API_KEY_1",
    "OPENAI_API_KEY_2",
    "OPENAI_API_KEY_3",
    "FIRECRAWL_API_KEY",
]

STEPS = [
    {
        "num": 1,
        "name": "News Fetch",
        "script": "uae_ai_news.py",
        "purpose": "Fetch UAE AI news from Google News RSS (English + Arabic)",
        "max_retries": 2,
        "retry_wait": 30,
        "input_check": None,
        "output_check": "step1",
    },
    {
        "num": 2,
        "name": "Arabic Translation",
        "script": "uae_ai_translate.py",
        "purpose": "Translate Arabic articles to English",
        "max_retries": 2,
        "retry_wait": 45,
        "input_check": None,
        "output_check": "step2",
    },
    {
        "num": 3,
        "name": "Deduplication",
        "script": "uae_ai_semantic_dedup_v2.py",
        "purpose": "Remove duplicate stories using OpenAI embeddings",
        "max_retries": 1,
        "retry_wait": 60,
        "input_check": PIPELINE_DIR / "news_output" / "deduped",
        "output_check": "step3",
    },
    {
        "num": 4,
        "name": "Scoring",
        "script": "uae_ai_scorer_v2.py",
        "purpose": "Score articles for C-suite relevance via DeepSeek",
        "max_retries": 1,
        "retry_wait": 60,
        "input_check": PIPELINE_DIR / "news_output" / "deduped" / "distinct_articles.json",
        "output_check": "step4",
    },
    {
        "num": 5,
        "name": "Firecrawl",
        "script": "uae_firecrawl_v1.py",
        "purpose": "Fetch full article content via Firecrawl",
        "max_retries": 2,
        "retry_wait": 45,
        "input_check": PIPELINE_DIR / "news_output" / "scored" / "newsletter_candidates.json",
        "output_check": "step5",
    },
    {
        "num": 6,
        "name": "Keypoints",
        "script": "uae_ai_keypoints_v2.py",
        "purpose": "Extract intelligence keypoints from full content via DeepSeek",
        "max_retries": 1,
        "retry_wait": 60,
        "input_check": PIPELINE_DIR / "news_output" / "firecrawled" / "firecrawled_articles.json",
        "output_check": "step6",
    },
    {
        "num": 7,
        "name": "Newsletter",
        "script": "nabdh_newsletter_v2.py",
        "purpose": "Generate the final HTML newsletter",
        "max_retries": 2,
        "retry_wait": 30,
        "input_check": PIPELINE_DIR / "news_output" / "keypoints" / "keypoints.json",
        "output_check": "step7",
    },
]

STEP_FAILURE_HINTS = {
    1: [
        "No internet connection — check your network",
        "Google News RSS may be temporarily unavailable — try again in a few minutes",
    ],
    2: [
        "DeepSeek API key is invalid or expired — check Deepseek_API_Key_1/2/3 in .env",
        "No Arabic articles were found to translate (check news_output/arabic/ folder)",
        "DeepSeek API rate limit hit — wait a few minutes and retry",
    ],
    3: [
        "OpenAI API key is invalid or expired — check OPENAI_API_KEY_1/2/3 in .env",
        "Not enough articles after Step 1/2 — the news fetch may have returned very few results",
    ],
    4: [
        "DeepSeek API key is invalid or expired — check Deepseek_API_Key_1/2/3 in .env",
        "distinct_articles.json is missing or empty — re-run from Step 3",
    ],
    5: [
        "Firecrawl API key is invalid or expired — check FIRECRAWL_API_KEY in .env",
        "newsletter_candidates.json is missing or empty — re-run from Step 4",
        "Network issue — no internet connection",
    ],
    6: [
        "DeepSeek API key is invalid or expired — check Deepseek_API_Key_1/2/3 in .env",
        "firecrawled_articles.json is missing or empty — re-run from Step 5",
    ],
    7: [
        "DeepSeek API key is invalid or expired — check Deepseek_API_Key_1 in .env",
        "keypoints.json is missing or empty — re-run from Step 6",
    ],
}

# ---------------------------------------------------------------------------
# Logger: writes to both terminal and log file simultaneously
# ---------------------------------------------------------------------------
class PipelineLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, "w", encoding="utf-8", buffering=1)

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def write(self, text: str, *, to_log: bool = True, end: str = "\n", flush: bool = False):
        """Print to terminal and optionally write to log."""
        print(text, end=end, flush=flush)
        if to_log:
            for line in (text + end).splitlines(keepends=True):
                self._fh.write(f"[{self._ts()}] {line}")
            if flush:
                self._fh.flush()

    def write_raw(self, text: str):
        """Write a raw subprocess output line (already has newline)."""
        print(text, end="", flush=True)
        self._fh.write(f"[{self._ts()}] {text}")
        self._fh.flush()

    def write_inline(self, text: str):
        """Overwrite current line (for countdown). Does NOT go to log."""
        print(f"\r{text}", end="", flush=True)

    def close(self):
        self._fh.flush()
        self._fh.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mask_key(value: str) -> str:
    """Show only last 4 chars of an API key."""
    if not value or len(value) <= 4:
        return "••••"
    return "••••••••" + value[-4:]


def dot_pad(label: str, width: int = 32) -> str:
    dots = max(1, width - len(label))
    return label + " " + ("." * dots) + " "


def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def load_json_safe(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def count_json_articles(path: Path) -> str:
    data = load_json_safe(path)
    if data is None:
        return "count unavailable"
    if isinstance(data, list):
        return str(len(data))
    if isinstance(data, dict):
        for key in ("articles", "items", "results", "data"):
            if key in data and isinstance(data[key], list):
                return str(len(data[key]))
        return str(len(data))
    return "count unavailable"


def count_json_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.json")))


def find_newsletter_html() -> Path | None:
    folder = PIPELINE_DIR / "news_output" / "newsletter"
    if not folder.exists():
        return None
    files = sorted(folder.glob("nabdh_*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------
def verify_step_output(step_num: int, logger: PipelineLogger, arabic_was_empty: bool = False) -> tuple[bool, str]:
    """Returns (passed, message)."""
    news = PIPELINE_DIR / "news_output"

    if step_num == 1:
        folder = news / "english"
        if not folder.exists() or count_json_files(folder) == 0:
            return False, "news_output/english/ has no JSON files"
        # Check at least one article in the first file
        files = list(folder.glob("*.json"))
        data = load_json_safe(files[0])
        if not data or (isinstance(data, list) and len(data) == 0):
            return False, "news_output/english/ JSON files appear empty"
        return True, ""

    if step_num == 2:
        if arabic_was_empty:
            return True, ""
        folder = news / "arabic_translated"
        if not folder.exists() or count_json_files(folder) == 0:
            return False, "news_output/arabic_translated/ has no JSON files"
        return True, ""

    if step_num == 3:
        path = news / "deduped" / "distinct_articles.json"
        if not path.exists():
            return False, "distinct_articles.json does not exist"
        data = load_json_safe(path)
        if data is None:
            return False, "distinct_articles.json is not valid JSON"
        count = len(data) if isinstance(data, list) else 0
        if count < 100:
            return False, f"distinct_articles.json has only {count} articles (minimum: 100)"
        return True, ""

    if step_num == 4:
        path = news / "scored" / "newsletter_candidates.json"
        if not path.exists():
            return False, "newsletter_candidates.json does not exist"
        data = load_json_safe(path)
        if data is None:
            return False, "newsletter_candidates.json is not valid JSON"
        count = len(data) if isinstance(data, list) else 0
        if count < 10:
            return False, f"newsletter_candidates.json has only {count} articles (minimum: 10)"
        return True, ""

    if step_num == 5:
        path = news / "firecrawled" / "firecrawled_articles.json"
        if not path.exists():
            return False, "firecrawled_articles.json does not exist"
        data = load_json_safe(path)
        if data is None:
            return False, "firecrawled_articles.json is not valid JSON"
        count = len(data) if isinstance(data, list) else 0
        if count < 10:
            return False, f"firecrawled_articles.json has only {count} articles (minimum: 10)"
        return True, ""

    if step_num == 6:
        path = news / "keypoints" / "keypoints.json"
        if not path.exists():
            return False, "keypoints.json does not exist"
        data = load_json_safe(path)
        if data is None:
            return False, "keypoints.json is not valid JSON"
        count = len(data) if isinstance(data, list) else 0
        if count < 5:
            return False, f"keypoints.json has only {count} keypoints (minimum: 5)"
        return True, ""

    if step_num == 7:
        html = find_newsletter_html()
        if html is None:
            return False, "No nabdh_*.html file found in news_output/newsletter/"
        size = html.stat().st_size
        if size < 50 * 1024:
            return False, f"Newsletter HTML is only {size // 1024}KB (minimum: 50KB)"
        return True, ""

    return True, ""


# ---------------------------------------------------------------------------
# Step summary stats (article counts for final report)
# ---------------------------------------------------------------------------
def get_step_summary_stat(step_num: int) -> str:
    news = PIPELINE_DIR / "news_output"
    try:
        if step_num == 1:
            total = 0
            for f in (news / "combined").glob("*.json"):
                d = load_json_safe(f)
                if isinstance(d, list):
                    total += len(d)
            if total == 0:
                # fall back to counting per-source files
                eng = count_json_files(news / "english")
                arb = count_json_files(news / "arabic")
                return f"{eng} English + {arb} Arabic files"
            return f"{total:,} articles fetched"

        if step_num == 2:
            count = count_json_files(news / "arabic_translated")
            return f"{count} files translated"

        if step_num == 3:
            path = news / "deduped" / "distinct_articles.json"
            data = load_json_safe(path)
            after = len(data) if isinstance(data, list) else 0
            # try to get before count from combined
            before = 0
            for f in (news / "combined").glob("*.json"):
                d = load_json_safe(f)
                if isinstance(d, list):
                    before += len(d)
            if before:
                return f"{before:,} → {after:,} distinct"
            return f"{after:,} distinct articles"

        if step_num == 4:
            path = news / "scored" / "newsletter_candidates.json"
            data = load_json_safe(path)
            count = len(data) if isinstance(data, list) else 0
            return f"{count} newsletter candidates"

        if step_num == 5:
            path = news / "firecrawled" / "firecrawled_articles.json"
            data = load_json_safe(path)
            if isinstance(data, list):
                full = sum(1 for a in data if a.get("full_content") or a.get("content"))
                fallback = len(data) - full
                return f"{full} full content / {fallback} fallback"
            return "count unavailable"

        if step_num == 6:
            path = news / "keypoints" / "keypoints.json"
            data = load_json_safe(path)
            count = len(data) if isinstance(data, list) else 0
            return f"{count} keypoints extracted"

        if step_num == 7:
            path = news / "keypoints" / "keypoints.json"
            data = load_json_safe(path)
            count = len(data) if isinstance(data, list) else 0
            return f"{count} stories in final issue"

    except Exception:
        pass
    return "count unavailable"


# ---------------------------------------------------------------------------
# Countdown display
# ---------------------------------------------------------------------------
def countdown(seconds: int, logger: PipelineLogger):
    bar_width = 20
    for remaining in range(seconds, 0, -1):
        filled = int((seconds - remaining) / seconds * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        logger.write_inline(f"  [WAIT]  {bar}  {remaining}s remaining...  ")
        time.sleep(1)
    print()  # newline after countdown


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
def run_preflight(logger: PipelineLogger) -> bool:
    logger.write("")
    logger.write("  PRE-FLIGHT CHECKS")
    logger.write("  ─────────────────")
    all_ok = True

    # .env file
    if ENV_FILE.exists():
        logger.write("  [✓] .env file found")
    else:
        logger.write(f"  [✗] .env file NOT found at: {ENV_FILE}")
        logger.write("      Create a .env file in the same folder as run_pipeline.py")
        all_ok = False

    # Load env
    load_dotenv(ENV_FILE, override=True)

    # API keys
    for key in REQUIRED_KEYS:
        value = os.environ.get(key, "")
        if value:
            logger.write(f"  [✓] {dot_pad(key, 30)}{mask_key(value)}")
        else:
            logger.write(f"  [✗] {dot_pad(key, 30)}NOT SET")
            logger.write(f"      Add   {key}=your_key_here   to your .env file")
            all_ok = False

    # Script files
    missing_scripts = []
    for step in STEPS:
        script_path = PIPELINE_DIR / step["script"]
        if not script_path.exists():
            missing_scripts.append(step["script"])
    if missing_scripts:
        for s in missing_scripts:
            logger.write(f"  [✗] Script missing: {s}")
        all_ok = False
    else:
        logger.write("  [✓] All 7 step scripts found")

    # Python packages
    packages = {"dotenv": "python-dotenv", "openai": "openai", "requests": "requests", "numpy": "numpy"}
    missing_pkgs = []
    for mod, pkg in packages.items():
        try:
            __import__(mod)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        logger.write(f"  [✗] Missing Python packages: {', '.join(missing_pkgs)}")
        logger.write(f"      Fix: pip install {' '.join(missing_pkgs)}")
        all_ok = False
    else:
        logger.write("  [✓] Python packages: dotenv, openai, requests, numpy ✓")

    # Internet check
    try:
        import requests as req
        req.get("https://news.google.com", timeout=5)
        logger.write("  [✓] Internet connection OK")
    except Exception:
        logger.write("  [!] Internet check FAILED — Step 1 may fail (network unavailable)")

    return all_ok


# ---------------------------------------------------------------------------
# Run a single step
# ---------------------------------------------------------------------------
def run_step(
    step: dict,
    logger: PipelineLogger,
    arabic_was_empty: bool = False,
) -> tuple[bool, float, str]:
    """
    Returns (success, duration_seconds, summary_stat).
    """
    script_path = PIPELINE_DIR / step["script"]
    num = step["num"]
    max_retries = step["max_retries"]
    retry_wait = step["retry_wait"]
    total_attempts = max_retries + 1

    duration = 0.0

    for attempt in range(1, total_attempts + 1):
        if attempt > 1:
            logger.write(
                f"\n  [RETRY] Step {num} failed (exit code {last_rc}). "
                f"Attempt {attempt} of {total_attempts} in {retry_wait} seconds..."
            )
            logger.write(f"  [RETRY] Reason: Script exited with non-zero code. Check output above.")
            countdown(retry_wait, logger)

        # Header
        now_str = datetime.now().strftime("%H:%M:%S")
        logger.write("")
        logger.write("  ──────────────────────────────────────────────────────────────")
        logger.write(f"  STEP {num} of 7 — {step['name'].upper()}")
        logger.write(f"  Script  : {step['script']}")
        logger.write(f"  Purpose : {step['purpose']}")
        logger.write(f"  Started : {now_str}")
        logger.write("  ──────────────────────────────────────────────────────────────")
        logger.write("")

        start = time.time()
        last_rc = 0

        try:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(PIPELINE_DIR),
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            for line in proc.stdout:
                logger.write_raw(line)

            proc.wait()
            last_rc = proc.returncode
        except Exception as exc:
            logger.write(f"\n  [ERROR] Failed to launch {step['script']}: {exc}")
            last_rc = -1

        duration = time.time() - start

        if last_rc != 0:
            continue  # retry

        # Output verification
        ok, msg = verify_step_output(num, logger, arabic_was_empty=arabic_was_empty)
        if not ok:
            logger.write(f"\n  [WARN] Step {num} exited successfully but output verification failed:")
            logger.write(f"         {msg}")
            logger.write(f"  [RETRY] Retrying step {num}...")
            last_rc = -99
            if attempt == total_attempts:
                break
            continue

        # Success
        summary_stat = get_step_summary_stat(num)
        logger.write("")
        logger.write(f"  STEP {num} COMPLETE ✓")
        logger.write(f"  Duration  : {fmt_duration(duration)}")
        logger.write(f"  Result    : {summary_stat}")
        _print_step_output_details(num, logger)
        logger.write("  ──────────────────────────────────────────────────────────────")
        return True, duration, summary_stat

    return False, duration, ""


def _print_step_output_details(step_num: int, logger: PipelineLogger):
    news = PIPELINE_DIR / "news_output"
    try:
        if step_num == 1:
            eng = count_json_files(news / "english")
            arb = count_json_files(news / "arabic")
            logger.write(f"  Output    : news_output/english/  ({eng} files)")
            logger.write(f"              news_output/arabic/   ({arb} files)")
            combined = news / "combined" / "all_articles.json"
            if combined.exists():
                c = count_json_articles(combined)
                logger.write(f"              news_output/combined/all_articles.json  ({c} articles)")
        elif step_num == 2:
            c = count_json_files(news / "arabic_translated")
            logger.write(f"  Output    : news_output/arabic_translated/  ({c} files)")
        elif step_num == 3:
            p = news / "deduped" / "distinct_articles.json"
            logger.write(f"  Output    : news_output/deduped/distinct_articles.json  ({count_json_articles(p)} articles)")
        elif step_num == 4:
            p = news / "scored" / "newsletter_candidates.json"
            logger.write(f"  Output    : news_output/scored/newsletter_candidates.json  ({count_json_articles(p)} articles)")
        elif step_num == 5:
            p = news / "firecrawled" / "firecrawled_articles.json"
            logger.write(f"  Output    : news_output/firecrawled/firecrawled_articles.json  ({count_json_articles(p)} articles)")
        elif step_num == 6:
            p = news / "keypoints" / "keypoints.json"
            logger.write(f"  Output    : news_output/keypoints/keypoints.json  ({count_json_articles(p)} keypoints)")
        elif step_num == 7:
            html = find_newsletter_html()
            if html:
                logger.write(f"  Output    : {html.relative_to(PIPELINE_DIR)}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Failure report
# ---------------------------------------------------------------------------
def print_failure_report(
    failed_step: dict,
    completed: list,
    logger: PipelineLogger,
    run_label: str,
):
    num = failed_step["num"]
    logger.write("")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.write(f"  ✗ PIPELINE STOPPED — STEP {num} FAILED")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.write("")
    logger.write("  What happened:")
    logger.write(f"    Step {num} ({failed_step['script']}) failed after {failed_step['max_retries'] + 1} attempts.")
    logger.write("")
    logger.write("  What was completed before the failure:")

    for step in STEPS:
        snum = step["num"]
        if snum in completed:
            marker = "[✓]"
            label = f"completed"
        elif snum == num:
            marker = "[✗]"
            label = f"FAILED after {failed_step['max_retries'] + 1} attempts"
        else:
            marker = "[ ]"
            label = "not started"
        logger.write(f"    {marker} Step {snum} — {step['name']:<25} ({label})")

    logger.write("")
    logger.write("  Most likely causes:")
    for hint in STEP_FAILURE_HINTS.get(num, ["Unknown error — review output above"]):
        logger.write(f"    • {hint}")

    logger.write("")
    logger.write("  What you can do:")
    logger.write("    1. Fix the issue above")
    logger.write(f"    2. Run:  python run_pipeline.py --start-from {num}")
    logger.write(f"       This will skip Step 1–{num - 1} (already done) and resume from Step {num}.")
    logger.write("")
    logger.write(f"  Log saved to: pipeline_logs/{LOG_DIR.name}/run_{run_label}.log")
    logger.write("  ══════════════════════════════════════════════════════════════")


# ---------------------------------------------------------------------------
# Success report
# ---------------------------------------------------------------------------
def print_success_report(
    results: list,
    total_duration: float,
    run_date: str,
    logger: PipelineLogger,
    log_name: str,
):
    logger.write("")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.write("  ✓ NABDH PIPELINE COMPLETE")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.write("")
    logger.write(f"  Run date   : {run_date}")
    logger.write(f"  Total time : {fmt_duration(total_duration)}")
    logger.write("")
    logger.write("  STEP SUMMARY")
    logger.write("  ────────────")
    for step_num, duration, stat in results:
        step = next(s for s in STEPS if s["num"] == step_num)
        dur_str = fmt_duration(duration).rjust(10)
        logger.write(f"  [✓] Step {step_num} — {step['name']:<25} {dur_str}    {stat}")
    logger.write("")
    logger.write("  OUTPUT")
    logger.write("  ──────")
    html = find_newsletter_html()
    if html:
        rel = html.relative_to(PIPELINE_DIR)
        logger.write(f"  Newsletter : {rel}")
        logger.write("  Open the file above in any browser to preview your newsletter.")
    else:
        logger.write("  Newsletter : (file not found — check news_output/newsletter/)")
    logger.write("")
    logger.write(f"  Log file   : pipeline_logs/{log_name}")
    logger.write("  ══════════════════════════════════════════════════════════════")


# ---------------------------------------------------------------------------
# Input validation for --start-from / --steps
# ---------------------------------------------------------------------------
def validate_step_input(step: dict, logger: PipelineLogger) -> bool:
    check = step.get("input_check")
    if check is None:
        return True
    path = Path(check)
    if not path.exists():
        logger.write(f"\n  [ERROR] Cannot start from Step {step['num']} — required input not found:")
        logger.write(f"          {path}")
        logger.write(f"  You need to run Step {step['num'] - 1} first.")
        return False
    return True


# ---------------------------------------------------------------------------
# Arabic empty check
# ---------------------------------------------------------------------------
def is_arabic_folder_empty() -> bool:
    folder = PIPELINE_DIR / "news_output" / "arabic"
    if not folder.exists():
        return True
    return count_json_files(folder) == 0


# ---------------------------------------------------------------------------
# Depth-aware div block finder — never use regex for nested HTML
# ---------------------------------------------------------------------------
def _find_div_block(html_str: str, open_tag: str) -> tuple[int, int]:
    """
    Finds (start, end) of a div block that begins with open_tag.
    Uses a depth counter so nested <div> tags never confuse the search.
    Returns (-1, -1) if not found or if the block is unclosed.
    """
    start = html_str.find(open_tag)
    if start == -1:
        return -1, -1
    depth, i = 0, start
    while i < len(html_str):
        if html_str[i:i+4] == '<div':
            depth += 1; i += 4
        elif html_str[i:i+6] == '</div>':
            depth -= 1
            if depth == 0:
                return start, i + 6
            i += 6
        else:
            i += 1
    return -1, -1


def _build_editorial_block(thesis: str, body_paras: list, date_str: str) -> str:
    """
    Builds a complete, self-contained <div class="editorial"> block.
    Every LLM-generated string is HTML-escaped before insertion so stray
    < or > in the AI output cannot break the page structure.
    thesis    : plain text for ed-kicker
    body_paras: list of plain-text paragraphs (up to 2 used)
    date_str  : e.g. "30 April 2026"
    """
    import html as _hl
    safe_thesis = _hl.escape(thesis)
    para_html = ""
    for i, para in enumerate(body_paras[:2]):
        safe = _hl.escape(para)
        if i == 0:
            para_html += (
                f'  <p class="ed-p">'
                f'<span class="ed-dropcap">{safe[0]}</span>{safe[1:]}'
                f'</p>\n'
            )
        else:
            para_html += f'  <p class="ed-p" style="clear:left;">{safe}</p>\n'
    return (
        '<div class="editorial">\n'
        '  <span class="ed-from-lbl">From the Editor</span>\n'
        f'  <div class="ed-thesis-wrap">'
        f'<span class="ed-thesis-lbl">This Edition&rsquo;s Thesis</span>'
        f'<p class="ed-kicker">{safe_thesis}</p>'
        f'</div>'
        f'<div class="ed-rule"></div>\n'
        f'{para_html}'
        f'  <div class="ed-sig">&mdash; NABDH Editorial Team'
        f' &nbsp;&middot;&nbsp; {date_str}</div>\n'
        '</div>'
    )


def _check_div_balance(html_str: str, context: str = "") -> bool:
    """Returns True if <div> open and close counts match exactly."""
    opens = html_str.count('<div')
    closes = html_str.count('</div>')
    if opens != closes:
        print(f"  [ERROR] Div mismatch {context}: {opens} opens vs {closes} closes")
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers shared by the surgical redo operations
# ---------------------------------------------------------------------------
def _make_op_logger() -> tuple:
    """Create a timestamped log file for a one-off operation."""
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f"run_{run_ts}.log"
    log_path = LOG_DIR / log_name
    return PipelineLogger(log_path), log_name


def _find_current_newsletter_html() -> Path | None:
    """Most recent non-backup newsletter HTML."""
    folder = PIPELINE_DIR / "news_output" / "newsletter"
    if not folder.exists():
        return None
    files = sorted(
        [p for p in folder.glob("nabdh_*.html") if "_backup_" not in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _backup_html(html_path: Path) -> Path:
    """Rename html_path to a timestamped backup in the same folder."""
    ts_suffix = datetime.now().strftime("%H%M%S")
    backup_name = f"{html_path.stem}_backup_{ts_suffix}.html"
    backup_path = html_path.parent / backup_name
    html_path.rename(backup_path)
    return backup_path


def _stream_subprocess(script_path: Path, logger: PipelineLogger) -> int:
    """Run script as subprocess with live streaming. Returns exit code."""
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(PIPELINE_DIR),
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    for line in proc.stdout:
        logger.write_raw(line)
    proc.wait()
    return proc.returncode


# ---------------------------------------------------------------------------
# --redo-newsletter
# ---------------------------------------------------------------------------
def run_redo_newsletter():
    try:
        _run_redo_newsletter_impl()
    except KeyboardInterrupt:
        print("\n\n  [INTERRUPTED] Stopped by user.")
        sys.exit(130)
    except Exception:
        tb = traceback.format_exc()
        print("\n\n  [FATAL] Unexpected error in --redo-newsletter:")
        print(tb)
        sys.exit(1)


def _run_redo_newsletter_impl():
    logger, log_name = _make_op_logger()
    load_dotenv(ENV_FILE, override=True)

    logger.write("")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.write("  NABDH — NEWSLETTER REGENERATION")
    logger.write("  Reusing keypoints: news_output/keypoints/keypoints.json")
    logger.write("  Previous newsletter will be backed up before overwriting.")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.write("")

    # Validate keypoints exist and are non-empty
    keypoints_path = PIPELINE_DIR / "news_output" / "keypoints" / "keypoints.json"
    if not keypoints_path.exists():
        logger.write("  [ERROR] No keypoints file found.")
        logger.write("          Run the full pipeline first: python run_pipeline.py")
        logger.write("          Or resume from Step 6:       python run_pipeline.py --start-from 6")
        logger.close()
        sys.exit(1)
    kp_data = load_json_safe(keypoints_path)
    if not kp_data or (isinstance(kp_data, list) and len(kp_data) == 0):
        logger.write("  [ERROR] keypoints.json is empty.")
        logger.write("          Re-run from Step 6: python run_pipeline.py --start-from 6")
        logger.close()
        sys.exit(1)

    # Backup existing newsletter if present
    existing = _find_current_newsletter_html()
    if existing:
        backup_path = _backup_html(existing)
        logger.write(f"  [OK] Backup saved: {backup_path.relative_to(PIPELINE_DIR)}")
        logger.write("")

    # Run nabdh_newsletter_v2.py
    script_path = PIPELINE_DIR / "nabdh_newsletter_v2.py"
    if not script_path.exists():
        logger.write("  [ERROR] nabdh_newsletter_v2.py not found in pipeline folder.")
        logger.close()
        sys.exit(1)

    now_str = datetime.now().strftime("%H:%M:%S")
    logger.write("  ──────────────────────────────────────────────────────────────")
    logger.write("  Running: nabdh_newsletter_v2.py")
    logger.write(f"  Started: {now_str}")
    logger.write("  ──────────────────────────────────────────────────────────────")
    logger.write("")

    start = time.time()
    try:
        rc = _stream_subprocess(script_path, logger)
    except Exception as exc:
        logger.write(f"\n  [ERROR] Failed to launch nabdh_newsletter_v2.py: {exc}")
        logger.close()
        sys.exit(1)
    duration = time.time() - start

    if rc != 0:
        logger.write("")
        logger.write(f"  [ERROR] nabdh_newsletter_v2.py exited with code {rc}.")
        logger.write("          Check the output above for details.")
        logger.write(f"  Log saved to: pipeline_logs/{log_name}")
        logger.close()
        sys.exit(1)

    # Verify new HTML exists and is big enough
    new_html_file = _find_current_newsletter_html()
    if new_html_file is None:
        logger.write("")
        logger.write("  [ERROR] Newsletter HTML not found after generation.")
        logger.write("          nabdh_newsletter_v2.py may have failed silently.")
        logger.close()
        sys.exit(1)

    size = new_html_file.stat().st_size
    if size < 50 * 1024:
        logger.write("")
        logger.write(f"  [ERROR] Generated HTML is only {size // 1024}KB (minimum: 50KB).")
        logger.write("          The newsletter may be incomplete. Check the script output above.")
        logger.close()
        sys.exit(1)

    rel = new_html_file.relative_to(PIPELINE_DIR)
    logger.write("")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.write("  [OK] Newsletter regenerated successfully.")
    logger.write(f"  File: {rel}")
    logger.write("  Open this file in any browser to preview.")
    logger.write(f"  Duration  : {fmt_duration(duration)}")
    logger.write(f"  Log saved : pipeline_logs/{log_name}")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.close()


# ---------------------------------------------------------------------------
# --redo-editorial
# ---------------------------------------------------------------------------
def run_redo_editorial():
    try:
        _run_redo_editorial_impl()
    except KeyboardInterrupt:
        print("\n\n  [INTERRUPTED] Stopped by user.")
        sys.exit(130)
    except Exception:
        tb = traceback.format_exc()
        print("\n\n  [FATAL] Unexpected error in --redo-editorial:")
        print(tb)
        sys.exit(1)


def _run_redo_editorial_impl():
    logger, log_name = _make_op_logger()
    load_dotenv(ENV_FILE, override=True)

    # Validate both inputs exist before printing the header
    keypoints_path = PIPELINE_DIR / "news_output" / "keypoints" / "keypoints.json"
    newsletter_file = _find_current_newsletter_html()

    if not keypoints_path.exists():
        print("  [ERROR] No keypoints file found.")
        print("          Run the full pipeline first: python run_pipeline.py")
        print("          Or resume from Step 6:       python run_pipeline.py --start-from 6")
        logger.close()
        sys.exit(1)

    if newsletter_file is None:
        print("  [ERROR] No newsletter HTML found in news_output/newsletter/")
        print("          Run the full pipeline or --redo-newsletter first.")
        logger.close()
        sys.exit(1)

    rel_html = newsletter_file.relative_to(PIPELINE_DIR)
    rel_kp = keypoints_path.relative_to(PIPELINE_DIR)

    logger.write("")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.write("  NABDH — EDITORIAL REGENERATION ONLY")
    logger.write(f"  Newsletter : {rel_html}")
    logger.write(f"  Keypoints  : {rel_kp}")
    logger.write("  Only the editorial section will be replaced.")
    logger.write("  Stories, hooks, tracker, closing stay unchanged.")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.write("")

    # Load keypoints, extract up to 12 titles
    keypoints = load_json_safe(keypoints_path)
    if not keypoints or not isinstance(keypoints, list):
        logger.write("  [ERROR] keypoints.json is empty or not a list.")
        logger.close()
        sys.exit(1)

    # Build signals string matching nabdh_newsletter_v2.py format (headline + signal)
    top5 = keypoints[:5]
    signals = "\n".join(
        f"- {kp.get('headline_reframe', '')} | {kp.get('the_signal', '')}"
        for kp in top5
        if kp.get("headline_reframe") or kp.get("title")
    )
    if not signals.strip():
        logger.write("  [ERROR] No signal data found in keypoints.json.")
        logger.write("          The keypoints file may be malformed.")
        logger.close()
        sys.exit(1)

    # Load newsletter HTML before calling the API so we can balance-check it first
    try:
        html_content = newsletter_file.read_text(encoding="utf-8")
    except Exception as exc:
        logger.write(f"  [ERROR] Could not read newsletter file: {exc}")
        logger.close()
        sys.exit(1)

    if not _check_div_balance(html_content, "existing newsletter"):
        logger.write("  [WARN] Existing newsletter already has unbalanced divs.")
        logger.write("         Proceeding, but consider --redo-newsletter for a clean start.")

    # Validate DeepSeek key
    api_key = os.environ.get("Deepseek_API_Key_1", "")
    if not api_key:
        logger.write("  [ERROR] Deepseek_API_Key_1 is not set in .env")
        logger.write("          Add it to your .env file and try again.")
        logger.close()
        sys.exit(1)

    try:
        from openai import OpenAI as _OpenAI
    except ImportError:
        logger.write("  [ERROR] openai package is not installed.")
        logger.write("          Fix: pip install openai")
        logger.close()
        sys.exit(1)

    client = _OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    system_prompt = (
        "You are the editor of NABDH — UAE's most read AI intelligence brief for "
        "ministers, sovereign fund directors, and C-suite executives.\n"
        "Write with authority and precision. Every word must earn its place.\n"
        "Plain text only. No markdown. No asterisks. No labels. No preamble."
    )
    user_prompt = (
        f"This week's top UAE AI signals:\n{signals}\n\n"
        "Write exactly 3 short paragraphs separated by a blank line. Nothing else.\n\n"
        "Paragraph 1 — Thesis (25 words max):\n"
        "One declarative sentence stating the single most important shift these signals "
        "represent. Start with a fact or a verdict. No scene-setting.\n\n"
        "Paragraph 2 — Evidence (45 words max):\n"
        "Name the 2 most consequential developments. State what each means operationally "
        "for a senior decision-maker. No adjectives like \"significant\" or \"landmark\".\n\n"
        "Paragraph 3 — Action (35 words max):\n"
        "One thing alert executives are doing right now. End with one uncomfortably "
        "specific strategic question every UAE board must answer before the next edition."
    )

    # Call DeepSeek, up to 3 attempts
    editorial_text = ""
    paragraphs: list[str] = []
    MAX_ATTEMPTS = 3

    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.write(f"  Calling DeepSeek... (attempt {attempt} of {MAX_ATTEMPTS})")
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=250,
                temperature=0.75,
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.write(f"  [WARN] DeepSeek API error on attempt {attempt}: {exc}")
            raw = ""

        if len(raw) < 120:
            logger.write(f"  [WARN] Response too short ({len(raw)} chars). Retrying...")
            if attempt < MAX_ATTEMPTS:
                time.sleep(5)
            continue

        paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if len(paras) < 2:
            logger.write(f"  [WARN] Too few paragraphs ({len(paras)}). Retrying...")
            if attempt < MAX_ATTEMPTS:
                time.sleep(5)
            continue

        editorial_text = raw
        paragraphs = paras
        break

    if not paragraphs:
        logger.write("")
        logger.write("  [ERROR] DeepSeek returned empty or too-short editorial after 3 attempts.")
        logger.write("          Check your Deepseek_API_Key_1 in .env and try again.")
        logger.close()
        sys.exit(1)

    logger.write(f"\n  [OK] Editorial received — {len(editorial_text)} chars")

    # Build replacement block: para 0 → thesis, paras 1-2 → body
    today_str = datetime.now().strftime("%d %B %Y").lstrip("0")
    new_editorial_block = _build_editorial_block(paragraphs[0], paragraphs[1:3], today_str)

    # Locate editorial div with depth-counter — never regex
    ed_start, ed_end = _find_div_block(html_content, '<div class="editorial">')
    if ed_start == -1:
        logger.write('  [ERROR] Could not locate <div class="editorial"> in the newsletter HTML.')
        logger.write("          The file may have been manually edited.")
        logger.write("          Run --redo-newsletter to regenerate the full newsletter instead.")
        logger.close()
        sys.exit(1)

    new_html_content = html_content[:ed_start] + new_editorial_block + html_content[ed_end:]

    # Verify div balance before touching disk
    if not _check_div_balance(new_html_content, "after editorial injection"):
        logger.write("  [ABORT] HTML structure is broken — file NOT saved.")
        logger.write("          Run --redo-newsletter to do a clean regeneration instead.")
        logger.close()
        sys.exit(1)

    # Backup original, write new content to original filename
    backup_path = _backup_html(newsletter_file)

    try:
        newsletter_file.write_text(new_html_content, encoding="utf-8")
    except Exception as exc:
        logger.write(f"  [ERROR] Could not write newsletter file: {exc}")
        logger.write(f"          Backup preserved at: {backup_path.relative_to(PIPELINE_DIR)}")
        logger.close()
        sys.exit(1)

    # Print preview — word-wrap at ~57 chars
    preview_raw = editorial_text[:300]
    words = preview_raw.split()
    preview_lines: list[str] = []
    current_line = ""
    for word in words:
        if current_line and len(current_line) + 1 + len(word) > 57:
            preview_lines.append(current_line)
            current_line = word
        else:
            current_line = (current_line + " " + word).lstrip()
    if current_line:
        preview_lines.append(current_line)

    logger.write("")
    logger.write("  PREVIEW")
    logger.write("  ─────────────────────────────────────────────────────────")
    for line in preview_lines:
        logger.write(f"  {line}")
    logger.write("  ─────────────────────────────────────────────────────────")
    logger.write("")
    logger.write("  [OK] Editorial injected successfully")
    logger.write(f"  [OK] Backup : {backup_path.relative_to(PIPELINE_DIR)}")
    logger.write(f"  [OK] Saved  : {rel_html}")
    logger.write("")
    logger.write("  Open the file in your browser to review the new editorial.")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="NABDH UAE AI Intelligence Newsletter Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    Run full pipeline
  python run_pipeline.py --start-from 4    Skip steps 1-3, start at scoring
  python run_pipeline.py --steps 1,2       Run only fetch and translate
  python run_pipeline.py --dry-run         Pre-flight checks only
  python run_pipeline.py --redo-newsletter Regenerate newsletter from saved keypoints
  python run_pipeline.py --redo-editorial  Regenerate only the editorial section
  python run_pipeline.py --help            Show this help
        """,
    )
    parser.add_argument("--start-from", type=int, metavar="N", help="Start from step N (1-7)")
    parser.add_argument("--steps", type=str, metavar="N,N,...", help="Run only specific steps, e.g. 1,3,5")
    parser.add_argument("--dry-run", action="store_true", help="Pre-flight checks only, do not run pipeline")
    parser.add_argument("--redo-newsletter", action="store_true",
                        help="Regenerate newsletter from saved keypoints (skips steps 1-6)")
    parser.add_argument("--redo-editorial", action="store_true",
                        help="Regenerate only the editorial section in the existing newsletter")
    args = parser.parse_args()

    # Surgical redo flags bypass pre-flight and step loop entirely
    if args.redo_editorial:
        run_redo_editorial()
        return

    if args.redo_newsletter:
        run_redo_newsletter()
        return

    # Determine which steps to run
    if args.steps:
        try:
            step_nums = [int(x.strip()) for x in args.steps.split(",")]
        except ValueError:
            print(f"  [ERROR] --steps must be comma-separated integers, e.g. --steps 1,3,5")
            sys.exit(1)
        steps_to_run = [s for s in STEPS if s["num"] in step_nums]
        if not steps_to_run:
            print("  [ERROR] No valid steps specified.")
            sys.exit(1)
    elif args.start_from:
        n = args.start_from
        if not 1 <= n <= 7:
            print(f"  [ERROR] --start-from must be between 1 and 7.")
            sys.exit(1)
        steps_to_run = [s for s in STEPS if s["num"] >= n]
    else:
        steps_to_run = STEPS[:]

    # Create log file
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f"run_{run_ts}.log"
    log_path = LOG_DIR / log_name
    logger = PipelineLogger(log_path)

    run_start = time.time()
    now = datetime.now()
    run_date_str = now.strftime("%A, %d %B %Y")
    run_time_str = now.strftime("%H:%M:%S")

    # Banner
    logger.write("")
    logger.write("  ══════════════════════════════════════════════════════════════")
    logger.write("  NABDH — UAE AI INTELLIGENCE PIPELINE")
    logger.write(f"  Starting run: {run_date_str}  {run_time_str}")
    logger.write("  ══════════════════════════════════════════════════════════════")

    # Pre-flight
    all_ok = run_preflight(logger)

    if not all_ok:
        logger.write("")
        logger.write("  ✗ Pre-flight checks FAILED. Fix the issues above and try again.")
        logger.write("")
        logger.close()
        sys.exit(1)

    if args.dry_run:
        logger.write("")
        logger.write("  [DRY RUN] All checks passed. Pipeline NOT started (--dry-run mode).")
        logger.write("")
        logger.close()
        sys.exit(0)

    # Countdown
    logger.write("")
    logger.write("  All checks passed. Starting pipeline in 3 seconds...")
    for i in range(3, 0, -1):
        logger.write_inline(f"  Starting in {i}...")
        time.sleep(1)
    print()

    # Input validation for non-step-1 starts
    if args.start_from and args.start_from > 1:
        first_step = steps_to_run[0]
        if not validate_step_input(first_step, logger):
            logger.close()
            sys.exit(1)

    # Run steps
    completed_nums = []
    results = []
    arabic_was_empty = is_arabic_folder_empty()

    for step in steps_to_run:
        num = step["num"]

        # Special case: skip Step 2 if no Arabic articles
        if num == 2 and arabic_was_empty:
            logger.write("")
            logger.write("  ──────────────────────────────────────────────────────────────")
            logger.write(f"  STEP 2 of 7 — ARABIC TRANSLATION  [SKIPPED]")
            logger.write("  Reason: news_output/arabic/ contains no JSON files.")
            logger.write("          This is normal if no Arabic articles were found today.")
            logger.write("  ──────────────────────────────────────────────────────────────")
            completed_nums.append(num)
            results.append((num, 0.0, "skipped — no Arabic articles"))
            continue

        success, duration, stat = run_step(step, logger, arabic_was_empty=arabic_was_empty)

        if not success:
            print_failure_report(step, completed_nums, logger, run_ts)
            logger.close()
            sys.exit(1)

        completed_nums.append(num)
        results.append((num, duration, stat))

        # After step 1, re-check if arabic folder is empty
        if num == 1:
            arabic_was_empty = is_arabic_folder_empty()

    total_duration = time.time() - run_start
    print_success_report(results, total_duration, run_date_str, logger, log_name)
    logger.close()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [INTERRUPTED] Pipeline stopped by user (Ctrl+C).")
        sys.exit(130)
    except Exception:
        tb = traceback.format_exc()
        print("\n\n  [FATAL] Unexpected error in orchestrator:")
        print(tb)
        # Try to write to a fallback log
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            crash_log = LOG_DIR / f"crash_{ts}.log"
            crash_log.write_text(f"[FATAL CRASH]\n{tb}", encoding="utf-8")
            print(f"  Crash log written to: {crash_log}")
        except Exception:
            pass
        sys.exit(1)
