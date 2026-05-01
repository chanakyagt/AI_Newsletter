"""
paywallfetcher.py — URL Resolution + Paywall Detection
UAE AI Newsletter Pipeline — Step 2

Resolves Google News RSS URLs to real publisher URLs via Chrome CDP,
then checks each resolved URL for paywall status.
"""

# ═══════════════════════════════════════════════════════════════
# BEFORE RUNNING THIS SCRIPT:
#
# Step 1 — Close ALL existing Chrome windows completely
#
# Step 2 — Open Command Prompt (cmd) and run this command:
#
# "C:\Program Files\Google\Chrome\Application\chrome.exe" ^
#   --remote-debugging-port=9222 ^
#   --user-data-dir="C:\ChromeCDP" ^
#   --no-first-run ^
#   --no-default-browser-check
#
# Step 3 — Wait for Chrome to open (5 seconds)
#
# Step 4 — Log into your Google account in that Chrome window
#           (this gives it real session cookies for Google News)
#
# Step 5 — Run this script in a separate terminal window
#
# Install dependencies first:
#   pip install pychrome requests
# ═══════════════════════════════════════════════════════════════

import json
import os
import queue
import random
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pychrome
import requests

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

NEWS_DATE = os.environ.get("NEWS_DATE", datetime.now().strftime("%Y-%m-%d"))
INPUT_DIRS = [
    os.path.join("news_output", NEWS_DATE, "english"),
    os.path.join("news_output", NEWS_DATE, "arabic_translated"),
]
OUTPUT_DIR      = "url_resolution"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

CDP_HOST          = "http://127.0.0.1:9222"
NUM_WORKERS       = 6
URL_TIMEOUT       = 15     # seconds to wait for Google News redirect
NAV_POLL_INTERVAL = 0.3    # seconds between URL checks while waiting
NAV_STABLE_COUNT  = 3      # consecutive matching checks before accepting URL
DELAY_MIN         = 0.5    # min random delay between navigations (anti-bot)
DELAY_MAX         = 1.5    # max random delay between navigations (anti-bot)
FETCH_CHARS       = 5000   # characters to read for paywall content scan
MIN_CONTENT_LEN   = 500    # below this → treat as paywalled

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PAYWALL DATA
# ═════════════════════════════════════════════════════════════════════════════

PAYWALLED_DOMAINS = {
    "thenational.ae",
    "arabianbusiness.com",
    "ft.com",
    "wsj.com",
    "bloomberg.com",
    "economist.com",
    "hbr.org",
    "reuters.com",
    "businessinsider.com",
    "forbes.com",
    "telegraph.co.uk",
    "nytimes.com",
    "washingtonpost.com",
    "theathletic.com",
    "thetimes.co.uk",
}

PAYWALL_SIGNALS = [
    "subscribe to continue",
    "subscription required",
    "sign in to read",
    "members only",
    "create account to continue",
    "premium content",
    'class="paywall"',
    'class="subscription-wall"',
    'id="piano-inline"',
    'class="tp-modal"',
    'id="reg-wall"',
    "data-paywall",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOGGING UTILITY
# ═════════════════════════════════════════════════════════════════════════════

def safe_print(print_lock, message):
    with print_lock:
        print(message)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TAB MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def create_tab(browser, worker_id, print_lock):
    """Create and warm up a Chrome tab for a worker. Retries 3 times."""
    for attempt in range(3):
        try:
            tab = browser.new_tab()
            tab.start()
            tab.Page.enable()
            tab.Runtime.enable()
            safe_print(print_lock, f"[{worker_id}] Tab ready.")
            return tab
        except Exception as exc:
            safe_print(print_lock, f"[{worker_id}] Tab creation error (attempt {attempt + 1}): {exc}")
            time.sleep(1)
    raise RuntimeError(f"[{worker_id}] Could not create Chrome tab after 3 attempts.")


def close_tab(browser, tab):
    """Safely close a tab without raising."""
    try:
        tab.stop()
        browser.close_tab(tab)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — URL RESOLUTION
# ═════════════════════════════════════════════════════════════════════════════

def resolve_url(tab, google_url):
    """
    Navigate to a Google News RSS URL and poll until the JS interstitial
    redirects to the real publisher URL.
    Returns the real URL string, or None on timeout.
    Raises on tab crash so the caller can replace the tab.
    """
    tab.Page.navigate(url=google_url)

    start       = time.time()
    last_url    = google_url
    stable_cnt  = 0

    while time.time() - start < URL_TIMEOUT:
        result      = tab.Runtime.evaluate(expression="window.location.href")
        current_url = result["result"]["value"]

        if "news.google.com" not in current_url:
            if current_url == last_url:
                stable_cnt += 1
                if stable_cnt >= NAV_STABLE_COUNT:
                    return current_url
            else:
                stable_cnt = 0
            last_url = current_url

        time.sleep(NAV_POLL_INTERVAL)

    # Last-chance read after timeout
    try:
        result    = tab.Runtime.evaluate(expression="window.location.href")
        final_url = result["result"]["value"]
        if "news.google.com" not in final_url:
            return final_url
    except Exception:
        pass

    return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PAYWALL DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_domain(url):
    try:
        domain = urlparse(url).netloc.lower()
        return domain[4:] if domain.startswith("www.") else domain
    except Exception:
        return ""


def check_paywall(url, user_agent):
    """
    Returns dict: {domain, paywall_status, paywall_method, paywall_signals}
    Order of checks:
      1. Hardcoded PAYWALLED_DOMAINS  → instant, no HTTP call
      2. HTTP 401/403 status
      3. Content length < MIN_CONTENT_LEN
      4. PAYWALL_SIGNALS scan of first FETCH_CHARS characters
    """
    domain = extract_domain(url)

    if domain in PAYWALLED_DOMAINS:
        return {
            "domain": domain,
            "paywall_status": "paywalled",
            "paywall_method": "domain_list",
            "paywall_signals": [],
        }

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=10)

        if resp.status_code in (401, 403):
            return {
                "domain": domain,
                "paywall_status": "paywalled",
                "paywall_method": "http_status",
                "paywall_signals": [f"HTTP {resp.status_code}"],
            }

        content = ""
        for chunk in resp.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                content += chunk
                if len(content) >= FETCH_CHARS:
                    break
        resp.close()

        if len(content.strip()) < MIN_CONTENT_LEN:
            return {
                "domain": domain,
                "paywall_status": "paywalled",
                "paywall_method": "short_content",
                "paywall_signals": [f"Content length: {len(content.strip())} chars"],
            }

        content_lower = content.lower()
        triggered = [s for s in PAYWALL_SIGNALS if s.lower() in content_lower]
        if triggered:
            return {
                "domain": domain,
                "paywall_status": "paywalled",
                "paywall_method": "content_scan",
                "paywall_signals": triggered,
            }

        return {
            "domain": domain,
            "paywall_status": "free",
            "paywall_method": "content_scan",
            "paywall_signals": [],
        }

    except requests.RequestException as exc:
        return {
            "domain": domain,
            "paywall_status": "fetch_error",
            "paywall_method": "fetch_error",
            "paywall_signals": [str(exc)],
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════════

def load_checkpoint():
    """Return (completed_results_list, completed_ids_set). Empty on missing/corrupt file."""
    p = Path(CHECKPOINT_FILE)
    if not p.exists():
        return [], set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data, {r["article_id"] for r in data}
    except Exception:
        return [], set()


def save_checkpoint(snapshot, checkpoint_lock):
    """Write checkpoint atomically: temp file → rename."""
    p    = Path(CHECKPOINT_FILE)
    tmp  = p.with_suffix(".tmp")
    with checkpoint_lock:
        try:
            tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            shutil.move(str(tmp), str(p))
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PROGRESS DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def print_article_line(print_lock, worker_id, count, total, article, result):
    source = (article.get("source") or "unknown source")[:20].lower()
    status = result["paywall_status"]
    real   = result.get("real_url") or ""
    method = result.get("paywall_method") or ""

    if status == "resolution_failed":
        url_col    = "RESOLUTION FAILED"
        status_col = "(timeout)"
    elif status == "fetch_error":
        url_col    = real[:58] or result.get("domain", "")
        status_col = "FETCH ERROR"
    elif status == "paywalled":
        url_col    = real[:58] or result.get("domain", "")
        status_col = f"PAYWALLED ({method})"
    else:
        url_col    = real[:58]
        status_col = "FREE"

    with print_lock:
        print(f"[{worker_id}] [{count:>3}/{total}] {source:<22} → {url_col:<60} {status_col}")


def print_summary_line(print_lock, snapshot, count, total, run_start):
    free      = sum(1 for r in snapshot if r["paywall_status"] == "free")
    paywalled = sum(1 for r in snapshot if r["paywall_status"] == "paywalled")
    failed    = sum(1 for r in snapshot if r["paywall_status"] in ("resolution_failed", "fetch_error"))
    elapsed   = time.time() - run_start
    remaining = total - count
    rate      = count / elapsed if elapsed > 0 else 0
    eta       = f"~{int(remaining / rate)}s" if rate > 0 else "~?s"
    with print_lock:
        print(
            f"── Progress: {count}/{total} complete │ "
            f"Free: {free} │ Paywalled: {paywalled} │ Failed: {failed} │ ETA: {eta} ──"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — WORKER
# ═════════════════════════════════════════════════════════════════════════════

def worker_func(worker_id, browser, work_queue, shared, locks, total_count, run_start):
    """
    Persistent worker thread. Owns one Chrome tab for its entire lifetime.
    Pulls articles from work_queue until empty. Recovers from tab crashes
    by opening a replacement tab and continuing with the next article.
    """
    print_lock      = locks["print"]
    results_lock    = locks["results"]
    counter_lock    = locks["counter"]
    checkpoint_lock = locks["checkpoint"]

    ua_index = hash(worker_id) % len(USER_AGENTS)

    try:
        tab = create_tab(browser, worker_id, print_lock)
    except RuntimeError as exc:
        safe_print(print_lock, f"[{worker_id}] Fatal — cannot create tab: {exc}. Exiting.")
        return

    while True:
        try:
            article = work_queue.get_nowait()
        except queue.Empty:
            break

        google_url = article.get("url", "")
        start_time = time.time()
        real_url   = None

        # ── Resolve URL via CDP ─────────────────────────────────────────────
        try:
            real_url = resolve_url(tab, google_url)
        except Exception as exc:
            safe_print(print_lock, f"[{worker_id}] Tab crashed during resolve: {exc}. Replacing tab.")
            close_tab(browser, tab)
            try:
                tab = create_tab(browser, worker_id, print_lock)
            except RuntimeError:
                safe_print(print_lock, f"[{worker_id}] Cannot replace tab — skipping article.")
                work_queue.task_done()
                continue

        elapsed = round(time.time() - start_time, 2)

        # ── Build result record ─────────────────────────────────────────────
        if not real_url:
            result = {
                **article,
                "google_url":              google_url,
                "real_url":                None,
                "domain":                  "",
                "paywall_status":          "resolution_failed",
                "paywall_method":          None,
                "paywall_signals":         [],
                "worker_id":               worker_id,
                "processed_at":            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "resolution_time_seconds": elapsed,
            }
        else:
            ua       = USER_AGENTS[ua_index % len(USER_AGENTS)]
            ua_index += 1
            paywall  = check_paywall(real_url, ua)
            result   = {
                **article,
                "google_url":              google_url,
                "real_url":                real_url,
                "domain":                  paywall["domain"],
                "paywall_status":          paywall["paywall_status"],
                "paywall_method":          paywall["paywall_method"],
                "paywall_signals":         paywall["paywall_signals"],
                "worker_id":               worker_id,
                "processed_at":            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "resolution_time_seconds": elapsed,
            }

        # ── Save (thread-safe) ──────────────────────────────────────────────
        with results_lock:
            shared["results"].append(result)
            snapshot = list(shared["results"])   # snapshot for checkpoint + stats

        save_checkpoint(snapshot, checkpoint_lock)

        with counter_lock:
            shared["counter"] += 1
            count = shared["counter"]

        print_article_line(print_lock, worker_id, count, total_count, article, result)

        if count % 10 == 0:
            print_summary_line(print_lock, snapshot, count, total_count, run_start)

        work_queue.task_done()

        # Random delay — prevents all 6 workers hitting Google in lockstep
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    close_tab(browser, tab)
    safe_print(print_lock, f"[{worker_id}] Queue exhausted. Tab closed.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — OUTPUT WRITERS
# ═════════════════════════════════════════════════════════════════════════════

def write_outputs(results, total_count, elapsed, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    free_results = [r for r in results if r["paywall_status"] == "free"]

    # ── resolved_urls.json ───────────────────────────────────────────────────
    (out / "resolved_urls.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── free_urls.json ───────────────────────────────────────────────────────
    (out / "free_urls.json").write_text(
        json.dumps(free_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Stats for report ─────────────────────────────────────────────────────
    free      = [r for r in results if r["paywall_status"] == "free"]
    paywalled = [r for r in results if r["paywall_status"] == "paywalled"]
    failed    = [r for r in results if r["paywall_status"] == "resolution_failed"]
    errors    = [r for r in results if r["paywall_status"] == "fetch_error"]
    resolved  = [r for r in results if r.get("real_url")]

    # Domain breakdowns
    paywalled_domain_stats = {}
    for r in paywalled:
        d = r.get("domain", "unknown")
        if d not in paywalled_domain_stats:
            paywalled_domain_stats[d] = {"count": 0, "method": r.get("paywall_method", "")}
        paywalled_domain_stats[d]["count"] += 1

    free_domain_stats = {}
    for r in free:
        d = r.get("domain", "unknown")
        free_domain_stats[d] = free_domain_stats.get(d, 0) + 1

    # Newly discovered paywalled domains (not in hardcoded list)
    new_paywalled = {
        d for d, info in paywalled_domain_stats.items()
        if d and d not in PAYWALLED_DOMAINS
        and info["method"] in ("content_scan", "http_status")
    }

    n       = max(total_count, 1)
    mins, s = divmod(int(elapsed), 60)
    run_dt  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def pct(x):
        return f"{x / n * 100:.1f}%"

    # ── paywall_report.txt ───────────────────────────────────────────────────
    lines = [
        "═" * 47,
        "  URL RESOLUTION + PAYWALL DETECTION REPORT",
        "═" * 47,
        f"  Run date        : {run_dt}",
        f"  Total articles  : {total_count}",
        f"  Workers used    : {NUM_WORKERS}",
        f"  Time taken      : {mins}m {s}s",
        "",
        "  RESULTS SUMMARY",
        "  " + "─" * 15,
        f"  Successfully resolved  : {len(resolved)}  ({pct(len(resolved))})",
        f"  Free and accessible    : {len(free)}  ({pct(len(free))})",
        f"  Paywalled              : {len(paywalled)}  ({pct(len(paywalled))})",
        f"  Resolution failed      : {len(failed)}  ({pct(len(failed))})",
        f"  Fetch errors           : {len(errors)}  ({pct(len(errors))})",
        "",
        "  PAYWALLED DOMAINS",
        "  " + "─" * 17,
    ]
    for d, info in sorted(paywalled_domain_stats.items(), key=lambda x: -x[1]["count"]):
        lines.append(f"  {d:<30} {info['count']} articles  [{info['method']}]")

    lines += ["", "  FREE DOMAINS", "  " + "─" * 12]
    for d, count in sorted(free_domain_stats.items(), key=lambda x: -x[1]):
        lines.append(f"  {d:<30} {count} articles")

    lines += ["", "  FAILED RESOLUTIONS", "  " + "─" * 19]
    if failed:
        lines.append(f"  {len(failed)} articles could not be resolved (timeout after {URL_TIMEOUT}s)")
        lines.append("  These remain on news.google.com and cannot be fetched.")
    else:
        lines.append("  None — all articles resolved successfully.")
    lines.append("═" * 47)

    (out / "paywall_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── new_paywalled_domains.txt ────────────────────────────────────────────
    domain_lines = [
        "# New paywalled domains detected this run",
        "# Consider adding these to PAYWALLED_DOMAINS in paywallfetcher.py",
    ] + sorted(new_paywalled)
    (out / "new_paywalled_domains.txt").write_text("\n".join(domain_lines) + "\n", encoding="utf-8")

    print(f"  [SAVED] resolved_urls.json         ({len(results)} articles)")
    print(f"  [SAVED] free_urls.json             ({len(free)} articles)")
    print(f"  [SAVED] paywall_report.txt")
    if new_paywalled:
        print(f"  [SAVED] new_paywalled_domains.txt  ({len(new_paywalled)} new domains)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — INPUT LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_articles():
    """
    Read all JSON files from INPUT_DIRS and merge into one deduplicated list.
    Deduplication is by article_id — first occurrence wins.
    Prints a per-directory summary so the user can see what was loaded.
    """
    seen_ids = set()
    all_articles = []

    for directory in INPUT_DIRS:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"  [WARN] Input directory not found, skipping: {dir_path}")
            continue

        files = sorted(dir_path.glob("*.json"))
        dir_count = 0
        for f in files:
            try:
                articles = json.loads(f.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"  [WARN] Could not read {f.name}: {exc}")
                continue
            for a in articles:
                aid = a.get("article_id")
                if aid and aid not in seen_ids:
                    seen_ids.add(aid)
                    all_articles.append(a)
                    dir_count += 1

        print(f"  [LOADED] {dir_path}  →  {dir_count} articles from {len(files)} file(s)")

    return all_articles


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 — MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Load input from all directories ──────────────────────────────────────
    print("  Loading articles...")
    articles = load_articles()
    if not articles:
        print("[ERROR] No articles found in input directories:")
        for d in INPUT_DIRS:
            print(f"  {d}")
        raise SystemExit(1)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    completed_results, completed_ids = load_checkpoint()
    pending = [a for a in articles if a.get("article_id") not in completed_ids]

    if completed_ids:
        print(
            f"[RESUME] Found checkpoint with {len(completed_ids)} completed articles. "
            f"Resuming from article {len(completed_ids) + 1}."
        )
    else:
        print(f"[FRESH RUN] Processing {len(articles)} articles with {NUM_WORKERS} parallel workers.")

    total_count = len(articles)

    if not pending:
        print("\n  All articles already processed. Delete checkpoint.json to re-run.\n")
        write_outputs(completed_results, total_count, 0, OUTPUT_DIR)
        raise SystemExit(0)

    # ── Connect to Chrome CDP ─────────────────────────────────────────────────
    try:
        browser = pychrome.Browser(url=CDP_HOST)
        browser.list_tab()   # throws if Chrome isn't reachable
    except Exception as exc:
        print(f"\n[ERROR] Cannot connect to Chrome on {CDP_HOST}")
        print("  Make sure you started Chrome with --remote-debugging-port=9222")
        print(f"  Details: {exc}\n")
        raise SystemExit(1)

    print(f"  Chrome CDP connected at {CDP_HOST}\n")

    # ── Shared state + locks ─────────────────────────────────────────────────
    shared = {
        "results": list(completed_results),  # pre-seed with checkpoint data
        "counter": len(completed_ids),        # already-done count
    }
    locks = {
        "print":      threading.Lock(),
        "results":    threading.Lock(),
        "counter":    threading.Lock(),
        "checkpoint": threading.Lock(),
    }

    # ── Work queue ───────────────────────────────────────────────────────────
    work_queue = queue.Queue()
    for article in pending:
        work_queue.put(article)

    run_start = time.time()

    input_dirs_str = "\n  ".join(f"→ {d}" for d in INPUT_DIRS)
    print(f"""{'='*55}
  URL RESOLUTION + PAYWALL DETECTION
{'='*55}
  Input dirs  : {input_dirs_str}
  Total       : {total_count} articles
  To process  : {len(pending)}  (skipping {len(completed_ids)} from checkpoint)
  Workers     : {NUM_WORKERS} parallel Chrome tabs
  Output dir  : {Path(OUTPUT_DIR).resolve()}
{'='*55}
""")

    # ── Launch worker threads ────────────────────────────────────────────────
    # Stagger starts slightly so tab creation doesn't collide
    workers = []
    for i in range(min(NUM_WORKERS, len(pending))):
        t = threading.Thread(
            target=worker_func,
            args=(f"worker-{i + 1}", browser, work_queue, shared, locks, total_count, run_start),
            daemon=True,
            name=f"worker-{i + 1}",
        )
        workers.append(t)
        t.start()
        time.sleep(0.4)

    for t in workers:
        t.join()

    # ── Final output ─────────────────────────────────────────────────────────
    elapsed = time.time() - run_start
    with locks["results"]:
        final_results = list(shared["results"])

    print(f"\n{'='*55}")
    print("  Writing output files...")
    write_outputs(final_results, total_count, elapsed, OUTPUT_DIR)

    mins, secs  = divmod(int(elapsed), 60)
    free_count  = sum(1 for r in final_results if r["paywall_status"] == "free")

    print(f"""
{'='*55}
  COMPLETE
{'='*55}
  Total articles     : {total_count}
  Free articles      : {free_count}
  Time taken         : {mins}m {secs}s
  Output directory   : {Path(OUTPUT_DIR).resolve()}/
{'='*55}
""")


if __name__ == "__main__":
    run()
