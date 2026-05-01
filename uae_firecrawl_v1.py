#!/usr/bin/env python3
"""
uae_firecrawl_v1.py — Step 6
Fetch full article content for the 155 newsletter candidates via Firecrawl.

Root cause of previous failures:
  Google News RSS URLs use a JavaScript redirect that fires ~1-2s after page load.
  Firecrawl's default waitFor=0ms means it reads the page before the redirect fires,
  so metadata.url stays on news.google.com and markdown is empty.
  Fix: use waitFor=2000 on first attempt, waitFor=5000 on retry.

Input:  news_output/scored/newsletter_candidates.json
Output: news_output/firecrawled/  (3 files)
  firecrawled_articles.json  — candidates + full_content field
  firecrawl_summary.json     — run stats
  firecrawl_log.txt          — human-readable report
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import sys
import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ──────────────────────────────── CONFIG ──────────────────────────────────────
NEWS_DATE   = os.environ.get("NEWS_DATE", datetime.now().strftime("%Y-%m-%d"))
INPUT_FILE  = Path(f"news_output/{NEWS_DATE}/scored/newsletter_candidates.json")
OUTPUT_DIR  = Path(f"news_output/{NEWS_DATE}/firecrawled")

FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
FIRECRAWL_URL     = "https://api.firecrawl.dev/v1/scrape"

WORKERS          = 5     # concurrent Firecrawl requests
CHECKPOINT_EVERY = 10    # save progress every N articles
MIN_CONTENT_CHARS = 200  # below this → treat as failed

# waitFor strategy: try 2s first, retry with 5s if redirect didn't fire in time
WAIT_MS_ATTEMPTS = [2000, 5000]
REQUEST_TIMEOUT  = 60   # seconds — must exceed waitFor + network overhead

_CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────── FIRECRAWL CALL ─────────────────────────────────
def _firecrawl_scrape(url: str, wait_ms: int) -> dict | None:
    """Single Firecrawl attempt with a specific waitFor value."""
    try:
        resp = requests.post(
            FIRECRAWL_URL,
            headers={
                "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
                "waitFor": wait_ms,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def _extract_result(raw: dict) -> tuple[str | None, str, int]:
    """
    From a Firecrawl response extract:
      (resolved_url, markdown, chars)
    resolved_url comes from metadata.url — the URL Firecrawl actually landed on.
    """
    data     = raw.get("data") or {}
    metadata = data.get("metadata") or {}
    markdown = data.get("markdown") or ""
    resolved = metadata.get("url") or metadata.get("sourceURL")
    # Discard if redirect didn't fire (still on google.com)
    if resolved and "google.com" in resolved:
        resolved = None
    return resolved, markdown, len(markdown)


# ──────────────────────────── PER-ARTICLE FETCH ──────────────────────────────
def fetch_article(article: dict) -> dict:
    """
    Scrape one article.
    Strategy:
      1. Firecrawl with waitFor=2000 (covers most Google News JS redirects)
      2. Retry  with waitFor=5000 (slower sites / heavy JS)
      3. Fallback to RSS summary
    resolved_url is extracted from Firecrawl's metadata.url after the redirect lands.
    """
    cs        = article.get("content_source", {})
    rss_url   = cs.get("rss_url") or article.get("url", "")
    paywalled = cs.get("is_paywalled", False)
    fallback  = article.get("summary", "")

    firecrawl_block = {
        "attempted":    True,
        "resolved_url": None,
        "wait_ms_used": None,
        "status":       None,
        "content_chars": 0,
        "full_content": None,
        "scraped_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if paywalled:
        firecrawl_block.update({"attempted": False, "status": "SKIPPED_PAYWALL",
                                 "full_content": fallback, "content_chars": len(fallback)})
        return {**article, "firecrawl": firecrawl_block}

    if not rss_url:
        firecrawl_block.update({"attempted": False, "status": "NO_URL",
                                 "full_content": fallback, "content_chars": len(fallback)})
        return {**article, "firecrawl": firecrawl_block}

    # Try each waitFor value in sequence
    for wait_ms in WAIT_MS_ATTEMPTS:
        raw = _firecrawl_scrape(rss_url, wait_ms)
        if not raw or not raw.get("success"):
            continue

        resolved, markdown, chars = _extract_result(raw)

        # Update resolved URL if we got a real one this attempt
        if resolved:
            firecrawl_block["resolved_url"] = resolved
        firecrawl_block["wait_ms_used"] = wait_ms

        if chars >= MIN_CONTENT_CHARS:
            firecrawl_block.update({
                "status":        "OK",
                "full_content":  markdown,
                "content_chars": chars,
            })
            return {**article, "firecrawl": firecrawl_block}

        # Redirect fired but content too thin — one more wait won't help for this site

    # All attempts exhausted
    firecrawl_block.update({
        "status":        "FALLBACK_RSS",
        "full_content":  fallback,
        "content_chars": len(fallback),
    })
    return {**article, "firecrawl": firecrawl_block}


# ──────────────────────────── CHECKPOINT ──────────────────────────────────────
def save_checkpoint(done: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"results": done, "count": len(done)}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(_CHECKPOINT_PATH)


def load_checkpoint() -> dict:
    if not _CHECKPOINT_PATH.exists():
        return {}
    try:
        data  = json.loads(_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        count = data.get("count", 0)
        print(f"  [RESUME] Checkpoint found: {count} articles already fetched")
        return data.get("results", {})
    except Exception as e:
        print(f"  [WARN] Checkpoint load failed ({e}) — starting fresh")
        return {}


# ──────────────────────────────── MAIN ───────────────────────────────────────
def main():
    run_start = time.time()

    print()
    print("═" * 63)
    print("  UAE AI NEWSLETTER — FIRECRAWL CONTENT FETCH")
    print(f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 63)
    print()

    if not FIRECRAWL_API_KEY:
        print("  [ERROR] FIRECRAWL_API_KEY not set in .env")
        raise SystemExit(1)
    print("  [KEY] FIRECRAWL_API_KEY ✓")
    print(f"  [STRATEGY] waitFor attempts: {WAIT_MS_ATTEMPTS}ms — covers Google News JS redirect")

    if not INPUT_FILE.exists():
        print(f"  [ERROR] Input not found: {INPUT_FILE}")
        raise SystemExit(1)

    candidates = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    print(f"  [INPUT] {len(candidates)} newsletter candidates")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prior     = load_checkpoint()
    prior_ids = set(prior.keys())

    todo = [a for a in candidates if a.get("article_id") not in prior_ids]
    print(f"  [QUEUE] {len(todo)} to fetch  |  {len(prior_ids)} already done")
    print(f"  [WORKERS] {WORKERS} parallel")
    print()

    results      = dict(prior)
    results_lock = threading.Lock()
    counter      = {"done": 0}
    counter_lock = threading.Lock()
    print_lock   = threading.Lock()

    def process(article: dict):
        enriched = fetch_article(article)
        aid    = article.get("article_id", "")
        fc     = enriched["firecrawl"]
        status = fc["status"]
        chars  = fc["content_chars"]
        wait   = fc.get("wait_ms_used") or "-"
        resolved_short = (fc.get("resolved_url") or "")[:55]

        with results_lock:
            results[aid] = enriched

        with counter_lock:
            counter["done"] += 1
            done = counter["done"]

        with print_lock:
            title = (article.get("title") or "")[:45]
            print(f"  [{done:>3}/{len(candidates)}] {status:<18} {chars:>6}c  w={wait}  {title}")
            if resolved_short:
                print(f"  {'':>8} → {resolved_short}")

        if done % CHECKPOINT_EVERY == 0:
            with results_lock:
                snap = dict(results)
            save_checkpoint(snap)
            with print_lock:
                print(f"  [CHECKPOINT] {done}/{len(candidates)} saved")

    if todo:
        print("  Fetching content...")
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(process, a): a for a in todo}
            for f in as_completed(futures):
                exc = f.exception()
                if exc:
                    aid = futures[f].get("article_id", "?")
                    with print_lock:
                        print(f"  [ERROR] {aid[:8]}: {exc!r}")

    save_checkpoint(results)
    print()

    # ── Build output in original rank order ───────────────────────────────────
    id_to = {r["article_id"]: r for r in results.values()}
    ordered = [id_to[a["article_id"]] for a in candidates if a["article_id"] in id_to]

    # ── Stats ─────────────────────────────────────────────────────────────────
    status_counts = {}
    total_chars   = 0
    for r in ordered:
        fc = r.get("firecrawl", {})
        s  = fc.get("status", "UNKNOWN")
        status_counts[s] = status_counts.get(s, 0) + 1
        total_chars += fc.get("content_chars", 0)

    ok_count   = status_counts.get("OK", 0)
    avg_chars  = round(total_chars / len(ordered)) if ordered else 0
    total_time = time.time() - run_start
    run_at     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    subcat_stats = {}
    for r in ordered:
        sub  = r.get("scoring", {}).get("subcategory", "UNKNOWN")
        fc   = r.get("firecrawl", {})
        s    = fc.get("status", "UNKNOWN")
        chars = fc.get("content_chars", 0)
        if sub not in subcat_stats:
            subcat_stats[sub] = {"total": 0, "ok": 0, "total_chars": 0}
        subcat_stats[sub]["total"] += 1
        subcat_stats[sub]["total_chars"] += chars
        if s == "OK":
            subcat_stats[sub]["ok"] += 1

    # ── Write output ──────────────────────────────────────────────────────────
    (OUTPUT_DIR / "firecrawled_articles.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "run_at":             run_at,
        "total_candidates":   len(candidates),
        "fetched":            len(ordered),
        "status_breakdown":   status_counts,
        "ok_full_content":    ok_count,
        "fallback_rss":       status_counts.get("FALLBACK_RSS", 0),
        "skipped_paywall":    status_counts.get("SKIPPED_PAYWALL", 0),
        "avg_content_chars":  avg_chars,
        "total_time_seconds": round(total_time, 1),
        "workers":            WORKERS,
        "wait_ms_strategy":   WAIT_MS_ATTEMPTS,
        "subcategory_breakdown": subcat_stats,
    }
    (OUTPUT_DIR / "firecrawl_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    subcat_lines = [
        f"    {sub:<20}: {v['ok']}/{v['total']} OK  avg {round(v['total_chars']/v['total']) if v['total'] else 0}c"
        for sub, v in sorted(subcat_stats.items())
    ]
    log = [
        "═" * 63,
        "  UAE AI NEWSLETTER — FIRECRAWL COMPLETE",
        f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "═" * 63, "",
        f"  Candidates        : {len(candidates)}",
        f"  Full content (OK) : {ok_count}",
        f"  Fallback RSS      : {status_counts.get('FALLBACK_RSS', 0)}",
        f"  Skipped paywall   : {status_counts.get('SKIPPED_PAYWALL', 0)}",
        f"  Avg content       : {avg_chars:,} chars",
        f"  Time              : {total_time / 60:.1f} minutes", "",
        "  BY SUBCATEGORY", "  ──────────────",
        *subcat_lines, "",
        "  OUTPUT → news_output/firecrawled/",
        "═" * 63,
    ]
    (OUTPUT_DIR / "firecrawl_log.txt").write_text("\n".join(log), encoding="utf-8")

    print("═" * 63)
    print("  UAE AI NEWSLETTER — FIRECRAWL COMPLETE")
    print("═" * 63)
    print()
    print(f"  Candidates        : {len(candidates)}")
    print(f"  Full content (OK) : {ok_count}")
    print(f"  Fallback RSS      : {status_counts.get('FALLBACK_RSS', 0)}")
    print(f"  Skipped paywall   : {status_counts.get('SKIPPED_PAYWALL', 0)}")
    print(f"  Avg content       : {avg_chars:,} chars")
    print(f"  Time              : ~{total_time / 60:.1f} minutes")
    print()
    print("  BY SUBCATEGORY")
    print("  ──────────────")
    for line in subcat_lines:
        print(line)
    print()
    print(f"  OUTPUT → {OUTPUT_DIR}/")
    print("═" * 63)
    print()


if __name__ == "__main__":
    main()
