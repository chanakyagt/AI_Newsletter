#!/usr/bin/env python3
"""
uae_ai_keypoints_v2.py — Step 7A
Extract structured intelligence keypoints from firecrawled articles via DeepSeek.
3 API keys × 5 workers = 15 parallel workers. Checkpoint-resumable.

Input:  news_output/firecrawled/firecrawled_articles.json
Output: news_output/keypoints/
  keypoints.json          — sorted array (urgency → score)
  keypoints_summary.json  — run stats + urgency distribution + cost
  keypoints_log.txt       — human-readable report
"""

import json
import os
import queue
import re
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ──────────────────────────────── CONFIG ──────────────────────────────────────
NEWS_DATE   = os.environ.get("NEWS_DATE", datetime.now().strftime("%Y-%m-%d"))
INPUT_FILE  = Path(f"news_output/{NEWS_DATE}/firecrawled/firecrawled_articles.json")
OUTPUT_DIR  = Path(f"news_output/{NEWS_DATE}/keypoints")
OUTPUT_FILE = OUTPUT_DIR / "keypoints.json"

DEEPSEEK_API_KEY_1 = os.environ.get("Deepseek_API_Key_1")
DEEPSEEK_API_KEY_2 = os.environ.get("Deepseek_API_Key_2")
DEEPSEEK_API_KEY_3 = os.environ.get("Deepseek_API_Key_3")
MODEL    = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

TODAY       = datetime.now().strftime("%Y-%m-%d")
TODAY_HUMAN = datetime.now().strftime("%d %B %Y")

WORKERS_PER_KEY  = 5
MAX_TOKENS       = 900
CHECKPOINT_EVERY = 20

URGENCY_ORDER    = {"IMMEDIATE": 0, "THIS WEEK": 1, "WATCH": 2, "THIS QUARTER": 3}
_CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"
# ──────────────────────────────────────────────────────────────────────────────

# Static system prompt — built once, reused across all 155 calls for cache hits
SYSTEM_PROMPT = f"""You are chief intelligence analyst for DHAKA'A (ذكاء), UAE's most authoritative AI newsletter. Readers are CXOs, CFOs, CTOs, CIOs and board members of major UAE organisations. Today is {TODAY}.

They have read everything. They need synthesis not summary. What the headline does not say. What it means for a specific UAE board decision in the next 90 days.

Be brutally specific. Generic observations, vague implications, and invented facts are strictly forbidden.

Return ONLY a valid JSON object. No preamble. No markdown. No explanation."""


# ──────────────────────────── AGE HELPER ─────────────────────────────────────
def calculate_age_days(pub_date_str: str) -> int:
    try:
        from datetime import datetime, timezone
        pub = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return max(0, (now - pub).days)
    except Exception:
        return 7  # unknown — assume mid-week


# ──────────────────────────── CONTENT & CITATION HELPERS ─────────────────────
def get_content(article: dict) -> str:
    fc     = article.get("firecrawl", {})
    status = fc.get("status", "")
    full   = fc.get("full_content", "") or ""
    if status == "OK" and len(full) > 200:
        return full[:4000]
    cs_summary = (article.get("content_source") or {}).get("summary", "")
    summary    = article.get("summary", "")
    return (cs_summary or summary or "")[:1000]


def get_citation_name(article: dict) -> str:
    # citation_source is preferred; falls back to root source field
    return (
        (article.get("citation_source") or {}).get("source_name")
        or article.get("source", "Unknown Source")
    )


def get_citation_url(article: dict) -> str:
    """
    Priority: firecrawl.resolved_url → citation_source.rss_url → root url.
    NEVER uses content_source. Paywalled sources are kept intentionally.
    """
    resolved = (article.get("firecrawl") or {}).get("resolved_url", "")
    if resolved and "google.com" not in resolved:
        return resolved
    cs_rss = (article.get("citation_source") or {}).get("rss_url", "")
    if cs_rss:
        return cs_rss
    return article.get("url", "#")


# ──────────────────────────── PROMPT BUILDER ─────────────────────────────────
def build_prompt(article: dict) -> str:
    title      = (article.get("title") or "")[:150]
    content    = get_content(article)
    cite_name  = get_citation_name(article)
    cite_url   = get_citation_url(article)
    pub_date   = (article.get("published_date") or "")[:10]
    aid        = article["article_id"]
    scoring    = article.get("scoring") or {}
    subcat     = scoring.get("subcategory", "")
    score      = scoring.get("final_score", 0)
    breakdown  = scoring.get("score_breakdown") or {}
    uae_sc     = breakdown.get("uae_relevance", 0)
    time_sc    = breakdown.get("timeliness", 0)
    is_paywall = (article.get("citation_source") or {}).get("is_paywalled", False)

    paywall_note = " [PAYWALLED — use citation regardless]" if is_paywall else ""
    age_days     = calculate_age_days(article.get("published_date") or "")

    return f"""Article date: {pub_date} | Today: {TODAY} | Age: {age_days} days old
Title: {title}
Source: {cite_name}{paywall_note}
Category: {subcat} | Score: {score} | UAE relevance: {uae_sc}/20 | Timeliness: {time_sc}/20
Content:
{content}

Return this exact JSON object:
{{
  "article_id": "{aid}",
  "headline_reframe": "WHO did WHAT with WHAT CONSEQUENCE. Specific nouns. No adjectives. Max 18 words.",
  "the_signal": "Non-obvious interpretation — what this reveals about a larger UAE AI structural shift the headline does not say. 2 sentences max, each under 120 chars.",
  "business_implication": "Specific decision a UAE board must make within 90 days. Not a category — the actual decision. E.g. not review AI strategy but decide whether to bid for federal AI services contract before Q3 deadline. 2 sentences.",
  "opportunity_or_threat": "OPPORTUNITY or THREAT — one sentence naming specific actor who wins or loses and exact mechanism. Under 150 chars.",
  "key_facts": [
    "Verbatim or near-verbatim fact: number, name, or date from article",
    "Second anchoring detail: deal size, policy name, timeline, or named official",
    "Third if present — omit entirely if not in article, do not invent"
  ],
  "power_quote": "One sentence that silences a boardroom. Specific claim. Named person if quoted. Under 180 chars.",
  "narrative_type": "Classify as exactly one: STANDALONE: Genuinely new event reported for first time. New deal, new announcement, new launch, new policy. Fresh news requiring no prior context. DEVELOPMENT: Update to an existing ongoing story. This is a new chapter of something already known. Signs: words like advances, moves to phase, update, progress, follows earlier announcement, building on prior. Reader needs prior context to understand why this matters. STRUCTURAL: Analysis, ranking, trend report, or pattern confirmation. Not a discrete new event. Research reports, index rankings, study finds, trend continues.",
  "narrative_reasoning": "One sentence explaining narrative type choice. Under 80 chars.",
  "urgency_label": "All articles in this pipeline are 0-7 days old from weekly RSS. Use ONLY these three labels — THIS QUARTER does not apply here: IMMEDIATE: Article is {age_days} days old (0-2 days) AND (STANDALONE OR DEVELOPMENT that changes a decision). Use for breaking news and decision-changing developments. Target: ~35% of articles. THIS WEEK: Article is 3-7 days old OR is a DEVELOPMENT that confirms expected progress OR is STANDALONE but slightly older. Default label when between IMMEDIATE and WATCH. Target: ~55% of articles. WATCH: STRUCTURAL articles with no near-term executive action. Analysis pieces, trend reports, rankings. Low urgency awareness items. Target: ~10% of articles. RULES: Never assign THIS QUARTER — that label is reserved for a separate pipeline step. If STANDALONE → minimum label is THIS WEEK regardless of age. If DEVELOPMENT and changes a decision → THIS WEEK minimum. If DEVELOPMENT and confirms expected progress → THIS WEEK (not WATCH, not THIS QUARTER). If STRUCTURAL and contains surprising finding → THIS WEEK. If STRUCTURAL and confirms known pattern → WATCH. Default when unsure: THIS WEEK. If you assign WATCH, narrative_type must be STRUCTURAL — never assign WATCH to STANDALONE or DEVELOPMENT.",
  "urgency_rationale": "One phrase explaining this urgency assignment. Under 60 chars.",
  "category_tag": "{subcat}",
  "source_citation": "{cite_name}",
  "citation_url": "{cite_url}",
  "published_date": "{pub_date}",
  "final_score": {score}
}}"""


# ──────────────────────────── CHECKPOINT ─────────────────────────────────────
def save_checkpoint(results: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"results": results, "count": len(results)}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(_CHECKPOINT_PATH)


def load_checkpoint() -> dict:
    if not _CHECKPOINT_PATH.exists():
        return {}
    try:
        data  = json.loads(_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        count = data.get("count", 0)
        print(f"  [RESUME] Checkpoint found: {count} articles already processed")
        return data.get("results", {})
    except Exception as e:
        print(f"  [WARN] Checkpoint load failed ({e}) — starting fresh")
        return {}


# ──────────────────────────── API CALL ───────────────────────────────────────
def call_deepseek(article: dict, client: OpenAI) -> dict | None:
    prompt = build_prompt(article)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            raw    = resp.choices[0].message.content.strip()
            # Strip accidental markdown fences
            raw    = re.sub(r"^```(?:json)?\s*", "", raw)
            raw    = re.sub(r"\s*```$", "", raw).strip()
            parsed = json.loads(raw)

            # Always enforce citation integrity — never trust LLM for URLs
            parsed["citation_url"]    = get_citation_url(article)
            parsed["source_citation"] = get_citation_name(article)

            return parsed

        except json.JSONDecodeError:
            time.sleep(2)
        except Exception as e:
            errs = str(e)
            if "429" in errs:
                raise RateLimitError(errs, response=None, body=None)
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                return None

    return None


# ──────────────────────────────── MAIN ───────────────────────────────────────
def main():
    run_start = time.time()
    run_at    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print()
    print("═" * 55)
    print("  DHAKA’A (ذكاء) — KEYPOINT EXTRACTION")
    print(f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 55)
    print()

    # ── Validate keys ─────────────────────────────────────────────────────────
    active_keys = []
    for i, key in enumerate([DEEPSEEK_API_KEY_1, DEEPSEEK_API_KEY_2, DEEPSEEK_API_KEY_3], 1):
        label = f"Deepseek_API_Key_{i}"
        if key:
            print(f"  [KEY] {label} ✓")
            active_keys.append((f"KEY-{i}", key))
        else:
            print(f"  [KEY] {label} ✗  (not set)")

    if not active_keys:
        print("  [ERROR] No DeepSeek API keys found.")
        raise SystemExit(1)

    num_workers = WORKERS_PER_KEY * len(active_keys)
    print(f"  [PARALLEL] {len(active_keys)} key(s) × {WORKERS_PER_KEY} workers = {num_workers} total")
    print()

    if not INPUT_FILE.exists():
        print(f"  [ERROR] Input not found: {INPUT_FILE}")
        raise SystemExit(1)

    articles = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    print(f"  [INPUT]  {len(articles)} articles loaded")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prior      = load_checkpoint()
    prior_ids  = set(prior.keys())
    todo       = [a for a in articles if a.get("article_id") not in prior_ids]
    print(f"  [QUEUE]  {len(todo)} to process  |  {len(prior_ids)} already done")
    print()

    # ── Shared state ──────────────────────────────────────────────────────────
    work_q       = queue.Queue()
    results_dict = dict(prior)
    failed_ids   = []
    results_lock = threading.Lock()
    failed_lock  = threading.Lock()
    counter      = {"n": 0}
    cnt_lock     = threading.Lock()
    print_lock   = threading.Lock()

    for a in todo:
        work_q.put(a)

    # One client per key — shared across that key's 5 workers (preserves cache)
    key_clients      = {label: OpenAI(api_key=key, base_url=BASE_URL)
                        for label, key in active_keys}
    key_pause_events = {label: threading.Event() for label, _ in active_keys}
    key_pause_locks  = {label: threading.Lock()  for label, _ in active_keys}
    for ev in key_pause_events.values():
        ev.set()   # set = not paused

    # ── Worker ────────────────────────────────────────────────────────────────
    def worker(worker_id: int, key_label: str, client: OpenAI,
               pause_event: threading.Event, pause_lock: threading.Lock):
        while True:
            try:
                article = work_q.get(timeout=2)
            except queue.Empty:
                break

            aid   = article.get("article_id", "")
            title = (article.get("title") or "")[:48]

            kp = None
            for attempt in range(3):
                pause_event.wait()   # block if this key is rate-paused

                try:
                    kp = call_deepseek(article, client)
                    break

                except RateLimitError:
                    if pause_lock.acquire(blocking=False):
                        pause_event.clear()
                        with print_lock:
                            print(f"\n  [{key_label}] Rate limited — pausing 60s")
                        time.sleep(60)
                        pause_event.set()
                        pause_lock.release()
                    else:
                        pause_event.wait()
                    # loop back — retry same article after pause

                except Exception as e:
                    wait = 3 * (attempt + 1)
                    with print_lock:
                        print(f"\n  [{key_label}] Error attempt {attempt+1}/3: {e!r} — retry in {wait}s")
                    time.sleep(wait)

            # Record result
            with cnt_lock:
                counter["n"] += 1
                n = counter["n"]
            total = len(articles)

            if kp:
                urgency = kp.get("urgency_label", "WATCH")
                score   = kp.get("final_score", 0)
                with results_lock:
                    results_dict[aid] = kp
                with print_lock:
                    print(f"  [{n:>3}/{total}] ✓  [{urgency:<12}]  {score:>3}pts  {title}")
            else:
                with failed_lock:
                    failed_ids.append(aid)
                with print_lock:
                    print(f"  [{n:>3}/{total}] ✗  FAILED  {title}")

            # Checkpoint
            if n % CHECKPOINT_EVERY == 0:
                with results_lock:
                    snap = dict(results_dict)
                save_checkpoint(snap)
                with print_lock:
                    print(f"  [CHECKPOINT] {n}/{total} processed — {len(snap)} keypoints saved")

            work_q.task_done()

    # ── Launch workers ────────────────────────────────────────────────────────
    if todo:
        print("  Extracting keypoints...")
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for key_label, _ in active_keys:
                    client   = key_clients[key_label]
                    pause_ev = key_pause_events[key_label]
                    pause_lk = key_pause_locks[key_label]
                    for wid in range(WORKERS_PER_KEY):
                        pool.submit(worker, wid, key_label, client, pause_ev, pause_lk)
        except KeyboardInterrupt:
            print("\n  [INTERRUPT] Ctrl+C — saving checkpoint before exit...")
            with results_lock:
                save_checkpoint(dict(results_dict))
            print(f"  [SAVED] {len(results_dict)} keypoints. Run again to resume.")
            raise SystemExit(0)
        print()

    # Final checkpoint
    save_checkpoint(results_dict)
    print(f"  [DONE] {len(results_dict)} keypoints extracted  |  {len(failed_ids)} failed")
    print()

    # ── Sort: urgency priority → score descending ─────────────────────────────
    keypoints = list(results_dict.values())
    keypoints.sort(key=lambda x: (
        URGENCY_ORDER.get(x.get("urgency_label", "WATCH"), 3),
        -x.get("final_score", 0),
    ))

    # ── Citation integrity check ───────────────────────────────────────────────
    missing_citation = [k for k in keypoints
                        if not k.get("citation_url") or not k.get("source_citation")]
    if missing_citation:
        print(f"  [WARN] {len(missing_citation)} keypoints missing citation fields — check manually")

    # ── Urgency + narrative distribution with sanity checks ───────────────────
    urgency_dist   = Counter(r.get("urgency_label")  for r in keypoints)
    narrative_dist = Counter(r.get("narrative_type") for r in keypoints)

    print("\n  URGENCY DISTRIBUTION")
    for label in ["IMMEDIATE", "THIS WEEK", "WATCH", "THIS QUARTER"]:
        count = urgency_dist.get(label, 0)
        pct   = count / len(keypoints) * 100 if keypoints else 0
        bar   = "█" * min(count, 40)
        warn  = "  ← WRONG: remove from prompt" if label == "THIS QUARTER" and count > 0 else ""
        print(f"    {label:<14}: {count:>3}  ({pct:.0f}%)  {bar}{warn}")

    print("\n  NARRATIVE TYPE DISTRIBUTION")
    for ntype in ["STANDALONE", "DEVELOPMENT", "STRUCTURAL"]:
        count = narrative_dist.get(ntype, 0)
        pct   = count / len(keypoints) * 100 if keypoints else 0
        print(f"    {ntype:<14}: {count:>3}  ({pct:.0f}%)")

    quarter_count   = urgency_dist.get("THIS QUARTER", 0)
    watch_count     = urgency_dist.get("WATCH", 0)
    immediate_count = urgency_dist.get("IMMEDIATE", 0)
    week_count      = urgency_dist.get("THIS WEEK", 0)

    if quarter_count > 0:
        print(f"\n  [ERROR] {quarter_count} articles got THIS QUARTER — this label must not appear in weekly keypoints")
        print(f"  [ERROR] Check prompt — THIS QUARTER rule must be removed entirely")

    if watch_count > len(keypoints) * 0.20:
        print(f"\n  [WARN] WATCH = {watch_count} ({watch_count/len(keypoints)*100:.0f}%) — too high")
        print(f"  [WARN] Only STRUCTURAL articles should be WATCH")

    if immediate_count < len(keypoints) * 0.15:
        print(f"\n  [WARN] IMMEDIATE = {immediate_count} — seems low for fresh weekly news")
        print(f"  [WARN] STANDALONE articles 0-2 days old should be IMMEDIATE")

    if week_count < len(keypoints) * 0.40:
        print(f"\n  [WARN] THIS WEEK = {week_count} — seems low")
        print(f"  [WARN] This should be the dominant label for weekly news")

    print(f"\n  EXPECTED: ~35% IMMEDIATE, ~55% THIS WEEK, ~10% WATCH, 0% THIS QUARTER")
    print()

    # ── Stats ─────────────────────────────────────────────────────────────────
    dist       = urgency_dist
    total_time = time.time() - run_start

    # Cost estimate: deepseek-chat pricing ($0.27/M input, $1.10/M output)
    n_total          = len(articles)
    est_in_tokens    = n_total * 550   # system (~230) + user prompt (~320)
    est_out_tokens   = n_total * 200
    est_cost         = (est_in_tokens * 0.27 + est_out_tokens * 1.10) / 1_000_000

    # ── Write keypoints.json ──────────────────────────────────────────────────
    OUTPUT_FILE.write_text(
        json.dumps(keypoints, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Write keypoints_summary.json ──────────────────────────────────────────
    summary = {
        "run_at":               run_at,
        "today":                TODAY,
        "input_articles":       len(articles),
        "succeeded":            len(keypoints),
        "failed":               len(failed_ids),
        "failed_ids":           failed_ids,
        "urgency_distribution":   dict(urgency_dist),
        "narrative_distribution": dict(narrative_dist),
        "citation_missing":     len(missing_citation),
        "est_input_tokens":     est_in_tokens,
        "est_output_tokens":    est_out_tokens,
        "est_cost_usd":         round(est_cost, 4),
        "total_time_seconds":   round(total_time, 1),
        "model":                MODEL,
        "workers":              num_workers,
        "keys_used":            len(active_keys),
    }
    (OUTPUT_DIR / "keypoints_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Write keypoints_log.txt ───────────────────────────────────────────────
    urgency_lines = [
        f"  {label:<14}: {dist.get(label, 0):>4}"
        for label in ["IMMEDIATE", "THIS WEEK", "THIS QUARTER", "WATCH"]
    ]
    citation_ok = len(keypoints) - len(missing_citation)
    log_lines = [
        "=" * 55,
        "  DHAKA'A — KEYPOINT EXTRACTION DONE",
        f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 55, "",
        f"  Processed       : {len(articles)}",
        f"  Succeeded       : {len(keypoints)}",
        f"  Failed          : {len(failed_ids)}",
        "",
        "  URGENCY SPLIT",
        "  -------------",
        *urgency_lines,
        "",
        "  CITATION CHECK",
        "  --------------",
        f"  citation_url present  : {citation_ok}/{len(keypoints)} {'OK' if not missing_citation else 'WARN'}",
        "  Paywalled sources kept: intentional",
        "",
        f"  Est. cost   : ~${est_cost:.4f}",
        f"  Time        : {total_time / 60:.1f} minutes",
        f"  Output      : {OUTPUT_FILE}",
        "=" * 55,
    ]
    (OUTPUT_DIR / "keypoints_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8"
    )

    # ── Console final output ──────────────────────────────────────────────────
    print()
    print("═" * 55)
    print("  DHAKA’A (ذكاء) — KEYPOINT EXTRACTION DONE")
    print("═" * 55)
    print()
    print(f"  Processed       : {len(articles)}")
    print(f"  Succeeded       : {len(keypoints)}")
    print(f"  Failed          : {len(failed_ids)}")
    print()
    print("  URGENCY SPLIT")
    print("  ─" * 15)
    print(f"  IMMEDIATE     : {urgency_dist.get('IMMEDIATE', 0):>4}  (target ~35%)")
    print(f"  THIS WEEK     : {urgency_dist.get('THIS WEEK', 0):>4}  (target ~55%)")
    print(f"  WATCH         : {urgency_dist.get('WATCH', 0):>4}  (target ~10%)")
    print(f"  THIS QUARTER  : {urgency_dist.get('THIS QUARTER', 0):>4}  (must be 0)")
    print()
    print("  CITATION CHECK")
    print("  ─" * 15)
    print(f"  citation_url present  : {citation_ok}/{len(keypoints)} {'✓' if not missing_citation else '⚠'}")
    print("  Paywalled sources kept: intentional ✓")
    print()
    print(f"  Est. cost   : ~${est_cost:.4f}")
    print(f"  Time        : ~{total_time / 60:.1f} minutes")
    print()
    print(f"  Output → {OUTPUT_FILE}")
    print("═" * 55)
    print()


if __name__ == "__main__":
    main()
