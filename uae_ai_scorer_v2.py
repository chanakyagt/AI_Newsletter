#!/usr/bin/env python3
"""
uae_ai_scorer_v2.py — Step 5
Score ~5100 articles via DeepSeek, apply source/tone bonuses, select newsletter candidates.

Input:  news_output/deduped/distinct_articles.json
Output: news_output/scored/  (5 files)
"""

import json
import os
import queue
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import sys
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ──────────────────────────────── CONFIG ──────────────────────────────────────
INPUT_FILE   = Path("news_output/deduped/distinct_articles.json")
OUTPUT_DIR   = Path("news_output/scored")

DEEPSEEK_API_KEY_1 = os.environ.get("Deepseek_API_Key_1")
DEEPSEEK_API_KEY_2 = os.environ.get("Deepseek_API_Key_2")
DEEPSEEK_API_KEY_3 = os.environ.get("Deepseek_API_Key_3")
DEEPSEEK_MODEL     = "deepseek-v4-flash"
BASE_URL           = "https://api.deepseek.com"

WORKERS_PER_KEY = 5       # concurrent workers per API key  (5 × 3 keys = 15 total)
BATCH_SIZE      = 3

TOP_ARTICLE_THRESHOLD = 70

PRIMARY_SUBCATEGORIES = {
    "INVESTMENT":       20,
    "GOVT_POLICY":      20,
    "ENTERPRISE_AI":    20,
    "NATIONAL_VISION":  20,
    "GLOBAL_DEAL":      20,
    "EXECUTIVE_VOICE":  20,
    "SECTOR_LOGISTICS": 20,
}

SECONDARY_SUBCATEGORIES = {
    "STARTUP":         3,
    "SECTOR_FINANCE":  3,
    "SECTOR_HEALTH":   3,
    "SECTOR_ENERGY":   3,
    "RISK_ETHICS":     3,
}

ALL_SUBCATEGORIES = {**PRIMARY_SUBCATEGORIES, **SECONDARY_SUBCATEGORIES}

CREDIBILITY_BONUS = {5: 8, 4: 5, 3: 2, 2: 0, 1: 0}
POSITIVITY_BONUS  = 5
CHECKPOINT_EVERY  = 50
# ──────────────────────────────────────────────────────────────────────────────

# ~1480 chars / ~370 tokens — within 400 token limit
SYSTEM_PROMPT = """\
You score UAE AI news articles for an executive newsletter targeting CXOs, CFOs, CTOs, and board members of major UAE organisations. The newsletter positions a UAE company as a voice of AI in the region. Tone is positive and opportunity-focused — UAE's AI future is exciting.

Score each article on 5 dimensions (0-20 each, total 0-100):

1. UAE_RELEVANCE: How directly does this involve UAE (govt/companies/policy/investment)?
20=UAE-exclusive, 15=UAE primary, 10=GCC with UAE angle, 5=global with UAE impact, 0=no UAE link

2. EXEC_VALUE: Does this change a UAE board-level decision in next 90 days?
20=immediate board action, 15=exec attention this quarter, 10=strategic context, 5=interesting not actionable, 0=no exec value

3. SPECIFICITY: Named actors + concrete numbers + verifiable dates?
20=all three, 15=two, 10=one, 5=vague, 0=speculation

4. SECTOR_IMPACT: Importance to UAE AI ecosystem?
20=national AI strategy/infrastructure/sovereign investment, 15=major sector transformation, 10=enterprise adoption, 5=adjacent tech, 0=irrelevant

5. TIMELINESS: How fresh?
20=last 48hrs, 15=this week, 10=this month, 5=older but relevant, 0=stale

Subcategory — assign exactly one:
INVESTMENT, GOVT_POLICY, ENTERPRISE_AI, NATIONAL_VISION, GLOBAL_DEAL, EXECUTIVE_VOICE, SECTOR_LOGISTICS, STARTUP, SECTOR_FINANCE, SECTOR_HEALTH, SECTOR_ENERGY, RISK_ETHICS

Tone — assign exactly one:
OPPORTUNITY (investment/launch/achievement/growth), NEUTRAL (factual/analysis), RISK (disruption/job loss/regulation burden)

Return JSON array only. No other text.
[{"id":"...","s":[u,e,sp,sec,t],"sub":"...","tone":"...","r":"rationale max 80 chars"}]"""

_CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"


# ──────────────────────────── ARTICLE HELPERS ─────────────────────────────────
def slim_article(article: dict) -> dict:
    return {
        "id":      article.get("article_id", ""),
        "title":   (article.get("title")   or "")[:150],
        "summary": (article.get("summary") or "")[:300],
        "date":    (article.get("published_date") or "")[:10],
    }


def build_user_message(batch: list) -> str:
    lines = [f"Score {len(batch)} articles:"]
    for i, article in enumerate(batch, 1):
        s = slim_article(article)
        lines.append(f"{i}|{s['id']}|{s['title']}|{s['summary']}|{s['date']}")
    return "\n".join(lines)


def default_score(article: dict) -> dict:
    return {
        "article_id": article.get("article_id", ""),
        "score_breakdown": {
            "uae_relevance": 0,
            "exec_value":    0,
            "specificity":   0,
            "sector_impact": 0,
            "timeliness":    0,
        },
        "base_score":  0,
        "subcategory": "ENTERPRISE_AI",
        "tone":        "NEUTRAL",
        "rationale":   "Scoring failed — default score assigned",
    }


# ──────────────────────────────── API CALL ────────────────────────────────────
def call_deepseek(client: OpenAI, batch: list) -> str:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(batch)},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def parse_response(raw: str, batch_articles: list):
    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    clean = clean.strip()

    try:
        items = json.loads(clean)
    except json.JSONDecodeError:
        return None

    if not isinstance(items, list):
        return None

    results = []
    for item in items:
        scores = item.get("s", [])
        if not isinstance(scores, list):
            scores = []
        scores = [int(v) for v in scores[:5]]
        while len(scores) < 5:
            scores.append(0)

        results.append({
            "article_id": item.get("id", ""),
            "score_breakdown": {
                "uae_relevance": scores[0],
                "exec_value":    scores[1],
                "specificity":   scores[2],
                "sector_impact": scores[3],
                "timeliness":    scores[4],
            },
            "base_score":  sum(scores),
            "subcategory": item.get("sub", "ENTERPRISE_AI"),
            "tone":        item.get("tone", "NEUTRAL"),
            "rationale":   str(item.get("r", ""))[:100],
        })

    return results if results else None


# ────────────────────────── SCORING BONUSES ──────────────────────────────────
def calculate_final_score(article: dict, llm_result: dict) -> dict:
    from url_resolution.unified_registry import get_profile

    base       = llm_result["base_score"]
    profile    = get_profile(source_name=article.get("source", ""))
    tier       = profile["credibility_tier"]
    cred_bonus = CREDIBILITY_BONUS.get(tier, 0)
    pos_bonus  = POSITIVITY_BONUS if llm_result.get("tone") == "OPPORTUNITY" else 0
    final      = min(base + cred_bonus + pos_bonus, 100)

    return {
        **llm_result,
        "credibility_tier":  tier,
        "credibility_bonus": cred_bonus,
        "positivity_bonus":  pos_bonus,
        "final_score":       final,
    }


# ──────────────────────────── CHECKPOINT ──────────────────────────────────────
def save_checkpoint(results: dict):
    _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "scored_ids": list(results.keys()),
        "scores":     results,
        "saved_at":   datetime.now().isoformat(),
        "count":      len(results),
    }
    tmp = _CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_CHECKPOINT_PATH)


def load_checkpoint() -> dict:
    if not _CHECKPOINT_PATH.exists():
        return {}
    try:
        data = json.loads(_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        count = data.get("count", 0)
        print(f"  [RESUME] Checkpoint found: {count:,} articles already scored")
        return data.get("scores", {})
    except Exception as e:
        print(f"  [WARN] Failed to load checkpoint ({e}) — starting fresh")
        return {}


# ────────────────────────── NEWSLETTER SELECTION ─────────────────────────────
def select_newsletter_candidates(scored_articles: list) -> list:
    groups = defaultdict(list)
    for a in scored_articles:
        sub = a.get("scoring", {}).get("subcategory", "ENTERPRISE_AI")
        groups[sub if sub in ALL_SUBCATEGORIES else "ENTERPRISE_AI"].append(a)

    candidates = []
    for subcat, cap in ALL_SUBCATEGORIES.items():
        bucket = sorted(
            groups[subcat],
            key=lambda x: x["scoring"]["final_score"],
            reverse=True,
        )[:cap]
        tier = "PRIMARY" if subcat in PRIMARY_SUBCATEGORIES else "SECONDARY"
        for rank, article in enumerate(bucket, 1):
            article["subcategory_rank"] = rank
            article["subcategory_cap"]  = cap
            article["subcategory_tier"] = tier
        candidates.extend(bucket)

    candidates.sort(key=lambda x: x["scoring"]["final_score"], reverse=True)
    for rank, article in enumerate(candidates, 1):
        article["newsletter_rank"] = rank

    return candidates


# ──────────────────────────────── MAIN ───────────────────────────────────────
def main():
    run_start = time.time()
    run_at    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print()
    print("═" * 63)
    print("  UAE AI NEWSLETTER — ARTICLE SCORING")
    print(f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 63)
    print()

    # ── Validate ──────────────────────────────────────────────────────────────
    active_keys = []
    for i, key in enumerate([DEEPSEEK_API_KEY_1, DEEPSEEK_API_KEY_2, DEEPSEEK_API_KEY_3], 1):
        label = f"DEEPSEEK_API_KEY_{i}"
        if key:
            print(f"  [KEY] {label} ✓")
            active_keys.append((f"KEY-{i}", key))
        else:
            print(f"  [KEY] {label} ✗  (not set)")

    if not active_keys:
        print("  [ERROR] No DeepSeek API keys found. Set DEEPSEEK_API_KEY_1/2/3 in .env")
        raise SystemExit(1)

    num_workers = WORKERS_PER_KEY * len(active_keys)
    print(f"  [PARALLEL] {len(active_keys)} key(s) × {WORKERS_PER_KEY} workers = {num_workers} total concurrent workers")

    try:
        from url_resolution.unified_registry import get_profile  # noqa: F401
        print("  [REGISTRY] unified_registry.py loaded ✓")
    except ImportError:
        print("  [ERROR] url_resolution/unified_registry.py not found.")
        print("  Run generate_unified_registry.py first.")
        raise SystemExit(1)

    if not INPUT_FILE.exists():
        print(f"  [ERROR] Input file not found: {INPUT_FILE}")
        raise SystemExit(1)

    # ── Load articles ─────────────────────────────────────────────────────────
    articles = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    print(f"  [INPUT]  {len(articles):,} articles loaded")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prior_scores = load_checkpoint()
    prior_ids    = set(prior_scores.keys())

    unscored = [a for a in articles if a.get("article_id") not in prior_ids]
    print(f"  [QUEUE]  {len(unscored):,} unscored  |  {len(prior_ids):,} already scored")
    print()

    # ── Build batches ─────────────────────────────────────────────────────────
    batches       = [unscored[i:i+BATCH_SIZE] for i in range(0, len(unscored), BATCH_SIZE)]
    total_batches = len(batches)
    print(f"  Batches to process : {total_batches:,}  ({BATCH_SIZE} articles each)")
    print(f"  Workers            : {num_workers} parallel  ({len(active_keys)} keys × {WORKERS_PER_KEY} per key)")
    print(f"  Checkpoint every   : {CHECKPOINT_EVERY} batches  ({CHECKPOINT_EVERY * BATCH_SIZE} articles)")
    print()

    # ── Shared state ──────────────────────────────────────────────────────────
    work_q       = queue.Queue()
    results_dict = dict(prior_scores)
    results_lock = threading.Lock()
    counter      = {"done": 0}
    counter_lock = threading.Lock()
    print_lock   = threading.Lock()

    for batch in batches:
        work_q.put(batch)

    # One client per key — shared across all WORKERS_PER_KEY workers on that key
    key_clients = {label: OpenAI(api_key=key, base_url=BASE_URL)
                   for label, key in active_keys}

    # Per-key rate pause: first worker to hit 429 clears the event (pauses all
    # workers on that key); the other keys' workers are completely unaffected.
    key_pause_events = {label: threading.Event() for label, _ in active_keys}
    key_pause_locks  = {label: threading.Lock()  for label, _ in active_keys}
    for ev in key_pause_events.values():
        ev.set()   # set = not paused

    # ── Worker ────────────────────────────────────────────────────────────────
    def worker(worker_id: int, key_label: str, client: OpenAI,
               pause_event: threading.Event, pause_lock: threading.Lock):
        while True:
            try:
                batch = work_q.get(timeout=2)
            except queue.Empty:
                break

            success = False
            for attempt in range(3):
                pause_event.wait()   # block only if this key is rate-paused

                try:
                    raw    = call_deepseek(client, batch)
                    parsed = parse_response(raw, batch)

                    if parsed is not None:
                        with results_lock:
                            for r in parsed:
                                if r["article_id"]:
                                    results_dict[r["article_id"]] = r
                        success = True
                        break
                    else:
                        with print_lock:
                            print(f"\n  [{key_label}] Parse failed (attempt {attempt+1}/3) — retrying")
                        time.sleep(2)

                except RateLimitError:
                    if pause_lock.acquire(blocking=False):
                        # First worker on this key to hit 429 — trigger key-level pause
                        pause_event.clear()
                        with print_lock:
                            print(f"\n  [{key_label}] Rate limited — pausing {WORKERS_PER_KEY} workers on this key 60s")
                        time.sleep(60)
                        pause_event.set()
                        pause_lock.release()
                    else:
                        # Another worker already handling the pause — just wait it out
                        pause_event.wait()
                    # Continue loop — will retry this batch after key recovers

                except Exception as e:
                    wait = 3 * (attempt + 1)
                    with print_lock:
                        print(f"\n  [{key_label}] Error attempt {attempt+1}/3: {e!r} — retry in {wait}s")
                    time.sleep(wait)

            if not success:
                with results_lock:
                    for article in batch:
                        aid = article.get("article_id", "")
                        if aid:
                            results_dict[aid] = default_score(article)
                with print_lock:
                    ids = [a.get("article_id", "?")[:8] for a in batch]
                    print(f"\n  [{key_label}] Batch permanently failed — default scores: {ids}")

            # Progress + checkpoint
            with counter_lock:
                counter["done"] += 1
                done = counter["done"]

            if total_batches > 0 and done % CHECKPOINT_EVERY == 0:
                with results_lock:
                    snapshot = dict(results_dict)
                save_checkpoint(snapshot)
                with print_lock:
                    pct = done / total_batches * 100
                    print(f"\n  [CHECKPOINT] {done}/{total_batches} batches ({pct:.0f}%) — {len(snapshot):,} scored")
            elif done % 5 == 0:
                with print_lock:
                    pct = done / total_batches * 100 if total_batches else 100
                    print(f"  Progress: {done}/{total_batches} batches ({pct:.0f}%)        ", end="\r")

            work_q.task_done()

    # ── Run workers ───────────────────────────────────────────────────────────
    if batches:
        print("  Scoring in progress...")
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for key_label, _ in active_keys:
                    client    = key_clients[key_label]
                    pause_ev  = key_pause_events[key_label]
                    pause_lk  = key_pause_locks[key_label]
                    for wid in range(WORKERS_PER_KEY):
                        pool.submit(worker, wid, key_label, client, pause_ev, pause_lk)
        except KeyboardInterrupt:
            print("\n  [INTERRUPT] Ctrl+C detected — saving checkpoint before exit...")
            with results_lock:
                save_checkpoint(dict(results_dict))
            print(f"  [SAVED] {len(results_dict):,} articles scored. Run again to resume from here.")
            raise SystemExit(0)
        print()

    # Final checkpoint
    save_checkpoint(results_dict)
    print(f"  [DONE] API scoring complete: {len(results_dict):,} articles scored")
    print()

    # ── Apply bonuses ─────────────────────────────────────────────────────────
    print("  Applying credibility and positivity bonuses...")
    scored_articles = []
    failed_count    = 0

    for article in articles:
        aid = article.get("article_id", "")
        if not aid:
            continue
        llm_result = results_dict.get(aid) or default_score(article)
        if llm_result.get("rationale", "").startswith("Scoring failed"):
            failed_count += 1
        scoring  = calculate_final_score(article, llm_result)
        enriched = {**article, "scoring": scoring}
        scored_articles.append(enriched)

    scored_articles.sort(key=lambda x: x["scoring"]["final_score"], reverse=True)

    # ── Select newsletter candidates ──────────────────────────────────────────
    candidates = select_newsletter_candidates(scored_articles)
    total_time = time.time() - run_start

    # ── Compute stats ─────────────────────────────────────────────────────────
    score_dist  = {"90-100": 0, "80-89": 0, "70-79": 0, "60-69": 0, "<60": 0}
    tone_dist   = defaultdict(int)
    cred_dist   = defaultdict(int)
    pos_count   = 0

    for a in scored_articles:
        fs   = a["scoring"]["final_score"]
        tone = a["scoring"]["tone"]
        tier = a["scoring"]["credibility_tier"]
        cb   = a["scoring"]["credibility_bonus"]
        pb   = a["scoring"]["positivity_bonus"]

        if   fs >= 90: score_dist["90-100"] += 1
        elif fs >= 80: score_dist["80-89"]  += 1
        elif fs >= 70: score_dist["70-79"]  += 1
        elif fs >= 60: score_dist["60-69"]  += 1
        else:          score_dist["<60"]    += 1

        tone_dist[tone] += 1
        if cb > 0:
            cred_dist[tier] += 1
        if pb > 0:
            pos_count += 1

    top_count = score_dist["90-100"] + score_dist["80-89"] + score_dist["70-79"]

    # Candidate breakdown by subcategory
    cand_counts = defaultdict(int)
    cand_tops   = defaultdict(int)
    for c in candidates:
        sub = c["scoring"]["subcategory"]
        cand_counts[sub] += 1
        cand_tops[sub] = max(cand_tops[sub], c["scoring"]["final_score"])

    api_calls         = total_batches
    est_input_tokens  = api_calls * 235
    est_output_tokens = api_calls * 80
    est_cost          = (est_input_tokens * 0.27 + est_output_tokens * 1.10) / 1_000_000

    # ── Write output files ────────────────────────────────────────────────────
    (OUTPUT_DIR / "scored_articles.json").write_text(
        json.dumps(scored_articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top_articles = [a for a in scored_articles
                    if a["scoring"]["final_score"] >= TOP_ARTICLE_THRESHOLD]
    (OUTPUT_DIR / "top_articles.json").write_text(
        json.dumps(top_articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "newsletter_candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    subcat_summary = {}
    for sub in ALL_SUBCATEGORIES:
        scores_for_sub = [a["scoring"]["final_score"] for a in scored_articles
                          if a["scoring"]["subcategory"] == sub]
        subcat_summary[sub] = {
            "total_scored":   len(scores_for_sub),
            "selected":       cand_counts.get(sub, 0),
            "top_score":      max(scores_for_sub) if scores_for_sub else 0,
            "avg_score":      round(sum(scores_for_sub) / len(scores_for_sub), 1) if scores_for_sub else 0,
        }

    score_summary = {
        "run_at":                    run_at,
        "input_articles":            len(articles),
        "scored_articles":           len(scored_articles),
        "failed_articles":           failed_count,
        "score_distribution":        score_dist,
        "top_articles":              top_count,
        "newsletter_candidates":     len(candidates),
        "subcategory_breakdown":     subcat_summary,
        "tone_breakdown":            dict(tone_dist),
        "credibility_bonus_applied": {f"tier_{k}": v for k, v in sorted(cred_dist.items())},
        "positivity_bonus_applied":  pos_count,
        "api_calls":                 api_calls,
        "total_time_seconds":        round(total_time, 1),
        "estimated_input_tokens":    est_input_tokens,
        "estimated_output_tokens":   est_output_tokens,
        "estimated_cost_usd":        round(est_cost, 4),
        "top_article_threshold":     TOP_ARTICLE_THRESHOLD,
        "scoring_model":             DEEPSEEK_MODEL,
    }
    (OUTPUT_DIR / "score_summary.json").write_text(
        json.dumps(score_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── scoring_log.txt ───────────────────────────────────────────────────────
    primary_lines   = [
        f"    {sub:<20}: {cand_counts.get(sub, 0):>2}  top: {cand_tops.get(sub, 0)}"
        for sub in PRIMARY_SUBCATEGORIES
    ]
    secondary_lines = [
        f"    {sub:<20}: {cand_counts.get(sub, 0):>2}  top: {cand_tops.get(sub, 0)}"
        for sub in SECONDARY_SUBCATEGORIES
    ]
    cred_lines = [
        f"    Tier {t} +{CREDIBILITY_BONUS[t]} ({['', '', 'regional', 'regional', 'major', 'govt'][t]:<8}): {cred_dist[t]:>5}"
        for t in sorted(cred_dist.keys(), reverse=True)
    ]

    log = [
        "═" * 63,
        "  UAE AI NEWSLETTER — SCORING COMPLETE",
        f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "═" * 63,
        "",
        f"  SCORED: {len(scored_articles):,} articles",
        "",
        "  SCORE DISTRIBUTION",
        "  ──────────────────",
        f"  90-100  : {score_dist['90-100']:>5}",
        f"  80-89   : {score_dist['80-89']:>5}",
        f"  70-79   : {score_dist['70-79']:>5}   ← newsletter-worthy",
        f"  60-69   : {score_dist['60-69']:>5}",
        f"  <60     : {score_dist['<60']:>5}",
        "",
        "  BONUSES",
        "  ───────",
        f"  Credibility applied : {sum(cred_dist.values()):>5} articles",
        *cred_lines,
        f"  Positivity +5       : {pos_count:>5} (OPPORTUNITY tone)",
        "",
        "  NEWSLETTER CANDIDATES (→ Firecrawl next)",
        "  ─────────────────────────────────────────",
        "  PRIMARY SUBCATEGORIES (top 20 each)",
        *primary_lines,
        "",
        "  SECONDARY SUBCATEGORIES (top 3 each)",
        *secondary_lines,
        "",
        f"  Total candidates    : {len(candidates)} → newsletter_candidates.json",
        "",
        "  COST",
        "  ────",
        f"  API calls           : {api_calls:,}",
        f"  Est. tokens         : ~{est_input_tokens + est_output_tokens:,}",
        f"  Est. cost           : ~${est_cost:.4f}",
        f"  Time                : {total_time / 60:.1f} minutes",
        "",
        "  OUTPUT → news_output/scored/",
        "═" * 63,
    ]
    (OUTPUT_DIR / "scoring_log.txt").write_text("\n".join(log), encoding="utf-8")

    # ── Console final output ──────────────────────────────────────────────────
    tier_labels = {5: "govt", 4: "major", 3: "regional", 2: "tier2", 1: "tier1"}

    print()
    print("═" * 63)
    print("  UAE AI NEWSLETTER — SCORING COMPLETE")
    print("═" * 63)
    print()
    print(f"  SCORED: {len(scored_articles):,} articles")
    print()
    print("  SCORE DISTRIBUTION")
    print("  ──────────────────")
    print(f"  90-100  : {score_dist['90-100']:>5}")
    print(f"  80-89   : {score_dist['80-89']:>5}")
    print(f"  70-79   : {score_dist['70-79']:>5}   ← newsletter-worthy")
    print(f"  60-69   : {score_dist['60-69']:>5}")
    print(f"  <60     : {score_dist['<60']:>5}")
    print()
    print("  BONUSES")
    print("  ───────")
    print(f"  Credibility applied   : {sum(cred_dist.values()):,} articles")
    for t in sorted(cred_dist.keys(), reverse=True):
        label = tier_labels.get(t, f"tier{t}")
        print(f"    Tier {t} +{CREDIBILITY_BONUS[t]} ({label:<8}): {cred_dist[t]:>5}")
    print(f"  Positivity +5         : {pos_count:>5} (OPPORTUNITY tone)")
    print()
    print("  NEWSLETTER CANDIDATES (→ Firecrawl next)")
    print("  ─────────────────────────────────────────")
    print("  PRIMARY SUBCATEGORIES (top 20 each)")
    for line in primary_lines:
        print(line)
    print()
    print("  SECONDARY SUBCATEGORIES (top 3 each)")
    for line in secondary_lines:
        print(line)
    print()
    print(f"  Total candidates       : {len(candidates)} → newsletter_candidates.json")
    print()
    print("  COST")
    print("  ────")
    print(f"  API calls             : {api_calls:,}")
    print(f"  Est. tokens           : ~{est_input_tokens + est_output_tokens:,}")
    print(f"  Est. cost             : ~${est_cost:.4f}")
    print(f"  Time                  : ~{total_time / 60:.1f} minutes")
    print()
    print(f"  OUTPUT → {OUTPUT_DIR}/")
    print("═" * 63)
    print()


if __name__ == "__main__":
    main()
