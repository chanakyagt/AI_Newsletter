#!/usr/bin/env python3
"""
fix_urgency_labels.py
Post-process keypoints.json to redistribute urgency labels.

Root cause: age-based IMMEDIATE rule (≤2 days) produced 0 hits because
articles span Feb 27 – Apr 27 and keypoints ran Apr 29.

New deterministic rules (content + recency, no strict age cutoff):
  IMMEDIATE: STANDALONE or DEVELOPMENT, score ≥ 75, published within 14 days of Apr 29
             Target ~35%
  WATCH:     STRUCTURAL, score < 80, regardless of age
             Target ~10%
  THIS WEEK: everything else
             Target ~55%
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

KEYPOINTS_FILE = Path("news_output/keypoints/keypoints.json")
SUMMARY_FILE   = Path("news_output/keypoints/keypoints_summary.json")
RUN_DATE       = datetime(2026, 4, 29, tzinfo=timezone.utc)

IMMEDIATE_SCORE_MIN = 75
IMMEDIATE_AGE_MAX   = 14   # days
WATCH_SCORE_MAX     = 79   # STRUCTURAL with score ≤ this → WATCH

URGENCY_ORDER = {"IMMEDIATE": 0, "THIS WEEK": 1, "WATCH": 2}


def classify(kp: dict) -> str:
    ntype = kp.get("narrative_type", "STANDALONE").upper()
    score = kp.get("final_score", 0)
    pub   = kp.get("published_date", "")

    age_days = 99
    try:
        pub_dt   = datetime.fromisoformat(pub + "T00:00:00+00:00")
        age_days = max(0, (RUN_DATE - pub_dt).days)
    except Exception:
        pass

    if ntype in ("STANDALONE", "DEVELOPMENT") and score >= IMMEDIATE_SCORE_MIN and age_days <= IMMEDIATE_AGE_MAX:
        return "IMMEDIATE"

    if ntype == "STRUCTURAL" and score <= WATCH_SCORE_MAX:
        return "WATCH"

    return "THIS WEEK"


def main():
    kps = json.loads(KEYPOINTS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(kps)} keypoints")

    before = Counter(k.get("urgency_label") for k in kps)
    print(f"\nBefore: {dict(before)}")

    for kp in kps:
        new_label = classify(kp)
        kp["urgency_label"] = new_label
        kp["urgency_rationale"] = kp.get("urgency_rationale", "") + " [post-processed]"

    after = Counter(k.get("urgency_label") for k in kps)
    total = len(kps)
    print(f"After:  {dict(after)}")
    print()
    for label in ["IMMEDIATE", "THIS WEEK", "WATCH"]:
        count = after.get(label, 0)
        pct   = count / total * 100 if total else 0
        bar   = "█" * min(count // 2, 30)
        print(f"  {label:<14}: {count:>4}  ({pct:.0f}%)  {bar}")

    # Re-sort by urgency → score descending
    kps.sort(key=lambda x: (
        URGENCY_ORDER.get(x.get("urgency_label", "THIS WEEK"), 1),
        -x.get("final_score", 0),
    ))

    KEYPOINTS_FILE.write_text(
        json.dumps(kps, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved → {KEYPOINTS_FILE}")

    # Update summary
    try:
        summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
        summary["urgency_distribution"] = dict(after)
        summary["urgency_post_processed"] = True
        SUMMARY_FILE.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Updated → {SUMMARY_FILE}")
    except Exception as e:
        print(f"[WARN] Could not update summary: {e}")

    print()
    print(f"Target: ~35% IMMEDIATE, ~55% THIS WEEK, ~10% WATCH")
    print(f"Got:    {after.get('IMMEDIATE',0)/total*100:.0f}% IMMEDIATE, "
          f"{after.get('THIS WEEK',0)/total*100:.0f}% THIS WEEK, "
          f"{after.get('WATCH',0)/total*100:.0f}% WATCH")


if __name__ == "__main__":
    main()
