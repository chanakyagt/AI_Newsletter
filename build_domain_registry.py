"""
build_domain_registry.py — Domain Registry Builder
UAE AI Newsletter Pipeline

Reads url_resolution/resolved_urls.json, classifies every unique domain
as free / paywalled / metered / unknown, and writes:

  url_resolution/domain_registry.json   ← machine-readable full registry
  url_resolution/domain_report.txt      ← human-readable summary for operator
  domain_registry.py                    ← importable Python module (project root)

Run:
  python build_domain_registry.py

No external dependencies — standard library only.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

INPUT_FILE          = Path("url_resolution") / "checkpoint.json"
OUTPUT_DIR          = Path("url_resolution")
REGISTRY_JSON       = OUTPUT_DIR / "domain_registry.json"
DOMAIN_REPORT_TXT   = OUTPUT_DIR / "domain_report.txt"
REGISTRY_PY         = Path("domain_registry.py")   # written to project root

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — KNOWN DOMAIN GROUND TRUTH
# These override detected paywall status. Add new domains here to correct
# misclassifications, then rerun the script.
# ═════════════════════════════════════════════════════════════════════════════

KNOWN_PAYWALLED = {
    # UAE
    "thenational.ae":          "paywalled",
    "arabianbusiness.com":     "paywalled",
    # International finance
    "ft.com":                  "paywalled",
    "wsj.com":                 "paywalled",
    "bloomberg.com":           "paywalled",
    "economist.com":           "paywalled",
    "hbr.org":                 "paywalled",
    "businessinsider.com":     "paywalled",
    "forbes.com":              "metered",
    "telegraph.co.uk":         "paywalled",
    "nytimes.com":             "metered",
    "washingtonpost.com":      "metered",
    "thetimes.co.uk":          "paywalled",
    "theathletic.com":         "paywalled",
    # News wires
    "reuters.com":             "paywalled",
    # UAE free sources (confirm free even if detection was uncertain)
    "gulfnews.com":            "free",
    "khaleejtimes.com":        "free",
    "zawya.com":               "free",
    "wam.ae":                  "free",
    "alarabiya.net":           "free",
    "albayan.ae":              "free",
    "alkhaleej.ae":            "free",
    "emaratalyoum.com":        "free",
    "alroeya.com":             "free",
    "gulfbusiness.com":        "free",
    "agbi.com":                "free",
    "arabnews.com":            "free",
    "aljazeera.com":           "free",
    "skynewsarabia.com":       "free",
    "english.aawsat.com":      "free",
    "forbesmiddleeast.com":    "free",
    "menabytes.com":           "free",
    "tahawultech.com":         "free",
    "computermiddleeast.com":  "free",
    "itpro.me":                "free",
    "mediaquest.net":          "free",
}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def load_articles():
    if not INPUT_FILE.exists():
        print(f"\n[ERROR] Input file not found: {INPUT_FILE}")
        print("  Run paywallfetcher.py first to generate resolved_urls.json\n")
        raise SystemExit(1)

    articles = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    print(f"  Loaded {len(articles)} articles from {INPUT_FILE}")
    return articles


def extract_domain_stats(articles):
    """
    Build per-domain counts from articles with reliable paywall data.
    Skips resolution_failed and fetch_error — those have no usable domain.
    Returns dict: domain → {free: int, paywalled: int}
    """
    stats = defaultdict(lambda: {"free": 0, "paywalled": 0})

    skipped = 0
    for a in articles:
        status = a.get("paywall_status", "")
        if status not in ("free", "paywalled"):
            skipped += 1
            continue
        domain = (a.get("domain") or "").strip().lower()
        if not domain:
            skipped += 1
            continue
        stats[domain][status] += 1

    print(f"  Skipped {skipped} articles (resolution_failed / fetch_error / missing domain)")
    print(f"  Found {len(stats)} unique domains across {sum(v['free'] + v['paywalled'] for v in stats.values())} articles")
    return dict(stats)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DOMAIN CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def classify_domain(domain, free_count, paywalled_count):
    """
    Classify one domain. Returns (status, source, confidence, notes).

    Priority:
      1. KNOWN_PAYWALLED ground truth  → always wins
      2. Detected signals from articles
         - All free        → free
         - All paywalled   → paywalled
         - Mixed           → metered
      3. Zero detections  → unknown
    """
    total = free_count + paywalled_count

    # ── Priority 1: hardcoded known list ─────────────────────────────────────
    if domain in KNOWN_PAYWALLED:
        known_status = KNOWN_PAYWALLED[domain]
        # Confidence: high regardless — ground truth overrides
        confidence = "high"
        notes = ""
        # Flag if detection contradicts known status (useful for review)
        if total > 0:
            detected = _detected_status(free_count, paywalled_count)
            if detected != known_status and detected != "metered":
                notes = f"detected as {detected} but overridden by known_list"
        return known_status, "known_list", confidence, notes

    # ── Priority 2: from detected article data ────────────────────────────────
    if total == 0:
        return "unknown", "inferred", "low", "no usable detections"

    status = _detected_status(free_count, paywalled_count)
    source = "detected"

    if total >= 5:
        confidence = "high"
    elif total >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    notes = ""
    if status == "metered":
        notes = f"mixed signals: {free_count} free, {paywalled_count} paywalled"
    elif confidence == "low":
        notes = "only seen once — verify manually"

    return status, source, confidence, notes


def _detected_status(free_count, paywalled_count):
    """Derive status from detection counts."""
    if free_count > 0 and paywalled_count == 0:
        return "free"
    if paywalled_count > 0 and free_count == 0:
        return "paywalled"
    return "metered"


def build_registry(domain_stats):
    """Return list of domain record dicts, sorted by article_count descending."""
    records = []
    for domain, counts in domain_stats.items():
        free_n      = counts["free"]
        paywalled_n = counts["paywalled"]
        total       = free_n + paywalled_n

        status, source, confidence, notes = classify_domain(domain, free_n, paywalled_n)

        records.append({
            "domain":                 domain,
            "paywall_status":         status,
            "classification_source":  source,
            "article_count_this_run": total,
            "free_detections":        free_n,
            "paywalled_detections":   paywalled_n,
            "confidence":             confidence,
            "notes":                  notes,
        })

    records.sort(key=lambda r: (-r["article_count_this_run"], r["domain"]))
    return records


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OUTPUT: domain_registry.json
# ═════════════════════════════════════════════════════════════════════════════

def write_registry_json(records, generated_at):
    by_status = lambda s: [r for r in records if r["paywall_status"] == s]

    payload = {
        "generated_at":     generated_at,
        "total_domains":    len(records),
        "free_domains":     len(by_status("free")),
        "paywalled_domains": len(by_status("paywalled")),
        "metered_domains":  len(by_status("metered")),
        "unknown_domains":  len(by_status("unknown")),
        "domains":          records,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [SAVED] {REGISTRY_JSON}  ({len(records)} domains)")
    return payload


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OUTPUT: domain_registry.py  (importable module)
# ═════════════════════════════════════════════════════════════════════════════

def write_registry_py(records, generated_at, summary):
    free_records      = [r for r in records if r["paywall_status"] == "free"]
    paywalled_records = [r for r in records if r["paywall_status"] == "paywalled"]
    metered_records   = [r for r in records if r["paywall_status"] == "metered"]
    unknown_records   = [r for r in records if r["paywall_status"] == "unknown"]

    def domain_lines(recs):
        if not recs:
            return ["    # (none)"]
        max_len = max(len(r["domain"]) for r in recs)
        return [
            f'    "{r["domain"]}":{" " * (max_len - len(r["domain"]) + 1)}"{r["paywall_status"]}",'
            for r in recs
        ]

    run_dt   = generated_at[:19].replace("T", " ")
    n        = summary["total_domains"]
    n_free   = summary["free_domains"]
    n_pay    = summary["paywalled_domains"]
    n_met    = summary["metered_domains"]
    n_unk    = summary["unknown_domains"]

    lines = [
        "# " + "═" * 63,
        "# DOMAIN REGISTRY — Auto-generated by build_domain_registry.py",
        f"# Generated   : {run_dt}",
        f"# Total domains: {n}  |  Free: {n_free}  |  Paywalled: {n_pay}  |  Metered: {n_met}  |  Unknown: {n_unk}",
        "#",
        "# DO NOT EDIT MANUALLY — regenerate by running build_domain_registry.py",
        "# To override a classification, add the domain to KNOWN_PAYWALLED",
        "# inside build_domain_registry.py and rerun.",
        "# " + "═" * 63,
        "",
        "DOMAIN_PAYWALL_STATUS = {",
    ]

    if free_records:
        lines.append("    # ── Free domains " + "─" * 44)
        lines.extend(domain_lines(free_records))
        lines.append("")

    if paywalled_records:
        lines.append("    # ── Paywalled domains " + "─" * 40)
        lines.extend(domain_lines(paywalled_records))
        lines.append("")

    if metered_records:
        lines.append("    # ── Metered domains (partial access) " + "─" * 25)
        lines.extend(domain_lines(metered_records))
        lines.append("")

    if unknown_records:
        lines.append("    # ── Unknown (seen once, unclassified) " + "─" * 23)
        lines.extend(domain_lines(unknown_records))
        lines.append("")

    lines += [
        "}",
        "",
        "",
        "# Pre-built lookup sets for O(1) membership testing.",
        "# Built once at import time and reused for every lookup.",
        'FREE_DOMAINS_SET      = frozenset(k for k, v in DOMAIN_PAYWALL_STATUS.items() if v == "free")',
        'PAYWALLED_DOMAINS_SET = frozenset(k for k, v in DOMAIN_PAYWALL_STATUS.items() if v == "paywalled")',
        'METERED_DOMAINS_SET   = frozenset(k for k, v in DOMAIN_PAYWALL_STATUS.items() if v == "metered")',
        "ALL_KNOWN_DOMAINS_SET = frozenset(DOMAIN_PAYWALL_STATUS.keys())",
        "",
        "",
        'def resolve_domain(raw_domain: str) -> str:',
        '    """',
        '    Strips subdomains to find a registry match.',
        '    business.thenational.ae → thenational.ae',
        '    tech.gulfnews.com       → gulfnews.com',
        '    Returns the matched registry domain, or the original if no match found.',
        '    """',
        '    d = raw_domain.lower().strip()',
        '    if d in ALL_KNOWN_DOMAINS_SET:',
        '        return d',
        '    parts = d.split(".")',
        '    for i in range(1, len(parts) - 1):',
        '        candidate = ".".join(parts[i:])',
        '        if candidate in ALL_KNOWN_DOMAINS_SET:',
        '            return candidate',
        '    return d',
        "",
        "",
        'def is_paywalled(domain: str) -> bool:',
        '    """Returns True if domain is paywalled or metered."""',
        '    d = resolve_domain(domain)',
        '    return d in PAYWALLED_DOMAINS_SET or d in METERED_DOMAINS_SET',
        "",
        "",
        'def is_free(domain: str) -> bool:',
        '    """Returns True if domain is confirmed free."""',
        '    return resolve_domain(domain) in FREE_DOMAINS_SET',
        "",
        "",
        'def is_known(domain: str) -> bool:',
        '    """Returns True if domain (or its parent domain) is in the registry."""',
        '    return resolve_domain(domain) in ALL_KNOWN_DOMAINS_SET',
        "",
        "",
        'def get_status(domain: str) -> str:',
        '    """Returns paywall status string. Returns \'unknown\' if domain not in registry."""',
        '    return DOMAIN_PAYWALL_STATUS.get(resolve_domain(domain), "unknown")',
    ]

    REGISTRY_PY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [SAVED] {REGISTRY_PY}  (importable module)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — OUTPUT: domain_report.txt
# ═════════════════════════════════════════════════════════════════════════════

def write_domain_report(records, generated_at, summary):
    run_dt   = generated_at[:19].replace("T", " ")
    n        = max(summary["total_domains"], 1)

    def pct(x):
        return f"{x / n * 100:.0f}%"

    def domain_block(recs):
        out = []
        for r in recs:
            src   = f"[{r['classification_source']}]"
            count = r["article_count_this_run"]
            out.append(f"  {r['domain']:<35} {count:>3} article{'s' if count != 1 else ' '}   {src}")
        return out

    free_recs      = [r for r in records if r["paywall_status"] == "free"]
    paywalled_recs = [r for r in records if r["paywall_status"] == "paywalled"]
    metered_recs   = [r for r in records if r["paywall_status"] == "metered"]
    unknown_recs   = [r for r in records if r["paywall_status"] == "unknown"]
    low_conf_recs  = [r for r in records if r["confidence"] == "low" and r["paywall_status"] != "unknown"]
    not_in_known   = [r for r in records if r["domain"] not in KNOWN_PAYWALLED and r["classification_source"] == "detected"]

    SEP = "═" * 59
    DIV = "─" * 43

    lines = [
        SEP,
        "  DOMAIN REGISTRY REPORT",
        f"  Generated : {run_dt}",
        f"  Source    : {INPUT_FILE}",
        SEP,
        "",
        "  SUMMARY",
        "  " + DIV[:7],
        f"  Total unique domains seen  : {summary['total_domains']}",
        f"  Free and accessible        : {summary['free_domains']}  ({pct(summary['free_domains'])})",
        f"  Paywalled                  : {summary['paywalled_domains']}  ({pct(summary['paywalled_domains'])})",
        f"  Metered (partial access)   : {summary['metered_domains']}  ({pct(summary['metered_domains'])})",
        f"  Unknown (seen once)        : {summary['unknown_domains']}  ({pct(summary['unknown_domains'])})",
        "",
        "  FREE DOMAINS (ranked by article count)",
        "  " + DIV,
    ]
    lines.extend(domain_block(free_recs) or ["  (none)"])

    lines += ["", "  PAYWALLED DOMAINS (ranked by article count)", "  " + DIV]
    lines.extend(domain_block(paywalled_recs) or ["  (none)"])

    lines += ["", "  METERED DOMAINS (partial access)", "  " + DIV]
    lines.extend(domain_block(metered_recs) or ["  (none)"])

    if unknown_recs:
        lines += ["", "  UNKNOWN DOMAINS", "  " + DIV]
        lines.extend(domain_block(unknown_recs))

    if low_conf_recs:
        lines += [
            "",
            "  LOW CONFIDENCE CLASSIFICATIONS (review these)",
            "  " + DIV,
        ]
        for r in low_conf_recs:
            note = r["notes"] or "verify manually"
            lines.append(
                f"  {r['domain']:<35} {r['article_count_this_run']:>2} article   "
                f"[detected: {r['paywall_status']}] — {note}"
            )

    if not_in_known:
        lines += [
            "",
            "  ACTION REQUIRED",
            "  " + DIV,
            "  The following domains were detected but are not in KNOWN_PAYWALLED.",
            "  If classification looks wrong, add them and rerun:",
            "",
        ]
        for r in not_in_known:
            lines.append(
                f"  {r['domain']:<35} detected as: {r['paywall_status']:<10} "
                f"({r['article_count_this_run']} article{'s' if r['article_count_this_run'] != 1 else ''}, "
                f"{r['confidence']} confidence)"
            )

    lines.append("")
    lines.append(SEP)

    DOMAIN_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [SAVED] {DOMAIN_REPORT_TXT}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run():
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"""
{'='*55}
  DOMAIN REGISTRY BUILDER
{'='*55}
  Input  : {INPUT_FILE}
  Output : {OUTPUT_DIR}/
{'='*55}
""")

    articles     = load_articles()
    domain_stats = extract_domain_stats(articles)
    records      = build_registry(domain_stats)

    by_status = {
        s: len([r for r in records if r["paywall_status"] == s])
        for s in ("free", "paywalled", "metered", "unknown")
    }
    summary = {
        "total_domains":    len(records),
        "free_domains":     by_status["free"],
        "paywalled_domains": by_status["paywalled"],
        "metered_domains":  by_status["metered"],
        "unknown_domains":  by_status["unknown"],
    }

    print()
    payload = write_registry_json(records, generated_at)
    write_registry_py(records, generated_at, summary)
    write_domain_report(records, generated_at, summary)

    print(f"""
{'='*55}
  COMPLETE
{'='*55}
  Total domains    : {summary['total_domains']}
  Free             : {summary['free_domains']}
  Paywalled        : {summary['paywalled_domains']}
  Metered          : {summary['metered_domains']}
  Unknown          : {summary['unknown_domains']}

  Import in other scripts:
    from domain_registry import is_paywalled, is_free, get_status
{'='*55}
""")


if __name__ == "__main__":
    run()
