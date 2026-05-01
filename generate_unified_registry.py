"""
generate_unified_registry.py — Final Registry Generator
UAE AI Newsletter Pipeline

Reads url_resolution/unified_registry.json, applies tier corrections,
and bakes all 1006 domains into two self-contained output files:

  url_resolution/unified_registry.py    — importable pipeline module (no JSON at runtime)
  url_resolution/registry_summary.txt   — human-readable summary

Run:
  python generate_unified_registry.py

No external dependencies — standard library only.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

INPUT_FILE   = Path("url_resolution") / "unified_registry.json"
OUTPUT_DIR   = Path("url_resolution")
REGISTRY_PY  = OUTPUT_DIR / "unified_registry.py"
SUMMARY_TXT  = OUTPUT_DIR / "registry_summary.txt"

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TIER OVERRIDES
# Domains that inference rules mis-scored. Applied after loading the JSON.
# ═════════════════════════════════════════════════════════════════════════════

TIER_OVERRIDES = {
    # False CNN signal matches — press wires, not CNN
    "acnnewswire.com":       1,
    "prnewswire.com":        1,
    "businesswire.com":      1,
    # False guardian signal matches
    "slguardian.org":        1,   # Sierra Leone Guardian, not The Guardian UK
    "guardian.ng":           2,   # Nigeria Guardian — legitimate national paper
    # False verge signal match
    "convergence-now.com":   1,   # not The Verge
    # US government — not UAE-relevant news sources
    "blogs.loc.gov":         1,
    "murphy.senate.gov":     1,
    "congress.gov":          1,
    "whitehouse.gov":        1,
    # Subdomains of known Tier 3 sources — upgrade to correct tier
    "aljazeera.net":         3,   # Al Jazeera Arabic — same org as aljazeera.com
    "english.alarabiya.net": 3,   # Al Arabiya English — same org
    "arabic.cnn.com":        3,   # CNN Arabic
    "edition.cnn.com":       3,   # CNN International
    "thomsonreuters.com":    3,   # Thomson Reuters corporate
    # Middle East regional — correct to Tier 2
    "wired.me":              2,   # Wired Middle East
    "economymiddleeast.com": 2,
    "gulftoday.ae":          2,   # UAE English daily
    "sharjah24.ae":          2,   # Sharjah news
    "24.ae":                 2,   # UAE news portal
}

# Human-readable notes for the summary report (subset of overrides)
OVERRIDE_NOTES = {
    "acnnewswire.com":       "not CNN, is press wire",
    "prnewswire.com":        "not CNN, is press wire",
    "businesswire.com":      "not CNN, is press wire",
    "slguardian.org":        "not The Guardian UK — Sierra Leone paper",
    "guardian.ng":           "Nigeria Guardian — legitimate but Tier 2",
    "convergence-now.com":   "not The Verge",
    "blogs.loc.gov":         "US Library of Congress blog — not UAE-relevant",
    "murphy.senate.gov":     "US Senator site — not UAE-relevant",
    "congress.gov":          "US Congress — not UAE-relevant",
    "whitehouse.gov":        "US Government — not UAE-relevant",
    "aljazeera.net":         "Al Jazeera Arabic — same org as aljazeera.com",
    "english.alarabiya.net": "Al Arabiya English subdomain",
    "arabic.cnn.com":        "CNN Arabic subdomain",
    "edition.cnn.com":       "CNN International subdomain",
    "thomsonreuters.com":    "Thomson Reuters corporate (same as reuters.com)",
    "wired.me":              "Wired Middle East edition",
    "economymiddleeast.com": "UAE-relevant regional business",
    "gulftoday.ae":          "UAE English daily",
    "sharjah24.ae":          "Sharjah official news",
    "24.ae":                 "UAE news portal",
}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FIELD ALIGNMENT CONSTANTS  (for generated DOMAIN_REGISTRY)
# ═════════════════════════════════════════════════════════════════════════════

_INNER_FIELDS   = ["source_name", "credibility_tier", "is_paywalled",
                   "paywall_status", "tier_source", "article_count"]
_MAX_FIELD_LEN  = max(len(f'"{f}":') for f in _INNER_FIELDS)   # 19


def _fmt_field(field: str, value_str: str) -> str:
    """Return one aligned inner-dict line (8-space indent)."""
    key = f'"{field}":'
    pad = " " * (_MAX_FIELD_LEN - len(key) + 1)
    return f"        {key}{pad}{value_str},"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NORMALISATION  (mirrors the function baked into unified_registry.py)
# ═════════════════════════════════════════════════════════════════════════════

def _normalise(name: str) -> str:
    name = name.lower().strip()
    name = name.replace("(uae)", "").replace("(egypt)", "")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — INPUT LOADER + OVERRIDE APPLIER
# ═════════════════════════════════════════════════════════════════════════════

def load_registry() -> dict:
    if not INPUT_FILE.exists():
        print(f"\n[ERROR] Not found: {INPUT_FILE}")
        print("  Run build_unified_registry.py first.\n")
        raise SystemExit(1)

    raw  = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    domains_raw = raw.get("domains", {})

    profiles = {}
    for domain, info in domains_raw.items():
        profiles[domain] = {
            "domain":           domain,
            "source_name":      info.get("source_name", domain),
            "credibility_tier": info.get("credibility_tier", 1),
            "is_paywalled":     info.get("is_paywalled", False),
            "paywall_status":   info.get("paywall_status", "unknown"),
            "tier_source":      info.get("tier_source", "inferred"),
            "article_count":    info.get("article_count", 0),
        }

    print(f"  Loaded {len(profiles)} domains from {INPUT_FILE}")
    return profiles


def apply_overrides(profiles: dict) -> dict:
    """Apply TIER_OVERRIDES in-place. Returns {domain: (old_tier, new_tier)}."""
    applied = {}
    for domain, new_tier in TIER_OVERRIDES.items():
        if domain in profiles:
            old_tier = profiles[domain]["credibility_tier"]
            profiles[domain]["credibility_tier"] = new_tier
            profiles[domain]["tier_source"]      = "overridden"
            # Update is_paywalled is independent of tier — leave unchanged
            if old_tier != new_tier:
                applied[domain] = (old_tier, new_tier)

    print(f"  Applied {len(applied)} effective tier changes "
          f"({len(TIER_OVERRIDES)} overrides defined, "
          f"{len(TIER_OVERRIDES) - len(applied)} already correct or domain absent)")
    return applied


def build_source_to_domain(profiles: dict) -> dict:
    """
    Map source_name → domain for entries where the source_name is a real
    human-readable name (hardcoded or overridden), not just the domain itself.
    """
    return {
        p["source_name"]: domain
        for domain, p in profiles.items()
        if p["tier_source"] in ("hardcoded", "overridden")
        and p["source_name"] != domain
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_summary(profiles: dict) -> dict:
    by_tier    = {t: 0 for t in (5, 4, 3, 2, 1)}
    by_paywall = {"free": 0, "paywalled": 0, "metered": 0, "unknown": 0}

    for p in profiles.values():
        by_tier[p["credibility_tier"]] = by_tier.get(p["credibility_tier"], 0) + 1
        by_paywall[p["paywall_status"]] = by_paywall.get(p["paywall_status"], 0) + 1

    return {
        "total_domains":    len(profiles),
        "tier_distribution": by_tier,
        "free":             by_paywall["free"],
        "paywalled":        by_paywall["paywalled"],
        "metered":          by_paywall["metered"],
        "unknown":          by_paywall.get("unknown", 0),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CODE GENERATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

_TIER_BANNERS = {
    5: "TIER 5 — UAE Government and Official Bodies",
    4: "TIER 4 — Major UAE Publications",
    3: "TIER 3 — Regional Credible Outlets",
    2: "TIER 2 — International and Regional Press",
    1: "TIER 1 — All remaining domains",
}


def _domain_entry_lines(p: dict) -> list:
    """Return the multi-line block for one DOMAIN_REGISTRY entry."""
    is_pw = "True" if p["is_paywalled"] else "False"
    return [
        f"    {json.dumps(p['domain'])}: {{",
        _fmt_field("source_name",      json.dumps(p["source_name"], ensure_ascii=False)),
        _fmt_field("credibility_tier", str(p["credibility_tier"])),
        _fmt_field("is_paywalled",     is_pw),
        _fmt_field("paywall_status",   json.dumps(p["paywall_status"])),
        _fmt_field("tier_source",      json.dumps(p["tier_source"])),
        _fmt_field("article_count",    str(p["article_count"])),
        "    },",
    ]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — OUTPUT: unified_registry.py
# ═════════════════════════════════════════════════════════════════════════════

def write_registry_py(
    profiles: dict,
    source_to_domain: dict,
    generated_at: str,
    summary: dict,
) -> None:
    td      = summary["tier_distribution"]
    run_dt  = generated_at[:19].replace("T", " ")
    n       = summary["total_domains"]
    out     = []

    # ── Section 1: file header ────────────────────────────────────────────
    out += [
        "# " + "═" * 71,
        "# UNIFIED SOURCE REGISTRY",
        f"# Generated: {run_dt}",
        f"# Domains: {n}  |  Free: {summary['free']}  |  "
        f"Paywalled: {summary['paywalled']}  |  Metered: {summary['metered']}",
        f"# Tier 5: {td[5]}  "
        f"Tier 4: {td[4]}  "
        f"Tier 3: {td[3]}  "
        f"Tier 2: {td[2]}  "
        f"Tier 1: {td[1]}",
        "#",
        "# LOOKUP METHODS:",
        "#   Before Firecrawl (source name only):",
        '#     profile = get_profile(source_name="Gulf News")',
        "#",
        "#   After Firecrawl (domain available):",
        '#     profile = get_profile(domain="gulfnews.com")',
        "#",
        "# DO NOT EDIT MANUALLY — regenerate via generate_unified_registry.py",
        "# " + "═" * 71,
        "",
        "import re as _re",
        "",
        "",
    ]

    # ── Section 2: DOMAIN_REGISTRY ────────────────────────────────────────
    out.append("DOMAIN_REGISTRY: dict[str, dict] = {")

    sorted_profiles = sorted(
        profiles.values(),
        key=lambda p: (-p["credibility_tier"], -p["article_count"], p["domain"]),
    )
    current_tier = None
    for p in sorted_profiles:
        t = p["credibility_tier"]
        if t != current_tier:
            if current_tier is not None:
                out.append("")
            out += [
                "    # " + "═" * 52,
                f"    # {_TIER_BANNERS[t]}",
                "    # " + "═" * 52,
            ]
            current_tier = t
        out.extend(_domain_entry_lines(p))

    out += ["}", "", ""]

    # ── Section 3: SOURCE_TO_DOMAIN ───────────────────────────────────────
    max_src_len = max((len(json.dumps(s, ensure_ascii=False))
                       for s in source_to_domain), default=4)
    out += [
        "# Maps English source names to domains.",
        "# Used in Phase 1 (before Firecrawl) when only source name is available.",
        "SOURCE_TO_DOMAIN: dict[str, str] = {",
    ]
    for src, dom in sorted(source_to_domain.items()):
        src_str = json.dumps(src, ensure_ascii=False)
        pad     = " " * (max_src_len - len(src_str) + 1)
        out.append(f"    {src_str}:{pad}{json.dumps(dom)},")
    out += ["}", "", ""]

    # ── Section 4: _normalise + NORMALISED_SOURCE_TO_DOMAIN ───────────────
    out += [
        "def _normalise(name: str) -> str:",
        '    name = name.lower().strip()',
        '    name = name.replace("(uae)", "").replace("(egypt)", "")',
        r'    name = _re.sub(r"[^\w\s]", "", name)',
        r'    name = _re.sub(r"\s+", " ", name).strip()',
        '    return name',
        "",
        "",
        "NORMALISED_SOURCE_TO_DOMAIN: dict[str, str] = {",
        "    _normalise(source): domain",
        "    for source, domain in SOURCE_TO_DOMAIN.items()",
        "}",
        "",
        "",
    ]

    # ── Section 5: frozensets ─────────────────────────────────────────────
    out += [
        "# Built once at import time — O(1) membership testing",
        'FREE_DOMAINS      = frozenset(',
        '    k for k, v in DOMAIN_REGISTRY.items()',
        '    if v["paywall_status"] == "free"',
        ')',
        'PAYWALLED_DOMAINS = frozenset(',
        '    k for k, v in DOMAIN_REGISTRY.items()',
        '    if v["paywall_status"] == "paywalled"',
        ')',
        'METERED_DOMAINS   = frozenset(',
        '    k for k, v in DOMAIN_REGISTRY.items()',
        '    if v["paywall_status"] == "metered"',
        ')',
        'HIGH_CREDIBILITY  = frozenset(',
        '    k for k, v in DOMAIN_REGISTRY.items()',
        '    if v["credibility_tier"] >= 3',
        ')',
        'TIER_5_DOMAINS    = frozenset(',
        '    k for k, v in DOMAIN_REGISTRY.items()',
        '    if v["credibility_tier"] == 5',
        ')',
        'TIER_4_PLUS       = frozenset(',
        '    k for k, v in DOMAIN_REGISTRY.items()',
        '    if v["credibility_tier"] >= 4',
        ')',
        'ALL_KNOWN_DOMAINS = frozenset(DOMAIN_REGISTRY.keys())',
        "",
        "",
    ]

    # ── Section 6: helper functions ───────────────────────────────────────
    out += [
        'def _resolve_subdomain(domain: str) -> str:',
        '    """',
        '    Strips subdomains progressively until a registry match is found.',
        '    business.thenational.ae  →  thenational.ae',
        '    english.alarabiya.net    →  alarabiya.net',
        '    Returns original domain string if no match found.',
        '    """',
        '    domain = domain.lower().strip()',
        '    if domain in ALL_KNOWN_DOMAINS:',
        '        return domain',
        '    parts = domain.split(".")',
        '    for i in range(1, len(parts) - 1):',
        '        candidate = ".".join(parts[i:])',
        '        if candidate in ALL_KNOWN_DOMAINS:',
        '            return candidate',
        '    return domain',
        '',
        '',
        'def _default_profile(identifier: str) -> dict:',
        '    """Returns a Tier 1 unknown profile when identifier is not in registry."""',
        '    return {',
        '        "source_name":       identifier,',
        '        "domain":            None,',
        '        "credibility_tier":  1,',
        '        "is_paywalled":      False,',
        '        "paywall_status":    "unknown",',
        '        "tier_source":       "default",',
        '        "lookup_method":     "default",',
        '        "found_in_registry": False,',
        '    }',
        '',
        '',
        'def get_profile(source_name: str = None, domain: str = None) -> dict:',
        '    """',
        '    Universal profile lookup. Works in both pipeline phases.',
        '',
        '    Phase 1 — before Firecrawl (source name only):',
        '        profile = get_profile(source_name="Gulf News")',
        '        profile = get_profile(source_name="Al Khaleej (UAE)")',
        '',
        '    Phase 2 — after Firecrawl (domain available):',
        '        profile = get_profile(domain="gulfnews.com")',
        '        profile = get_profile(domain="business.thenational.ae")',
        '',
        '    Returns dict with keys:',
        '        source_name, domain, credibility_tier, is_paywalled,',
        '        paywall_status, tier_source, lookup_method, found_in_registry',
        '',
        '    Never raises. Unknown identifiers return Tier 1 default.',
        '    """',
        '    result        = None',
        '    lookup_method = "default"',
        '    resolved_d    = None',
        '',
        '    if domain:',
        '        d = domain.lower().strip()',
        '        if d in DOMAIN_REGISTRY:',
        '            result, resolved_d, lookup_method = (',
        '                DOMAIN_REGISTRY[d], d, "domain_direct"',
        '            )',
        '        else:',
        '            rd = _resolve_subdomain(d)',
        '            if rd in DOMAIN_REGISTRY:',
        '                result, resolved_d, lookup_method = (',
        '                    DOMAIN_REGISTRY[rd], rd, "domain_subdomain"',
        '                )',
        '',
        '    elif source_name:',
        '        if source_name in SOURCE_TO_DOMAIN:',
        '            d = SOURCE_TO_DOMAIN[source_name]',
        '            if d in DOMAIN_REGISTRY:',
        '                result, resolved_d, lookup_method = (',
        '                    DOMAIN_REGISTRY[d], d, "source_direct"',
        '                )',
        '        if result is None:',
        '            norm = _normalise(source_name)',
        '            if norm in NORMALISED_SOURCE_TO_DOMAIN:',
        '                d = NORMALISED_SOURCE_TO_DOMAIN[norm]',
        '                if d in DOMAIN_REGISTRY:',
        '                    result, resolved_d, lookup_method = (',
        '                        DOMAIN_REGISTRY[d], d, "source_normalised"',
        '                    )',
        '',
        '    if result is None:',
        '        return _default_profile(source_name or domain or "unknown")',
        '',
        '    return {',
        '        "source_name":       result["source_name"],',
        '        "domain":            resolved_d,',
        '        "credibility_tier":  result["credibility_tier"],',
        '        "is_paywalled":      result["is_paywalled"],',
        '        "paywall_status":    result["paywall_status"],',
        '        "tier_source":       result["tier_source"],',
        '        "lookup_method":     lookup_method,',
        '        "found_in_registry": True,',
        '    }',
        '',
        '',
        'def get_credibility(source_name: str = None, domain: str = None) -> int:',
        '    """Returns credibility tier 1-5. Works with either identifier."""',
        '    return get_profile(source_name=source_name, domain=domain)["credibility_tier"]',
        '',
        '',
        'def is_paywalled(source_name: str = None, domain: str = None) -> bool:',
        '    """Returns True if paywalled or metered. Works with either identifier."""',
        '    return get_profile(',
        '        source_name=source_name, domain=domain',
        '    )["paywall_status"] in ("paywalled", "metered")',
        '',
        '',
        'def is_free(source_name: str = None, domain: str = None) -> bool:',
        '    """Returns True if confirmed free. Works with either identifier."""',
        '    return get_profile(',
        '        source_name=source_name, domain=domain',
        '    )["paywall_status"] == "free"',
        '',
        '',
        'def get_source_name(domain: str) -> str:',
        '    """Returns human-readable source name for a domain."""',
        '    return get_profile(domain=domain)["source_name"]',
    ]

    # ── Section 7: self-test ──────────────────────────────────────────────
    out += [
        '',
        '',
        'if __name__ == "__main__":',
        '    tests = [',
        '        # (method, identifier, expected_tier, expected_paywalled)',
        '        ("domain",      "gulfnews.com",         4, False),',
        '        ("domain",      "thenational.ae",        4, True),',
        '        ("domain",      "wam.ae",                5, False),',
        '        ("domain",      "business.gulfnews.com", 4, False),',
        '        ("source_name", "Gulf News",             4, False),',
        '        ("source_name", "The National",          4, True),',
        '        ("source_name", "Al Khaleej (UAE)",      4, False),',
        '        ("source_name", "CompletelyUnknown XYZ", 1, False),',
        '        ("domain",      "unknowndomain123.com",  1, False),',
        '    ]',
        '    print("Running self-test...")',
        '    passed = failed = 0',
        '    for method, identifier, exp_tier, exp_pw in tests:',
        '        p         = get_profile(**{method: identifier})',
        '        tier_ok   = p["credibility_tier"] == exp_tier',
        '        pw_ok     = p["is_paywalled"] == exp_pw',
        '        ok        = tier_ok and pw_ok',
        '        passed   += int(ok)',
        '        failed   += int(not ok)',
        r"        symbol    = '✓' if ok else '✗'",
        "        print(f'  {symbol} {method:12} \"{identifier}\"')",
        '        if not tier_ok:',
        "            print(f'      tier: expected {exp_tier}, got {p[\"credibility_tier\"]}')",
        '        if not pw_ok:',
        "            print(f'      paywalled: expected {exp_pw}, got {p[\"is_paywalled\"]}')",
        '    print(f"\\n  {passed}/{passed + failed} tests passed")',
        '    if failed == 0:',
        '        print("  Registry is ready for pipeline use.")',
        '    else:',
        '        print("  Fix failures before using in pipeline.")',
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PY.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  [SAVED] {REGISTRY_PY}  ({n} domains baked in)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — OUTPUT: registry_summary.txt
# ═════════════════════════════════════════════════════════════════════════════

def write_summary(
    profiles: dict,
    generated_at: str,
    summary: dict,
    applied_overrides: dict,
) -> None:
    run_dt = generated_at[:19].replace("T", " ")
    td     = summary["tier_distribution"]
    n      = summary["total_domains"]
    SEP    = "═" * 59
    DIV    = "─" * 46

    # High value free: Tier 3+, free, 50+ articles
    hv_free = sorted(
        [p for p in profiles.values()
         if p["credibility_tier"] >= 3 and p["paywall_status"] == "free"
         and p["article_count"] >= 50],
        key=lambda p: (-p["article_count"], p["domain"]),
    )

    # High value paywalled: Tier 3+, paywalled
    hv_pay = sorted(
        [p for p in profiles.values()
         if p["credibility_tier"] >= 3 and p["is_paywalled"]],
        key=lambda p: (-p["article_count"], p["domain"]),
    )

    lines = [
        SEP,
        "  UNIFIED REGISTRY SUMMARY",
        f"  Generated: {run_dt}",
        SEP,
        "",
        f"  TOTAL DOMAINS: {n}",
        "",
        "  CREDIBILITY DISTRIBUTION",
        "  " + DIV[:25],
        f"  Tier 5 (UAE Govt)      : {td[5]:>4} domains  — primary truth sources",
        f"  Tier 4 (Major UAE)     : {td[4]:>4} domains  — core newsletter sources",
        f"  Tier 3 (Regional)      : {td[3]:>4} domains  — credible supporting sources",
        f"  Tier 2 (Intl Press)    : {td[2]:>4} domains  — international context",
        f"  Tier 1 (Unknown/Other) : {td[1]:>4} domains  — default, lowest priority",
        "",
        "  Note: All domains NOT in Tier 2+ are Tier 1 by default.",
        "  Tier 1 domains can still appear in the newsletter if no better",
        "  source covers the same story.",
        "",
        "  PAYWALL STATUS",
        "  " + DIV[:14],
        f"  Free      : {summary['free']:>4} domains",
        f"  Paywalled : {summary['paywalled']:>4} domains",
        f"  Metered   : {summary['metered']:>4} domains",
        "",
        "  HIGH VALUE FREE  (Tier 3+, free, 50+ articles)",
        "  " + DIV,
    ]

    for p in hv_free:
        lines.append(
            f"  {p['domain']:<35} Tier {p['credibility_tier']}  free      "
            f"{p['article_count']:>4} articles"
        )
    if not hv_free:
        lines.append("  (none meeting criteria)")

    lines += [
        "",
        "  HIGH VALUE PAYWALLED  (Tier 3+, paywalled/metered)",
        "  " + DIV,
    ]
    for p in hv_pay:
        lines.append(
            f"  {p['domain']:<35} Tier {p['credibility_tier']}  "
            f"{p['paywall_status']:<10} {p['article_count']:>4} articles"
        )
    if not hv_pay:
        lines.append("  (none)")

    lines += [
        "",
        "  OVERRIDES APPLIED",
        "  " + DIV[:17],
        f"  {len(applied_overrides)} domains had inference errors corrected:",
        "",
    ]
    for domain, (old_t, new_t) in sorted(applied_overrides.items()):
        note = OVERRIDE_NOTES.get(domain, "")
        lines.append(f"  {domain:<35} Tier {old_t}→{new_t}  ({note})")

    lines += ["", SEP]

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [SAVED] {SUMMARY_TXT}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run() -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"""
{'='*55}
  UNIFIED REGISTRY GENERATOR
{'='*55}
  Input  : {INPUT_FILE}
  Output : {OUTPUT_DIR}/
{'='*55}
""")

    profiles          = load_registry()
    applied_overrides = apply_overrides(profiles)
    source_to_domain  = build_source_to_domain(profiles)
    summary           = compute_summary(profiles)

    td = summary["tier_distribution"]
    print(
        f"  Tiers → "
        f"5:{td[5]}  4:{td[4]}  3:{td[3]}  2:{td[2]}  1:{td[1]}  |  "
        f"Free:{summary['free']}  Paywalled:{summary['paywalled']}  "
        f"Metered:{summary['metered']}"
    )
    print()

    write_registry_py(profiles, source_to_domain, generated_at, summary)
    write_summary(profiles, generated_at, summary, applied_overrides)

    print(f"""
{'='*55}
  COMPLETE
{'='*55}
  Domains baked in  : {summary['total_domains']}
  Source mappings   : {len(source_to_domain)}
  Overrides applied : {len(applied_overrides)}

  Run self-test:
    python url_resolution/unified_registry.py
{'='*55}
""")


if __name__ == "__main__":
    run()
