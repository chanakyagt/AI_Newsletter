"""
build_unified_registry.py — Unified Source Credibility + Paywall Registry Builder
UAE AI Newsletter Pipeline

Reads url_resolution/domain_registry.json (1006 domains, paywall already classified)
and merges it with hardcoded credibility tier knowledge to produce:

  url_resolution/unified_registry.json   — full machine-readable registry
  url_resolution/unified_registry.py     — importable pipeline module
  url_resolution/registry_report.txt     — human-readable summary

Run:
  python build_unified_registry.py

No external dependencies — standard library only.
"""

import json
import re
import collections
from datetime import datetime, timezone
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

INPUT_FILE    = Path("url_resolution") / "domain_registry.json"
OUTPUT_DIR    = Path("url_resolution")
REGISTRY_JSON = OUTPUT_DIR / "unified_registry.json"
REGISTRY_PY   = OUTPUT_DIR / "unified_registry.py"
REPORT_TXT    = OUTPUT_DIR / "registry_report.txt"

# Default tier_reason text for hardcoded domains (keyed by tier)
TIER_REASONS = {
    5: "UAE government or official body",
    4: "Major UAE daily or regional heavyweight",
    3: "Regional credible outlet",
    2: "International tech/business press",
    1: "Known source — explicitly assigned tier 1",
}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HARDCODED CREDIBILITY TIERS
# Add new domains here and rerun to update the registry.
# ═════════════════════════════════════════════════════════════════════════════

KNOWN_DOMAIN_TIERS = {
    # ── Tier 5: UAE Government and Official Bodies ─────────────────────────
    "wam.ae":                    {"tier": 5, "source_name": "WAM — UAE State News Agency"},
    "uaecabinet.ae":             {"tier": 5, "source_name": "UAE Cabinet"},
    "government.ae":             {"tier": 5, "source_name": "UAE Government Portal"},
    "mediaoffice.ae":            {"tier": 5, "source_name": "UAE Government Media Office"},
    "dubaimediaoffice.gov.ae":   {"tier": 5, "source_name": "Dubai Media Office"},
    "adgm.com":                  {"tier": 5, "source_name": "Abu Dhabi Global Market"},
    "cbuae.gov.ae":              {"tier": 5, "source_name": "Central Bank UAE"},
    "tdra.gov.ae":               {"tier": 5, "source_name": "TDRA UAE"},
    "uaespace.gov.ae":           {"tier": 5, "source_name": "UAE Space Agency"},

    # ── Tier 4: Major UAE English and Arabic Dailies ───────────────────────
    "gulfnews.com":              {"tier": 4, "source_name": "Gulf News"},
    "khaleejtimes.com":          {"tier": 4, "source_name": "Khaleej Times"},
    "thenational.ae":            {"tier": 4, "source_name": "The National"},
    "zawya.com":                 {"tier": 4, "source_name": "Zawya"},
    "arabianbusiness.com":       {"tier": 4, "source_name": "Arabian Business"},
    "albayan.ae":                {"tier": 4, "source_name": "Al Bayan (UAE)"},
    "alkhaleej.ae":              {"tier": 4, "source_name": "Al Khaleej (UAE)"},
    "alittihad.ae":              {"tier": 4, "source_name": "Al Ittihad (UAE)"},
    "emaratalyoum.com":          {"tier": 4, "source_name": "Emarat Al Youm (UAE)"},
    "alroeya.com":               {"tier": 4, "source_name": "Al Roeya (UAE)"},
    "gulfbusiness.com":          {"tier": 4, "source_name": "Gulf Business"},
    "agbi.com":                  {"tier": 4, "source_name": "AGBI"},
    "menabytes.com":             {"tier": 4, "source_name": "MENAbytes"},

    # ── Tier 3: Regional Credible Outlets ─────────────────────────────────
    "alarabiya.net":             {"tier": 3, "source_name": "Al Arabiya"},
    "skynewsarabia.com":         {"tier": 3, "source_name": "Sky News Arabia"},
    "aljazeera.com":             {"tier": 3, "source_name": "Al Jazeera"},
    "arabnews.com":              {"tier": 3, "source_name": "Arab News"},
    "english.aawsat.com":        {"tier": 3, "source_name": "Asharq Al-Awsat"},
    "forbesmiddleeast.com":      {"tier": 3, "source_name": "Forbes Middle East"},
    "cnnarabic.com":             {"tier": 3, "source_name": "CNN Arabic"},
    "bbcarabic.com":             {"tier": 3, "source_name": "BBC Arabic"},
    "reuters.com":               {"tier": 3, "source_name": "Reuters"},
    "bloomberg.com":             {"tier": 3, "source_name": "Bloomberg"},
    "ft.com":                    {"tier": 3, "source_name": "Financial Times"},
    "bbc.com":                   {"tier": 3, "source_name": "BBC"},
    "bbc.co.uk":                 {"tier": 3, "source_name": "BBC"},
    "cnn.com":                   {"tier": 3, "source_name": "CNN"},
    "apnews.com":                {"tier": 3, "source_name": "Associated Press"},
    "alahram.org.eg":            {"tier": 3, "source_name": "Al-Ahram (Egypt)"},

    # ── Tier 2: International Tech and Business Press ──────────────────────
    "techcrunch.com":            {"tier": 2, "source_name": "TechCrunch"},
    "wired.com":                 {"tier": 2, "source_name": "Wired"},
    "theverge.com":              {"tier": 2, "source_name": "The Verge"},
    "venturebeat.com":           {"tier": 2, "source_name": "VentureBeat"},
    "technologyreview.com":      {"tier": 2, "source_name": "MIT Technology Review"},
    "wsj.com":                   {"tier": 2, "source_name": "Wall Street Journal"},
    "economist.com":             {"tier": 2, "source_name": "The Economist"},
    "businessinsider.com":       {"tier": 2, "source_name": "Business Insider"},
    "forbes.com":                {"tier": 2, "source_name": "Forbes"},
    "fortune.com":               {"tier": 2, "source_name": "Fortune"},
    "hbr.org":                   {"tier": 2, "source_name": "Harvard Business Review"},
    "zdnet.com":                 {"tier": 2, "source_name": "ZDNet"},
    "tahawultech.com":           {"tier": 2, "source_name": "Tahawul Tech"},
    "computermiddleeast.com":    {"tier": 2, "source_name": "Computer Middle East"},
    "itpro.me":                  {"tier": 2, "source_name": "IT Pro ME"},
}

# Auto-generated reverse map: source_name → domain (used by build_source_lookups)
SOURCE_NAME_TO_DOMAIN = {v["source_name"]: k for k, v in KNOWN_DOMAIN_TIERS.items()}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NORMALISATION
# ═════════════════════════════════════════════════════════════════════════════

def normalise(name: str) -> str:
    """Lowercase, strip, remove regional suffixes, remove punctuation, collapse spaces."""
    name = name.lower().strip()
    name = name.replace("(uae)", "").replace("(egypt)", "")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TIER INFERENCE (for domains not in KNOWN_DOMAIN_TIERS)
# ═════════════════════════════════════════════════════════════════════════════

_QUALITY_SIGNALS = [
    "reuters", "bloomberg", "bbc", "cnn", "apnews",
    "aljazeera", "alarabiya", "skynews",
    "economist", "techcrunch", "wired", "verge",
    "wsj", "nytimes", "washingtonpost", "guardian",
]

def infer_tier(domain: str, article_count: int, paywall_status: str) -> tuple:
    """
    Returns (tier: int, reason: str). First matching rule wins.
    paywall_status is passed for potential future rules — current rules ignore it.
    Credibility and paywall are independent: never award Tier 3+ for being paywalled.
    """
    # Rule 1: UAE government TLD
    if domain.endswith(".gov.ae"):
        return 5, "UAE government domain (.gov.ae)"

    # Rule 2: Other government TLDs
    if any(domain.endswith(tld) for tld in (".gov", ".gov.uk", ".gov.au", ".mil")):
        return 3, "Government domain"

    # Rule 3: Quality-signal substring in domain
    for signal in _QUALITY_SIGNALS:
        if signal in domain:
            return 2, f"Quality signal '{signal}' in domain"

    # Rule 4: UAE domain (.ae) — high article count signals established source
    if domain.endswith(".ae") and article_count >= 100:
        return 3, "UAE domain (.ae) with high article count"

    # Rule 5: UAE domain (.ae) — moderate article count
    if domain.endswith(".ae") and article_count >= 20:
        return 2, "UAE domain (.ae) with moderate article count"

    # Rule 6: Any UAE domain (.ae) — low article count
    if domain.endswith(".ae"):
        return 1, "UAE domain (.ae) — low article count"

    # Rule 7: High global article count suggests established source
    if article_count >= 200:
        return 2, "High article count (200+) — established source"

    if article_count >= 50:
        return 1, "Moderate article count (50+)"

    # Rule 8: Default
    return 1, "Unknown source — default tier"

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — INPUT LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_domain_registry():
    if not INPUT_FILE.exists():
        print(f"\n[ERROR] Input file not found: {INPUT_FILE}")
        print("  Run build_domain_registry.py first.\n")
        raise SystemExit(1)
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    entries = data.get("domains", [])
    print(f"  Loaded {len(entries)} domain entries from {INPUT_FILE}")
    return entries

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PROFILE BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_profile(entry: dict) -> dict:
    """Build a complete unified profile for one domain entry."""
    domain         = entry["domain"]
    paywall_status = entry.get("paywall_status", "unknown")
    article_count  = entry.get("article_count_this_run", 0)
    confidence     = entry.get("confidence", "low")
    is_pw          = paywall_status in ("paywalled", "metered")

    if domain in KNOWN_DOMAIN_TIERS:
        info        = KNOWN_DOMAIN_TIERS[domain]
        tier        = info["tier"]
        source_name = info["source_name"]
        tier_source = "hardcoded"
        tier_reason = TIER_REASONS[tier]
    else:
        tier, tier_reason = infer_tier(domain, article_count, paywall_status)
        source_name = domain   # domain is its own display name for unknowns
        tier_source = "inferred"

    return {
        "domain":           domain,
        "source_name":      source_name,
        "credibility_tier": tier,
        "tier_source":      tier_source,
        "tier_reason":      tier_reason,
        "paywall_status":   paywall_status,
        "is_paywalled":     is_pw,
        "article_count":    article_count,
        "confidence":       confidence,
    }


def build_all_profiles(entries: list) -> dict:
    """Return dict keyed by domain."""
    profiles = {}
    for entry in entries:
        p = build_profile(entry)
        profiles[p["domain"]] = p
    return profiles

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SOURCE NAME LOOKUP BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_source_lookups(profiles: dict) -> tuple:
    """
    Returns (source_to_domain, normalised_source_to_domain).

    Primary entries: source_name from KNOWN_DOMAIN_TIERS → domain
    Secondary entries: domain → domain (for unknowns with no explicit source name)
    Normalised: apply normalise() to all source names for fuzzy matching
    """
    source_to_domain = {}

    # Primary: hardcoded source_name → domain
    for domain, info in KNOWN_DOMAIN_TIERS.items():
        source_to_domain[info["source_name"]] = domain

    # Secondary: use domain itself as source name for all unmapped domains
    already_mapped = set(source_to_domain.values())
    for domain in profiles:
        if domain not in already_mapped:
            source_to_domain[domain] = domain

    # Normalised variant
    normalised = {}
    for src_name, domain in source_to_domain.items():
        norm = normalise(src_name)
        if norm and norm not in normalised:  # first occurrence wins
            normalised[norm] = domain

    return source_to_domain, normalised

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — OUTPUT: unified_registry.json
# ═════════════════════════════════════════════════════════════════════════════

def write_registry_json(profiles: dict, source_to_domain: dict, generated_at: str) -> dict:
    by_tier = collections.Counter(p["credibility_tier"] for p in profiles.values())
    by_src  = collections.Counter(p["tier_source"] for p in profiles.values())

    payload = {
        "generated_at":       generated_at,
        "total_domains":      len(profiles),
        "hardcoded_domains":  by_src["hardcoded"],
        "inferred_domains":   by_src["inferred"],
        "tier_distribution": {
            f"tier_{t}": by_tier[t] for t in (5, 4, 3, 2, 1)
        },
        "domains": {
            p["domain"]: {
                "source_name":      p["source_name"],
                "credibility_tier": p["credibility_tier"],
                "tier_source":      p["tier_source"],
                "tier_reason":      p["tier_reason"],
                "paywall_status":   p["paywall_status"],
                "is_paywalled":     p["is_paywalled"],
                "article_count":    p["article_count"],
                "confidence":       p["confidence"],
            }
            for p in sorted(profiles.values(), key=lambda x: (-x["article_count"], x["domain"]))
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [SAVED] {REGISTRY_JSON}  ({len(profiles)} domains)")
    return payload

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — OUTPUT: unified_registry.py  (importable module)
# ═════════════════════════════════════════════════════════════════════════════

def write_registry_py(
    profiles: dict,
    source_to_domain: dict,
    norm_source_to_domain: dict,
    generated_at: str,
    summary: dict,
):
    run_dt   = generated_at[:19].replace("T", " ")
    n        = summary["total_domains"]
    n_hard   = summary["hardcoded_domains"]
    n_inf    = summary["inferred_domains"]
    tier_d   = summary["tier_distribution"]

    out = []

    # ── Header ───────────────────────────────────────────────────────────────
    out += [
        "# " + "═" * 63,
        "# UNIFIED SOURCE REGISTRY",
        "# Auto-generated by build_unified_registry.py",
        f"# Generated    : {run_dt}",
        f"# Total domains: {n}  |  Hardcoded: {n_hard}  |  Inferred: {n_inf}",
        "#",
        "# Supports two lookup methods:",
        '#   Phase 1 (before Firecrawl): get_profile(source_name="Gulf News")',
        '#   Phase 2 (after Firecrawl):  get_profile(domain="gulfnews.com")',
        "#",
        "# DO NOT EDIT MANUALLY — regenerate by running build_unified_registry.py",
        "# " + "═" * 63,
        "",
        "import re",
        "import logging",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "",
    ]

    # ── DOMAIN_REGISTRY ───────────────────────────────────────────────────────
    out.append("# ── Primary registry keyed by domain ─────────────────────────────────")
    out.append("DOMAIN_REGISTRY = {")

    tier_section_comments = {
        5: "    # ── Tier 5: UAE Government ──────────────────────────────────────────",
        4: "    # ── Tier 4: Major UAE Dailies ─────────────────────────────────────",
        3: "    # ── Tier 3: Regional Credible ─────────────────────────────────────",
        2: "    # ── Tier 2: International Press ───────────────────────────────────",
        1: "    # ── Tier 1: Unknown / Low-count ───────────────────────────────────",
    }
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
            out.append(tier_section_comments[t])
            current_tier = t
        is_pw   = "True" if p["is_paywalled"] else "False"
        src_str = json.dumps(p["source_name"], ensure_ascii=False)
        out.append(
            f'    {json.dumps(p["domain"])}: '
            f'{{"source_name": {src_str}, '
            f'"credibility_tier": {t}, '
            f'"is_paywalled": {is_pw}, '
            f'"paywall_status": "{p["paywall_status"]}"}},'
        )

    out += ["}", "", ""]

    # ── SOURCE_TO_DOMAIN ──────────────────────────────────────────────────────
    out.append("# ── Source name to domain bridge ─────────────────────────────────────")
    out.append("SOURCE_TO_DOMAIN = {")
    for src, dom in sorted(source_to_domain.items()):
        out.append(f"    {json.dumps(src, ensure_ascii=False)}: {json.dumps(dom)},")
    out += ["}", "", ""]

    # ── NORMALISED_SOURCE_TO_DOMAIN ────────────────────────────────────────────
    out.append("# ── Normalised source name lookup (handles variations) ────────────────")
    out.append("NORMALISED_SOURCE_TO_DOMAIN = {")
    for norm, dom in sorted(norm_source_to_domain.items()):
        out.append(f"    {json.dumps(norm, ensure_ascii=False)}: {json.dumps(dom)},")
    out += ["}", "", ""]

    # ── Frozensets ────────────────────────────────────────────────────────────
    out += [
        "# ── Pre-built frozensets for O(1) membership testing ─────────────────",
        'FREE_DOMAINS      = frozenset(k for k, v in DOMAIN_REGISTRY.items() if v["paywall_status"] == "free")',
        'PAYWALLED_DOMAINS = frozenset(k for k, v in DOMAIN_REGISTRY.items() if v["paywall_status"] == "paywalled")',
        'METERED_DOMAINS   = frozenset(k for k, v in DOMAIN_REGISTRY.items() if v["paywall_status"] == "metered")',
        'TIER_5_DOMAINS    = frozenset(k for k, v in DOMAIN_REGISTRY.items() if v["credibility_tier"] == 5)',
        'TIER_4_DOMAINS    = frozenset(k for k, v in DOMAIN_REGISTRY.items() if v["credibility_tier"] >= 4)',
        "",
        "",
        "# " + "═" * 63,
        "# PUBLIC API",
        "# " + "═" * 63,
        "",
    ]

    # ── Functions (embedded as literal lines) ─────────────────────────────────
    out += [
        'def resolve_subdomain(domain: str) -> str:',
        '    """',
        '    Strips subdomains to find a registry match.',
        '    business.thenational.ae  →  thenational.ae',
        '    tech.gulfnews.com        →  gulfnews.com',
        '    Returns matched registry domain, or original if no match found.',
        '    """',
        '    d = domain.lower().strip()',
        '    if d in DOMAIN_REGISTRY:',
        '        return d',
        '    parts = d.split(".")',
        '    for i in range(1, len(parts) - 1):',
        '        candidate = ".".join(parts[i:])',
        '        if candidate in DOMAIN_REGISTRY:',
        '            return candidate',
        '    return d',
        '',
        '',
        'def _normalise(name: str) -> str:',
        '    """Lowercase, strip, remove (UAE)/(Egypt) suffixes, remove punctuation."""',
        '    name = name.lower().strip()',
        '    name = name.replace("(uae)", "").replace("(egypt)", "")',
        r'    name = re.sub(r"[^\w\s]", "", name)',
        r'    name = re.sub(r"\s+", " ", name).strip()',
        '    return name',
        '',
        '',
        'def get_profile(source_name: str = None, domain: str = None) -> dict:',
        '    """',
        '    Main lookup function. Works in both pipeline phases.',
        '',
        '    Phase 1 (before Firecrawl): get_profile(source_name="Gulf News")',
        '    Phase 2 (after Firecrawl):  get_profile(domain="gulfnews.com")',
        '',
        '    Returns a dict with keys:',
        '        source_name, domain, credibility_tier, is_paywalled,',
        '        paywall_status, lookup_method',
        '',
        '    Never raises — always returns a valid dict.',
        '    Unknown sources get tier 1 and unknown paywall status.',
        '    """',
        '    def _default(method="default"):',
        '        return {',
        '            "source_name":      source_name or domain or "unknown",',
        '            "domain":           domain,',
        '            "credibility_tier": 1,',
        '            "is_paywalled":     False,',
        '            "paywall_status":   "unknown",',
        '            "lookup_method":    method,',
        '        }',
        '',
        '    def _hit(entry, d, method):',
        '        return {',
        '            "source_name":      entry["source_name"],',
        '            "domain":           d,',
        '            "credibility_tier": entry["credibility_tier"],',
        '            "is_paywalled":     entry["is_paywalled"],',
        '            "paywall_status":   entry["paywall_status"],',
        '            "lookup_method":    method,',
        '        }',
        '',
        '    # ── Phase 2: domain lookup ─────────────────────────────────────',
        '    if domain:',
        '        d = resolve_subdomain(domain)',
        '        if d in DOMAIN_REGISTRY:',
        '            method = "domain" if d == domain.lower().strip() else "subdomain"',
        '            return _hit(DOMAIN_REGISTRY[d], d, method)',
        '        logger.debug("Domain not in unified registry: %s", domain)',
        '        return _default("default")',
        '',
        '    # ── Phase 1: source name lookup ────────────────────────────────',
        '    if source_name:',
        '        # Direct match',
        '        if source_name in SOURCE_TO_DOMAIN:',
        '            d = SOURCE_TO_DOMAIN[source_name]',
        '            if d in DOMAIN_REGISTRY:',
        '                return _hit(DOMAIN_REGISTRY[d], d, "source_name")',
        '        # Normalised / fuzzy match',
        '        norm = _normalise(source_name)',
        '        if norm in NORMALISED_SOURCE_TO_DOMAIN:',
        '            d = NORMALISED_SOURCE_TO_DOMAIN[norm]',
        '            if d in DOMAIN_REGISTRY:',
        '                return _hit(DOMAIN_REGISTRY[d], d, "normalised")',
        '        logger.debug("Source name not in unified registry: %s", source_name)',
        '        return _default("default")',
        '',
        '    return _default("default")',
        '',
        '',
        'def get_credibility(source_name: str = None, domain: str = None) -> int:',
        '    """Returns credibility tier 1-5. Works with either identifier."""',
        '    return get_profile(source_name=source_name, domain=domain)["credibility_tier"]',
        '',
        '',
        'def is_paywalled(source_name: str = None, domain: str = None) -> bool:',
        '    """Returns True if paywalled or metered. Works with either identifier."""',
        '    return get_profile(source_name=source_name, domain=domain)["is_paywalled"]',
        '',
        '',
        'def is_free(source_name: str = None, domain: str = None) -> bool:',
        '    """Returns True if confirmed free. Works with either identifier."""',
        '    return get_profile(source_name=source_name, domain=domain)["paywall_status"] == "free"',
    ]

    REGISTRY_PY.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  [SAVED] {REGISTRY_PY}  (importable module)")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — OUTPUT: registry_report.txt
# ═════════════════════════════════════════════════════════════════════════════

def write_registry_report(profiles: dict, source_to_domain: dict, generated_at: str, summary: dict):
    run_dt = generated_at[:19].replace("T", " ")
    n      = max(summary["total_domains"], 1)
    td     = summary["tier_distribution"]

    def pct(x):
        return f"{x / n * 100:.0f}%"

    SEP = "═" * 59
    DIV = "─" * 50

    by_status = collections.Counter(p["paywall_status"] for p in profiles.values())

    # High-value free (tier 3+, free, sorted by article count)
    hv_free = sorted(
        [p for p in profiles.values() if p["credibility_tier"] >= 3 and p["paywall_status"] == "free"],
        key=lambda p: (-p["article_count"], p["domain"]),
    )

    # High-value paywalled (tier 3+, paywalled or metered)
    hv_pay = sorted(
        [p for p in profiles.values() if p["credibility_tier"] >= 3 and p["is_paywalled"]],
        key=lambda p: (-p["article_count"], p["domain"]),
    )

    # Inferred at tier 2+ (worth reviewing)
    inferred_high = sorted(
        [p for p in profiles.values() if p["tier_source"] == "inferred" and p["credibility_tier"] >= 2],
        key=lambda p: (-p["credibility_tier"], -p["article_count"], p["domain"]),
    )

    known_src_count = len(KNOWN_DOMAIN_TIERS)
    unknown_src_count = summary["total_domains"] - known_src_count

    lines = [
        SEP,
        "  UNIFIED REGISTRY REPORT",
        f"  Generated : {run_dt}",
        f"  Source    : {INPUT_FILE}",
        SEP,
        "",
        "  CREDIBILITY TIER DISTRIBUTION",
        "  " + DIV[:30],
        f"  Tier 5 (UAE Govt)     : {td['tier_5']:>4} domains",
        f"  Tier 4 (Major UAE)    : {td['tier_4']:>4} domains",
        f"  Tier 3 (Regional)     : {td['tier_3']:>4} domains",
        f"  Tier 2 (Intl Press)   : {td['tier_2']:>4} domains",
        f"  Tier 1 (Unknown)      : {td['tier_1']:>4} domains",
        "",
        "  PAYWALL STATUS",
        "  " + DIV[:14],
        f"  Free                  : {by_status['free']:>4} domains",
        f"  Paywalled             : {by_status['paywalled']:>4} domains",
        f"  Metered               : {by_status['metered']:>4} domains",
        f"  Unknown               : {by_status['unknown']:>4} domains",
        "",
        "  HIGH VALUE FREE SOURCES  (Tier 3+ and free, by article count)",
        "  " + DIV,
    ]
    for p in hv_free:
        lines.append(
            f"  {p['domain']:<35} Tier {p['credibility_tier']}  free    "
            f"{p['article_count']:>4} articles"
        )

    lines += [
        "",
        "  HIGH VALUE PAYWALLED  (Tier 3+ and paywalled/metered)",
        "  " + DIV,
    ]
    for p in hv_pay:
        lines.append(
            f"  {p['domain']:<35} Tier {p['credibility_tier']}  {p['paywall_status']:<10} "
            f"{p['article_count']:>4} articles"
        )

    lines += [
        "",
        "  SOURCE NAME COVERAGE",
        "  " + DIV[:20],
        f"  Known source name mappings    : {known_src_count}",
        f"  Unknown (domain used as name) : {unknown_src_count}",
        "",
        "  INFERRED TIERS 2+  (review these — auto-assigned, not hardcoded)",
        "  " + DIV,
    ]
    if inferred_high:
        for p in inferred_high:
            lines.append(
                f"  {p['domain']:<35} Tier {p['credibility_tier']}  "
                f"[{p['tier_reason']}]"
            )
    else:
        lines.append("  None — all Tier 2+ domains are hardcoded.")

    lines += ["", SEP]

    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [SAVED] {REPORT_TXT}")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run():
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"""
{'='*55}
  UNIFIED REGISTRY BUILDER
{'='*55}
  Input  : {INPUT_FILE}
  Output : {OUTPUT_DIR}/
{'='*55}
""")

    entries  = load_domain_registry()
    profiles = build_all_profiles(entries)
    source_to_domain, norm_source_to_domain = build_source_lookups(profiles)

    by_tier = collections.Counter(p["credibility_tier"] for p in profiles.values())
    by_src  = collections.Counter(p["tier_source"] for p in profiles.values())

    summary = {
        "total_domains":     len(profiles),
        "hardcoded_domains": by_src["hardcoded"],
        "inferred_domains":  by_src["inferred"],
        "tier_distribution": {f"tier_{t}": by_tier[t] for t in (5, 4, 3, 2, 1)},
    }

    print()
    write_registry_json(profiles, source_to_domain, generated_at)
    write_registry_py(profiles, source_to_domain, norm_source_to_domain, generated_at, summary)
    write_registry_report(profiles, source_to_domain, generated_at, summary)

    td = summary["tier_distribution"]
    print(f"""
{'='*55}
  COMPLETE
{'='*55}
  Total domains  : {summary['total_domains']}
  Hardcoded tiers: {summary['hardcoded_domains']}
  Inferred tiers : {summary['inferred_domains']}

  Tier 5 (UAE Govt)   : {td['tier_5']}
  Tier 4 (Major UAE)  : {td['tier_4']}
  Tier 3 (Regional)   : {td['tier_3']}
  Tier 2 (Intl Press) : {td['tier_2']}
  Tier 1 (Unknown)    : {td['tier_1']}

  Import in pipeline scripts:
    from url_resolution.unified_registry import get_profile, is_paywalled, is_free
{'='*55}
""")


if __name__ == "__main__":
    run()
