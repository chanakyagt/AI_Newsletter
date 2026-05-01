"""
UAE AI News — Single Source Script
────────────────────────────────────────────────────────────────────────────
Fetches AI-related news from Google News RSS targeting UAE executives,
government officials, business leaders and decision makers.

Covers: government policy, business transformation, investment, executive
voices, startups, sector impact, national vision, global deals.

Output:
  news_output/
    english/   → one JSON per theme
    arabic/    → one JSON per theme
    combined/  → all_articles.json + metadata.json

Run:
  python uae_ai_news.py
────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import re
import time
import hashlib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG
# ═════════════════════════════════════════════════════════════════════════════

NEWS_DATE               = os.environ.get("NEWS_DATE", datetime.now().strftime("%Y-%m-%d"))
OUTPUT_DIR              = os.path.join("news_output", NEWS_DATE)
REQUEST_DELAY_SECONDS   = 1.2
REQUEST_TIMEOUT_SECONDS = 15
MAX_SUMMARY_CHARS       = 600
USER_AGENT              = "Mozilla/5.0 (compatible; UAEAINewsBot/1.0)"

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ENGLISH KEYWORD SETS
# Each tuple = one RSS query (keywords joined with spaces in URL)
# ═════════════════════════════════════════════════════════════════════════════

ENGLISH_KEYWORD_SETS = {

    # ── GOVERNMENT & POLICY ──────────────────────────────────────────────────
    "gov_policy": [
        ("AI", "UAE", "government", "policy"),
        ("artificial intelligence", "UAE", "regulation", "2025"),
        ("AI", "UAE", "ministry", "strategy"),
        ("AI", "UAE", "federal", "initiative"),
        ("UAE", "AI", "law", "framework"),
        ("UAE", "National AI Strategy", "update"),
        ("UAE", "AI", "smart government"),
        ("UAE", "AI", "public sector", "transformation"),
        ("Mohammed bin Rashid", "AI", "vision"),
        ("UAE AI Office", "announcement"),
        ("Omar Al Olama", "AI", "UAE"),
        ("UAE", "AI", "digital economy", "policy"),
    ],

    # ── BUSINESS TRANSFORMATION ───────────────────────────────────────────────
    "biz_transformation": [
        ("AI", "UAE", "business", "transformation"),
        ("AI", "UAE", "enterprise", "adoption"),
        ("AI", "Dubai", "company", "implementation"),
        ("AI", "Abu Dhabi", "corporate", "strategy"),
        ("generative AI", "UAE", "business"),
        ("AI", "UAE", "operational", "efficiency"),
        ("AI", "UAE", "ROI", "results"),
        ("AI", "UAE", "digital transformation"),
        ("AI", "UAE", "automation", "enterprise"),
        ("AI", "UAE", "productivity", "firm"),
    ],

    # ── INVESTMENT & ECONOMY ──────────────────────────────────────────────────
    "investment_economy": [
        ("AI", "UAE", "investment", "billion"),
        ("AI", "UAE", "funding", "venture"),
        ("AI", "Abu Dhabi", "investment", "fund"),
        ("AI", "Dubai", "economic", "growth"),
        ("AI", "UAE", "GDP", "impact"),
        ("AI", "UAE", "ADGM", "tech"),
        ("AI", "UAE", "DIFC", "fintech"),
        ("artificial intelligence", "UAE", "economy", "2025"),
        ("AI", "UAE", "sovereign wealth", "technology"),
        ("Mubadala", "AI", "investment"),
        ("ADQ", "AI", "technology"),
        ("G42", "AI", "UAE"),
    ],

    # ── EXECUTIVE VOICES ─────────────────────────────────────────────────────
    "executive_voices": [
        ("UAE", "CEO", "AI", "strategy"),
        ("UAE", "executive", "artificial intelligence", "vision"),
        ("UAE", "business leader", "AI", "future"),
        ("UAE", "chairman", "AI", "transformation"),
        ("UAE", "minister", "AI", "announcement"),
        ("GITEX", "AI", "UAE", "leader"),
        ("World Economic Forum", "UAE", "AI"),
        ("UAE", "AI", "summit", "keynote"),
        ("UAE", "AI", "board", "decision"),
        ("Dubai AI Festival", "executive"),
    ],

    # ── STARTUPS & INNOVATION ─────────────────────────────────────────────────
    "startups_innovation": [
        ("AI", "startup", "UAE", "funding"),
        ("AI", "startup", "Dubai", "launch"),
        ("AI", "startup", "Abu Dhabi", "innovation"),
        ("UAE", "AI", "unicorn", "tech"),
        ("Hub71", "AI", "startup"),
        ("Dubai Future Foundation", "AI"),
        ("UAE", "AI", "accelerator", "2025"),
        ("AI", "UAE", "Series A", "raise"),
        ("AI", "UAE", "deep tech", "startup"),
        ("UAE", "AI", "innovation", "breakthrough"),
    ],

    # ── SECTOR IMPACT ─────────────────────────────────────────────────────────
    "sector_impact": [
        ("AI", "UAE", "banking", "finance"),
        ("AI", "UAE", "healthcare", "hospital"),
        ("AI", "UAE", "energy", "ADNOC"),
        ("AI", "UAE", "real estate", "PropTech"),
        ("AI", "UAE", "logistics", "supply chain"),
        ("AI", "UAE", "retail", "ecommerce"),
        ("AI", "UAE", "telecom", "etisalat"),
        ("AI", "UAE", "education", "EdTech"),
        ("AI", "UAE", "construction", "smart city"),
        ("AI", "UAE", "agriculture", "food security"),
    ],

    # ── NATIONAL VISION ───────────────────────────────────────────────────────
    "national_vision": [
        ("UAE Vision 2031", "AI"),
        ("UAE Centennial 2071", "artificial intelligence"),
        ("Smart Dubai", "AI", "initiative"),
        ("Abu Dhabi", "AI", "national program"),
        ("UAE", "AI", "sovereign", "capability"),
        ("UAE", "AI", "data center", "infrastructure"),
        ("UAE", "AI", "cloud", "Microsoft", "Google"),
        ("UAE", "AI", "chip", "NVIDIA", "partnership"),
        ("UAE", "AI", "research", "university"),
    ],

    # ── GLOBAL DEALS INVOLVING UAE ────────────────────────────────────────────
    "global_deals_uae": [
        ("UAE", "AI", "partnership", "US"),
        ("UAE", "AI", "deal", "billion"),
        ("UAE", "OpenAI", "partnership"),
        ("UAE", "Microsoft", "AI", "investment"),
        ("UAE", "Google", "AI", "cloud"),
        ("UAE", "Amazon", "AWS", "AI"),
        ("UAE", "Anthropic", "AI"),
        ("UAE", "Meta", "AI", "deal"),
        ("UAE", "AI", "China", "technology"),
        ("UAE", "AI", "trade", "agreement"),
    ],
}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ARABIC KEYWORD SETS
# Targets Arabic-language UAE newspapers (Al Ittihad, Al Bayan, Al Khaleej)
# ═════════════════════════════════════════════════════════════════════════════

ARABIC_KEYWORD_SETS = {

    # ── GOVERNMENT & POLICY (حكومة وسياسة) ───────────────────────────────────
    "gov_policy_ar": [
        ("الذكاء الاصطناعي", "الإمارات", "حكومة", "سياسة"),
        ("الذكاء الاصطناعي", "الإمارات", "استراتيجية", "وطنية"),
        ("الذكاء الاصطناعي", "وزارة", "الإمارات", "قرار"),
        ("الذكاء الاصطناعي", "الإمارات", "تشريع", "قانون"),
        ("الإمارات", "الذكاء الاصطناعي", "مبادرة", "حكومية"),
        ("عمر العلماء", "الذكاء الاصطناعي"),
        ("محمد بن راشد", "الذكاء الاصطناعي"),
        ("الإمارات", "الذكاء الاصطناعي", "رؤية", "2031"),
        ("الإمارات", "الذكاء الاصطناعي", "القطاع العام"),
        ("مكتب الذكاء الاصطناعي", "الإمارات"),
    ],

    # ── BUSINESS TRANSFORMATION (تحول الأعمال) ───────────────────────────────
    "biz_transformation_ar": [
        ("الذكاء الاصطناعي", "الإمارات", "أعمال", "تحول"),
        ("الذكاء الاصطناعي", "دبي", "شركة", "تطبيق"),
        ("الذكاء الاصطناعي", "أبوظبي", "مؤسسة", "استراتيجية"),
        ("الذكاء الاصطناعي", "الإمارات", "إنتاجية", "نمو"),
        ("الذكاء الاصطناعي", "الإمارات", "التحول الرقمي"),
        ("الذكاء الاصطناعي", "الإمارات", "أتمتة", "كفاءة"),
        ("الذكاء الاصطناعي التوليدي", "الإمارات", "أعمال"),
        ("الإمارات", "الذكاء الاصطناعي", "نتائج", "شركة"),
    ],

    # ── INVESTMENT & ECONOMY (استثمار واقتصاد) ───────────────────────────────
    "investment_economy_ar": [
        ("الذكاء الاصطناعي", "الإمارات", "استثمار", "مليار"),
        ("الذكاء الاصطناعي", "أبوظبي", "صندوق", "تمويل"),
        ("الذكاء الاصطناعي", "الإمارات", "اقتصاد", "نمو"),
        ("مبادلة", "الذكاء الاصطناعي", "استثمار"),
        ("G42", "الذكاء الاصطناعي", "الإمارات"),
        ("الذكاء الاصطناعي", "الإمارات", "ناتج محلي"),
        ("الإمارات", "الذكاء الاصطناعي", "تمويل", "مشروع"),
    ],

    # ── EXECUTIVE VOICES (قيادات وتصريحات) ───────────────────────────────────
    "executive_voices_ar": [
        ("الإمارات", "رئيس تنفيذي", "الذكاء الاصطناعي"),
        ("الإمارات", "وزير", "الذكاء الاصطناعي", "تصريح"),
        ("الإمارات", "قيادة", "الذكاء الاصطناعي", "مستقبل"),
        ("قمة", "الذكاء الاصطناعي", "الإمارات", "خطاب"),
        ("جيتكس", "الذكاء الاصطناعي", "الإمارات"),
        ("الإمارات", "رجال الأعمال", "الذكاء الاصطناعي"),
        ("الإمارات", "الذكاء الاصطناعي", "رؤساء", "شركات"),
    ],

    # ── STARTUPS & INNOVATION (شركات ناشئة وابتكار) ──────────────────────────
    "startups_innovation_ar": [
        ("الذكاء الاصطناعي", "شركة ناشئة", "الإمارات", "تمويل"),
        ("الذكاء الاصطناعي", "ابتكار", "الإمارات", "إطلاق"),
        ("الذكاء الاصطناعي", "دبي", "ريادة الأعمال"),
        ("هاب71", "الذكاء الاصطناعي", "ناشئة"),
        ("الإمارات", "الذكاء الاصطناعي", "تقنية", "جديدة"),
        ("مؤسسة دبي للمستقبل", "الذكاء الاصطناعي"),
        ("الإمارات", "الذكاء الاصطناعي", "براءة اختراع", "ابتكار"),
    ],

    # ── SECTOR IMPACT (قطاعات) ────────────────────────────────────────────────
    "sector_impact_ar": [
        ("الذكاء الاصطناعي", "الإمارات", "بنوك", "مالية"),
        ("الذكاء الاصطناعي", "الإمارات", "صحة", "مستشفى"),
        ("الذكاء الاصطناعي", "أدنوك", "طاقة"),
        ("الذكاء الاصطناعي", "الإمارات", "عقارات"),
        ("الذكاء الاصطناعي", "الإمارات", "لوجستيات", "نقل"),
        ("الذكاء الاصطناعي", "الإمارات", "تعليم", "مدارس"),
        ("الذكاء الاصطناعي", "الإمارات", "مدينة ذكية"),
    ],

    # ── GLOBAL DEALS IN ARABIC (صفقات عالمية) ────────────────────────────────
    "global_deals_ar": [
        ("الإمارات", "الذكاء الاصطناعي", "شراكة", "أمريكا"),
        ("الإمارات", "الذكاء الاصطناعي", "اتفاقية", "مليار"),
        ("الإمارات", "مايكروسوفت", "الذكاء الاصطناعي"),
        ("الإمارات", "جوجل", "الذكاء الاصطناعي", "سحابة"),
        ("الإمارات", "إنفيديا", "الذكاء الاصطناعي", "رقائق"),
        ("الإمارات", "أوبن إيه آي", "شراكة"),
    ],
}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RSS URL BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_rss_url(keywords: tuple, language: str = "en", country: str = "AE", past_week: bool = True) -> str:
    """
    Build a Google News RSS search URL from a tuple of keywords.

    Args:
        keywords   : Tuple of keyword strings (English or Arabic)
        language   : 'en' for English feeds, 'ar' for Arabic feeds
        country    : ISO country code, default 'AE' for UAE
        past_week  : Append &as_qdr=w to restrict to last 7 days
    """
    query = " ".join(keywords)
    encoded_query = urllib.parse.quote(query)

    if language == "ar":
        hl, gl, ceid = "ar", "AE", "AE:ar"
    else:
        hl, gl, ceid = "en", country, f"{country}:en"

    url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded_query}"
        f"&hl={hl}&gl={gl}&ceid={ceid}"
    )
    if past_week:
        url += "&as_qdr=w"

    return url

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ARTICLE SCHEMA
# ═════════════════════════════════════════════════════════════════════════════

def make_article(title, source, published_date, url, summary, language, theme, keywords) -> dict:
    """
    Returns one structured article dict.
    article_id = MD5 of URL — used for deduplication.
    relevance_score / keep / editorial_note are left null for LLM filter stage.
    """
    return {
        "article_id":       hashlib.md5(url.encode()).hexdigest(),
        "title":            title.strip(),
        "source":           source.strip(),
        "published_date":   published_date,
        "url":              url.strip(),
        "summary":          summary.strip(),
        "language":         language,
        "theme":            theme,
        "matched_keywords": list(keywords),
        "fetched_at":       datetime.now(timezone.utc).isoformat(),
        # ── Filled by LLM filter stage ────────────────────────────────────────
        "relevance_score":  None,   # 0.0 – 1.0
        "keep":             None,   # true | false
        "editorial_note":   None,   # short reason why kept or dropped
    }

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FETCH & PARSE
# ═════════════════════════════════════════════════════════════════════════════

def fetch_rss(url: str) -> str | None:
    """Fetch raw RSS XML. Returns None on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    [FETCH ERROR] {e}")
        return None


def clean_html(text: str) -> str:
    """Strip HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&amp;",  "&",  text)
    text = re.sub(r"&lt;",   "<",  text)
    text = re.sub(r"&gt;",   ">",  text)
    text = re.sub(r"&quot;", '"',  text)
    text = re.sub(r"&#39;",  "'",  text)
    text = re.sub(r"\s+",    " ",  text)
    return text.strip()


def parse_rss(xml_text: str, theme: str, keywords: tuple, language: str) -> list[dict]:
    """Parse Google News RSS XML into a list of structured article dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"    [PARSE ERROR] {e}")
        return articles

    channel = root.find("channel")
    if channel is None:
        return articles

    for item in channel.findall("item"):
        title_el = item.find("title")
        title = title_el.text if title_el is not None and title_el.text else "No Title"

        link_el = item.find("link")
        url = link_el.text if link_el is not None and link_el.text else ""
        if not url:
            continue

        source_el = item.find("source")
        source = source_el.text if source_el is not None and source_el.text else "Unknown Source"

        pub_el = item.find("pubDate")
        if pub_el is not None and pub_el.text:
            try:
                published_date = parsedate_to_datetime(pub_el.text).isoformat()
            except Exception:
                published_date = pub_el.text
        else:
            published_date = "Unknown"

        desc_el = item.find("description")
        if desc_el is not None and desc_el.text:
            summary = clean_html(desc_el.text)[:MAX_SUMMARY_CHARS]
        else:
            summary = title[:MAX_SUMMARY_CHARS]

        articles.append(make_article(
            title=title,
            source=source,
            published_date=published_date,
            url=url,
            summary=summary,
            language=language,
            theme=theme,
            keywords=keywords,
        ))

    return articles

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DEDUPLICATION
# ═════════════════════════════════════════════════════════════════════════════

def deduplicate(articles: list[dict]) -> list[dict]:
    """
    Deduplicate by article_id (URL hash).
    If the same article was matched by multiple keyword sets,
    keep the first occurrence and merge the keyword lists.
    """
    seen = {}
    for article in articles:
        aid = article["article_id"]
        if aid not in seen:
            seen[aid] = article
        else:
            for kw in article["matched_keywords"]:
                if kw not in seen[aid]["matched_keywords"]:
                    seen[aid]["matched_keywords"].append(kw)
    return list(seen.values())

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FILE OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def ensure_dirs():
    for subdir in ["english", "arabic", "combined"]:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)


def write_json(filepath: str, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    count = len(data) if isinstance(data, list) else 1
    print(f"  [SAVED] {filepath}  ({count} records)")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run():
    ensure_dirs()
    all_articles_global = []

    run_stats = {
        "run_started_at":              datetime.now(timezone.utc).isoformat(),
        "run_completed_at":            None,
        "total_feeds_fetched":         0,
        "total_feeds_failed":          0,
        "total_articles_raw":          0,
        "total_articles_deduplicated": 0,
        "themes_english":              {},
        "themes_arabic":               {},
    }

    def process_themes(keyword_sets: dict, language: str, subfolder: str, stats_key: str):
        """Shared logic for processing both English and Arabic theme groups."""
        for theme, keyword_list in keyword_sets.items():
            print(f"\n  [THEME] {theme}")
            theme_articles = []

            for kw_tuple in keyword_list:
                lang_code = "ar" if language == "Arabic" else "en"
                url = build_rss_url(kw_tuple, language=lang_code, past_week=True)
                print(f"    Fetching: {' + '.join(kw_tuple)}")

                xml_text = fetch_rss(url)
                if xml_text:
                    parsed = parse_rss(xml_text, theme=theme, keywords=kw_tuple, language=language)
                    theme_articles.extend(parsed)
                    run_stats["total_feeds_fetched"] += 1
                    print(f"      → {len(parsed)} articles")
                else:
                    run_stats["total_feeds_failed"] += 1

                time.sleep(REQUEST_DELAY_SECONDS)

            deduped = deduplicate(theme_articles)
            run_stats["total_articles_raw"] += len(theme_articles)
            run_stats[stats_key][theme]      = len(deduped)
            all_articles_global.extend(deduped)

            out_path = os.path.join(OUTPUT_DIR, subfolder, f"{theme}.json")
            write_json(out_path, deduped)

    # ── English ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  FETCHING ENGLISH FEEDS")
    print("="*65)
    process_themes(ENGLISH_KEYWORD_SETS, "English", "english", "themes_english")

    # ── Arabic ────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  FETCHING ARABIC FEEDS")
    print("="*65)
    process_themes(ARABIC_KEYWORD_SETS, "Arabic", "arabic", "themes_arabic")

    # ── Combined output ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("WRITING COMBINED OUTPUT")
    print("="*65)

    final = deduplicate(all_articles_global)

    def safe_sort_key(a):
        try:
            return datetime.fromisoformat(a["published_date"])
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    final.sort(key=safe_sort_key, reverse=True)

    run_stats["total_articles_deduplicated"] = len(final)
    run_stats["run_completed_at"]            = datetime.now(timezone.utc).isoformat()

    write_json(os.path.join(OUTPUT_DIR, "combined", "all_articles.json"), final)
    write_json(os.path.join(OUTPUT_DIR, "combined", "metadata.json"), run_stats)

    print(f"""
{'='*65}
  RUN COMPLETE
{'='*65}
  Feeds fetched      : {run_stats['total_feeds_fetched']}
  Feeds failed       : {run_stats['total_feeds_failed']}
  Raw articles       : {run_stats['total_articles_raw']}
  After dedup        : {run_stats['total_articles_deduplicated']}
  Output directory   : {os.path.abspath(OUTPUT_DIR)}/
{'='*65}
""")


if __name__ == "__main__":
    run()
