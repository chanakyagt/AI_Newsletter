#!/usr/bin/env python3
"""
nabdh_newsletter_v2.py
NABDH (نبض) — UAE AI Intelligence Newsletter Generator
Reads: news_output/keypoints/keypoints.json
Writes: news_output/newsletter/nabdh_YYYY-MM-DD.html
"""

import re, base64, json, os, time, sys, argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

INPUT_FILE   = Path("news_output/keypoints/keypoints.json")
TRACKER_FILE = Path("quarterly_tracker.json")
OUTPUT_DIR   = Path("news_output/newsletter")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("Deepseek_API_Key_1", "")
MODEL    = "deepseek-v4-pro"   # DeepSeek V4 Pro — confirmed API string
FALLBACK = "deepseek-chat"     # routes to latest V3 if V4 unavailable
BASE_URL = "https://api.deepseek.com"

TODAY    = datetime.now().strftime("%d %B %Y")
YEAR     = datetime.now().strftime("%Y")
QUARTER  = f"Q{(datetime.now().month-1)//3+1} {YEAR}"
DATE_STR = datetime.now().strftime("%Y-%m-%d")

URGENCY_CFG = {
    "IMMEDIATE": {"color":"#C0392B","bg":"rgba(192,57,43,0.07)","border":"rgba(192,57,43,0.25)","ar":"عاجل"},
    "THIS WEEK": {"color":"#0A00FE","bg":"rgba(10,0,254,0.05)", "border":"rgba(10,0,254,0.18)", "ar":"هذا الأسبوع"},
    "WATCH":     {"color":"#2D6A4F","bg":"rgba(45,106,79,0.06)","border":"rgba(45,106,79,0.22)","ar":"للمتابعة"},
}

TRACKER_STATUS = {
    "EMERGING": {"label":"EMERGING","ar":"ناشئ", "color":"#E85D04"},
    "ACTIVE":   {"label":"ONGOING", "ar":"جارٍ", "color":"#0A00FE"},
    "STABLE":   {"label":"STABLE",  "ar":"مستقر","color":"#2D6A4F"},
}

NEGATIVE_WORDS = ["breach","hack","scandal","collapse","layoff","shutdown","crisis","fraud","fine","arrest","attack","failure"]

# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def clean(text: str) -> str:
    """Strip all markdown before inserting LLM output into HTML."""
    if not text:
        return ""
    # 1. Remove section/paragraph labels LLMs love to add
    text = re.sub(r'(?im)^\s*\*{0,2}\s*(paragraph|section)\s*\d+\s*[:\-–—]?\s*\*{0,2}\s*$', '', text)
    text = re.sub(r'(?im)^\s*\*{0,2}\s*(first|second|third|opening|closing|intro|summary)\s*(paragraph)?\s*[:\-–—]?\s*\*{0,2}\s*$', '', text)
    # 2. Strip bold/italic PAIRS first (so content inside is kept)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*\*(.+?)\*\*',     r'\1', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__',         r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*(.+?)\*',         r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_(.+?)_',           r'\1', text, flags=re.DOTALL)
    # 3. Strip bullet markers at line start (lone * or - or •)
    text = re.sub(r'(?m)^\s*[\*\-•]\s+', '', text)
    # 4. Kill any remaining stray asterisks and underscores
    text = text.replace('*', '').replace('_', ' ').replace('`', '')
    # 5. Strip markdown headers
    text = re.sub(r'(?m)^\s*#{1,6}\s+', '', text)
    # 6. Collapse excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def safe_truncate(text: str, limit: int) -> str:
    if not text or len(text) <= limit:
        return text or ""
    cut = text[:limit]
    space = cut.rfind(' ')
    return (cut[:space] if space > limit * 0.75 else cut) + "…"

def load_logo() -> str:
    """Load Waves logo as base64 data URI. Tries SVG first, then PNG."""
    for name in ["waveslogo-white.png","waves-logo.svg","waves_logo.svg","waves-logo.png","waves_logo.png","logo.svg","logo.png"]:
        p = Path(name)
        if p.exists():
            data = base64.b64encode(p.read_bytes()).decode()
            mime = "image/svg+xml" if name.endswith(".svg") else "image/png"
            print(f"  [LOGO] Loaded: {name}")
            return f"data:{mime};base64,{data}"
    print("  [LOGO] WARNING: No logo file found. Place waves-logo.svg in same directory.")
    return ""

_EDITION_COUNTER = OUTPUT_DIR / "edition_counter.json"

def get_edition_number() -> int:
    override = os.environ.get("NABDH_EDITION_OVERRIDE", "").strip()
    if override.isdigit():
        return int(override)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if _EDITION_COUNTER.exists():
        try:
            data = json.loads(_EDITION_COUNTER.read_text(encoding="utf-8"))
            return int(data.get("next_edition", 1))
        except Exception:
            pass
    return 1

def _increment_edition_number():
    current = get_edition_number()
    _EDITION_COUNTER.write_text(
        json.dumps({"next_edition": current + 1}, indent=2),
        encoding="utf-8"
    )

# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_keypoints():
    kps = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    filtered = []
    for kp in kps:
        hl   = kp.get("headline_reframe","").lower()
        opp  = kp.get("opportunity_or_threat","").upper()
        impl = kp.get("business_implication","").lower()
        neg  = any(w in hl for w in NEGATIVE_WORDS)
        if neg and opp.startswith("THREAT"):
            if not any(w in impl for w in ["opportunity","advantage","ahead","prepare","lead","position"]):
                continue
        filtered.append(kp)

    immediate = sorted([k for k in filtered if k.get("urgency_label")=="IMMEDIATE"],
                       key=lambda x: x.get("final_score",0), reverse=True)
    this_week = sorted([k for k in filtered if k.get("urgency_label")=="THIS WEEK"],
                       key=lambda x: x.get("final_score",0), reverse=True)
    watch     = sorted([k for k in filtered if k.get("urgency_label")=="WATCH"],
                       key=lambda x: x.get("final_score",0), reverse=True)
    return immediate + this_week, watch

def load_tracker():
    if not TRACKER_FILE.exists():
        return []
    data  = json.loads(TRACKER_FILE.read_text(encoding="utf-8"))
    items = [i for i in data.get("items",[]) if i.get("status") != "RESOLVED"]
    return sorted(items, key=lambda x: {"EMERGING":0,"ACTIVE":1,"STABLE":2}.get(x.get("status"),3))[:6]

# ═══════════════════════════════════════════════════════════
# LLM CALLS
# ═══════════════════════════════════════════════════════════

def generate_editorial(keypoints: list, client: OpenAI) -> str:
    top = keypoints[:5]
    signals = "\n".join([
        f"- {kp.get('headline_reframe', '')} | {kp.get('the_signal', '')}"
        for kp in top
    ])

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the editor of NABDH — UAE's most read AI intelligence brief for "
                            "ministers, sovereign fund directors, and C-suite executives.\n"
                            "Write with authority and precision. Every word must earn its place.\n"
                            "Plain text only. No markdown. No asterisks. No labels. No preamble."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
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
                    }
                ],
                temperature=0.75,
                max_tokens=250,
            )
            content = resp.choices[0].message.content
            if content and content.strip() and len(content.strip()) >= 120:
                print(f"  [EDITORIAL] {len(content)} chars received")
                print(f"  [EDITORIAL PREVIEW] {repr(content[:150])}")
                return clean(content)
            else:
                print(f"  [EDITORIAL] Attempt {attempt+1}: response too short or empty, retrying...")
                time.sleep(3)
        except Exception as e:
            print(f"  [EDITORIAL] Attempt {attempt+1} error: {e}")
            time.sleep(5)

    # Hard fallback — never leave editorial empty
    fallback = (
        "The UAE is not building an AI strategy. It is restructuring the state around one.\n\n"
        "The AI and Development Council now sits above every federal ministry. "
        "Microsoft's $15.2 billion commitment secures the compute layer. "
        "Executives should know: the procurement window for first-mover positioning closes faster than any published timeline suggests.\n\n"
        "The boards moving now are auditing AI readiness against government timelines, not market ones. "
        "The question every UAE boardroom must answer: which part of your strategy depends on infrastructure the sovereign will soon control?"
    )
    print("  [EDITORIAL] Using fallback content")
    return fallback


def generate_hooks(keypoints: list, client: OpenAI) -> dict:
    top4 = keypoints[:4]
    items = "\n".join([
        f"{i+1}. {clean(kp.get('headline_reframe',''))} | {clean(kp.get('business_implication',''))}"
        for i, kp in enumerate(top4)
    ])
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write hook lines for UAE's most senior executives. "
                    "Each hook: one sentence, under 16 words, factually grounded, psychologically compelling. "
                    "Use scarcity, insider knowledge, consequence framing, or pattern interruption. "
                    "No markdown. No labels. Return ONLY a valid JSON array: [{\"id\":1,\"hook\":\"...\"}]"
                )
            },
            {
                "role": "user",
                "content": f"Write one hook line for each of these {len(top4)} UAE AI stories:\n{items}"
            }
        ],
        temperature=0,
        max_tokens=300,
    )
    try:
        raw = re.sub(r'```json|```','', resp.choices[0].message.content.strip()).strip()
        return {h["id"]: h["hook"] for h in json.loads(raw)}
    except Exception:
        return {}


def generate_closing(keypoints: list, client: OpenAI) -> str:
    lead = keypoints[0] if keypoints else {}
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write the closing of NABDH newsletter. "
                    "Output ONLY clean prose. No markdown. No labels. No asterisks. "
                    "Tone: Emirati pride, forward ambition, quiet urgency."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Lead intelligence: {lead.get('headline_reframe','')}\n"
                    f"Signal: {lead.get('the_signal','')}\n\n"
                    "Write one closing paragraph, 70-85 words. "
                    "Make UAE's most powerful executives feel proud and responsible simultaneously. "
                    "End with one specific question for their next leadership meeting. "
                    "No generic calls to action. No subscription mentions. Just clean prose."
                )
            }
        ],
        temperature=0,
        max_tokens=200,
    )
    return clean(resp.choices[0].message.content.strip())

# ═══════════════════════════════════════════════════════════
# HTML COMPONENT BUILDERS
# ═══════════════════════════════════════════════════════════

def urgency_badge(urg: str) -> str:
    c = URGENCY_CFG.get(urg, URGENCY_CFG["THIS WEEK"])
    return (
        f'<span class="badge-urg" style="background:{c["bg"]};color:{c["color"]};border-color:{c["border"]};">'
        f'{urg} &nbsp;&middot;&nbsp; {c["ar"]}</span>'
    )

def cat_badge(cat: str) -> str:
    return f'<span class="badge-cat">{cat.replace("_"," ")}</span>'

def section_divider(en: str, ar: str) -> str:
    return f'''<div class="sec-div">
  <div class="sec-line"></div>
  <span class="sec-tag">{en} &nbsp;&middot;&nbsp; {ar}</span>
  <div class="sec-line"></div>
</div>'''

def build_lead(kp: dict, hook: str) -> str:
    url    = kp.get("citation_url","#")
    urg    = kp.get("urgency_label","THIS WEEK")
    cat    = kp.get("category_tag","")
    source = kp.get("source_citation","")
    facts  = [f for f in (kp.get("key_facts") or []) if f][:3]
    opp    = clean(kp.get("opportunity_or_threat",""))
    is_opp = opp.upper().startswith("OPPORTUNITY")
    oc, ob = ("#1a5c1e","rgba(102,235,110,0.25)") if is_opp else ("#7B241C","rgba(192,57,43,0.2)")
    oi     = "▲" if is_opp else "▼"

    facts_html = "".join([
        f'<div class="fact-row"><div class="fact-dot"></div>'
        f'<span class="fact-text">{clean(f)}</span></div>'
        for f in facts
    ])
    hook_html = f'<div class="lead-hook">"{clean(hook)}"</div>' if hook else ""

    return f'''<div class="lead-wrap">
  <div class="badges-row">{urgency_badge(urg)}{cat_badge(cat)}</div>
  {hook_html}
  <h2 class="lead-h"><a href="{url}" target="_blank" rel="noopener">{clean(kp.get("headline_reframe",""))}</a></h2>
  <div class="signal-box">
    <span class="signal-lbl">The Signal</span>
    <p class="signal-txt">{clean(kp.get("the_signal",""))}</p>
  </div>
  <div class="impl-box">
    <span class="impl-lbl">Board Action Required</span>
    <p class="impl-txt">{clean(kp.get("business_implication",""))}</p>
  </div>
  <div class="pq-box">
    <div class="pq-mark">"</div>
    <p class="pq-txt">{clean(kp.get("power_quote",""))}</p>
  </div>
  <div class="facts-wrap">
    <span class="facts-lbl">Key Intelligence</span>
    {facts_html}
  </div>
  <div class="opp-badge" style="background:rgba(0,0,0,0.04);color:{oc};border:1px solid {ob};">
    {oi} {safe_truncate(opp, 120)}
  </div>
  <div class="lead-foot">
    <a href="{url}" target="_blank" rel="noopener" class="read-more">Read full intelligence &rarr;</a>
    <span class="src-cred">&mdash; {source}</span>
  </div>
</div>'''

def build_featured_full(kp: dict, hook: str) -> str:
    url    = kp.get("citation_url","#")
    urg    = kp.get("urgency_label","THIS WEEK")
    cat    = kp.get("category_tag","")
    source = kp.get("source_citation","")
    hook_html = f'<div class="feat-hook">"{clean(hook)}"</div>' if hook else ""
    return f'''<div class="feat-full">
  <div class="badges-row">{urgency_badge(urg)}{cat_badge(cat)}</div>
  {hook_html}
  <h3 class="feat-h-lg"><a href="{url}" target="_blank" rel="noopener">{clean(kp.get("headline_reframe",""))}</a></h3>
  <p class="feat-sig">{clean(kp.get("the_signal",""))}</p>
  <div class="feat-impl">
    <strong style="color:#0A00FE;font-weight:600;">Board action:</strong>
    {safe_truncate(clean(kp.get("business_implication","")), 200)}
  </div>
  <div class="feat-foot">
    <a href="{url}" target="_blank" rel="noopener" class="read-more">Full brief &rarr;</a>
    <span class="src-cred">&mdash; {source}</span>
  </div>
</div>'''

def build_featured_half(kp: dict, hook: str) -> str:
    url    = kp.get("citation_url","#")
    urg    = kp.get("urgency_label","THIS WEEK")
    cat    = kp.get("category_tag","")
    source = kp.get("source_citation","")
    hook_html = f'<div class="feat-hook-sm">"{clean(hook)}"</div>' if hook else ""
    return f'''<div class="feat-half">
  <div class="badges-row">{urgency_badge(urg)}{cat_badge(cat)}</div>
  {hook_html}
  <h3 class="feat-h-sm"><a href="{url}" target="_blank" rel="noopener">{clean(kp.get("headline_reframe",""))}</a></h3>
  <p class="feat-sig-sm">{safe_truncate(clean(kp.get("the_signal","")), 160)}</p>
  <div class="feat-impl-sm">
    <strong style="color:#0A00FE;font-weight:600;">Action:</strong>
    {safe_truncate(clean(kp.get("business_implication","")), 140)}
  </div>
  <div class="feat-foot">
    <a href="{url}" target="_blank" rel="noopener" class="read-more">Full brief &rarr;</a>
    <span class="src-cred">&mdash; {source}</span>
  </div>
</div>'''

def build_brief(kp: dict) -> str:
    url    = kp.get("citation_url","#")
    urg    = kp.get("urgency_label","THIS WEEK")
    source = kp.get("source_citation","")
    color  = URGENCY_CFG.get(urg, URGENCY_CFG["THIS WEEK"])["color"]
    return f'''<div class="brief-item">
  <div class="brief-dot" style="background:{color};"></div>
  <div>
    <span class="brief-urg" style="color:{color};">{urg}</span>
    <a href="{url}" target="_blank" rel="noopener" class="brief-h">{clean(kp.get("headline_reframe",""))}</a>
    <p class="brief-sig">{safe_truncate(clean(kp.get("the_signal","")), 160)}</p>
    <span class="brief-src">&mdash; {source}</span>
  </div>
</div>'''

def build_tracker(tracker: list) -> str:
    header = f'''<div class="tracker-header">
  <span class="tracker-meta-lbl">Quarterly Tracker &mdash; {QUARTER} &nbsp;<span style="direction:rtl;color:rgba(10,0,254,0.35);">متابعة الربع</span></span>
  <h2 class="tracker-title">Structural Shifts to Monitor</h2>
  <p class="tracker-sub">These developments will not demand action this week &mdash; but they are reshaping the ground beneath this quarter. Board-level awareness required.</p>
</div>'''

    if not tracker:
        return f'''<div class="tracker-section">
  {header}
  <div class="tracker-placeholder">
    <span class="tracker-placeholder-ar">نبض</span>
    <h3 style="font-family:'Playfair Display',serif;font-size:18px;color:#271F5C;margin-bottom:8px;">Tracker Launches Next Edition</h3>
    <p style="font-size:13px;color:#626468;line-height:1.6;max-width:380px;margin:0 auto;">From Edition 2, this section monitors the slow-burn structural developments reshaping UAE&rsquo;s AI landscape &mdash; the signals that define the next 90 days.</p>
  </div>
</div>'''

    items_html = ""
    for i, t in enumerate(tracker, 1):
        cfg  = TRACKER_STATUS.get(t.get("status","ACTIVE"), TRACKER_STATUS["ACTIVE"])
        upd  = '<span class="tracker-updated">UPDATED</span>' if t.get("updated_this_week") else ""
        items_html += f'''<div class="tracker-item">
  <div class="tracker-num">{i:02d}</div>
  <div>
    <div class="tracker-status-row">
      <span class="tracker-status-lbl" style="color:{cfg['color']};">{cfg['label']}</span>
      <span class="tracker-status-ar">{cfg['ar']}</span>
      {upd}
    </div>
    <a href="{t.get('latest_url','#')}" target="_blank" rel="noopener" class="tracker-h">{t.get('theme','')}</a>
    <p class="tracker-why">{t.get('why_it_matters','')}</p>
    <p class="tracker-latest"><strong>Latest:</strong> {t.get('latest_headline','')}</p>
    <span class="tracker-src">&mdash; {t.get('latest_source','')}</span>
  </div>
</div>'''

    return f'<div class="tracker-section">{header}{items_html}</div>'

def build_citations(main_feed: list, tracker: list) -> str:
    try:
        from url_resolution.unified_registry import get_profile as _get_profile
        def _pw(source_name=""):
            return _get_profile(source_name=source_name).get("is_paywalled", False)
    except ImportError:
        def _pw(source_name=""):
            return False

    all_items = list(main_feed)
    for t in tracker:
        all_items.append({
            "headline_reframe": t.get("latest_headline", t.get("theme","")),
            "source_citation":  t.get("latest_source",""),
            "citation_url":     t.get("latest_url","#"),
            "published_date":   t.get("last_updated",""),
        })

    rows = ""
    for i, kp in enumerate(all_items, 1):
        src  = kp.get("source_citation","")
        url  = kp.get("citation_url","#")
        ttl  = safe_truncate(clean(kp.get("headline_reframe","")), 80)
        date = kp.get("published_date","")[:10]
        sub  = '<span class="sub-note">(subscriber)</span>' if _pw(source_name=src) else ""
        rows += f'''<div class="cit-item">
  <span class="cit-num">[{i}]</span>
  <strong class="cit-src">{src}</strong>{sub} &mdash;
  <a href="{url}" target="_blank" rel="noopener" class="cit-link">{ttl}</a>
  <span class="cit-date"> ({date})</span>
</div>'''
    return rows

# ═══════════════════════════════════════════════════════════
# MAIN HTML BUILDER
# ═══════════════════════════════════════════════════════════

def build_html(keypoints, watch_items, tracker, editorial, hooks, closing, edition_num, logo_uri) -> str:
    from collections import Counter

    def dedup(items):
        seen, out = [], []
        for kp in items:
            words = set(w for w in clean(kp.get("headline_reframe","")).lower().split() if len(w) > 4)
            if not any(len(words & s) / max(len(words), 1) > 0.55 for s in seen):
                out.append(kp)
                seen.append(words)
        return out

    main_feed = keypoints
    lead_kp   = main_feed[0] if main_feed else {}
    featured  = main_feed[1:4]
    briefs    = dedup(main_feed[4:12] + watch_items[:3])[:8]
    all_feed  = main_feed[:12]

    # Editorial rendering
    raw_paras = [p.strip() for p in re.split(r'\n\s*\n', editorial) if p.strip()]
    if len(raw_paras) <= 1:
        raw_paras = [p.strip() for p in editorial.split('\n') if p.strip()]
    if not raw_paras and editorial.strip():
        raw_paras = [editorial.strip()]

    editorial_html = ""
    if raw_paras:
        thesis = raw_paras[0]
        editorial_html += (
            f'<div class="ed-thesis-wrap">'
            f'<span class="ed-thesis-lbl">This Edition&rsquo;s Thesis</span>'
            f'<p class="ed-kicker">{thesis}</p>'
            f'</div><div class="ed-rule"></div>'
        )
        body_paras = raw_paras[1:] if len(raw_paras) > 1 else raw_paras
        for i, p in enumerate(body_paras):
            if i == 0:
                editorial_html += f'<p class="ed-p"><span class="ed-dropcap">{p[0]}</span>{p[1:]}</p>'
            else:
                editorial_html += f'<p class="ed-p" style="clear:left;">{p}</p>'
    else:
        print("  [WARN] Editorial returned empty — check LLM response above")
        editorial_html = '<p class="ed-p">Intelligence briefing compiled from this week\'s top UAE AI signals.</p>'

    # Featured layout: 1 full-width + 2-column half-width grid
    feat_html = ""
    if featured:
        feat_html += build_featured_full(featured[0], hooks.get(2,""))
        if len(featured) >= 2:
            left  = build_featured_half(featured[1], hooks.get(3,""))
            right = build_featured_half(featured[2], hooks.get(4,"")) if len(featured) >= 3 else '<div class="feat-half feat-empty"></div>'
            feat_html += f'<div class="feat-grid">{left}{right}</div>'

    briefs_html  = "".join([build_brief(kp) for kp in briefs])
    tracker_html = build_tracker(tracker)
    cit_html     = build_citations(all_feed, tracker)

    urg_counts = Counter(k.get("urgency_label","THIS WEEK") for k in all_feed)
    chips = "".join([
        f'<span class="urg-chip" style="color:{URGENCY_CFG.get(u,URGENCY_CFG["THIS WEEK"])["color"]};'
        f'border-color:{URGENCY_CFG.get(u,URGENCY_CFG["THIS WEEK"])["color"]}40;'
        f'background:{URGENCY_CFG.get(u,URGENCY_CFG["THIS WEEK"])["color"]}0D;">'
        f'{c} {u}</span>'
        for u, c in urg_counts.items()
    ])

    if logo_uri:
        logo_hdr = f'<a href="https://www.wavesad.com" target="_blank" rel="noopener" class="logo-link"><img src="{logo_uri}" alt="Waves AD" class="logo-img-hdr"/></a>'
        logo_ftr = f'<a href="https://www.wavesad.com" target="_blank" rel="noopener" class="logo-link-ftr"><img src="{logo_uri}" alt="Waves AD" class="logo-img-ftr"/></a>'
    else:
        logo_hdr = '<a href="https://www.wavesad.com" target="_blank" rel="noopener" class="logo-link"><span class="logo-fallback">waves</span></a>'
        logo_ftr = '<a href="https://www.wavesad.com" target="_blank" rel="noopener" class="logo-link-ftr"><span class="logo-fallback">waves</span></a>'

    hook_lead = clean(hooks.get(1,""))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<meta name="color-scheme" content="light"/>
<title>NABDH &mdash; UAE AI Intelligence Brief &middot; {TODAY}</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
/* RESET & BASE */
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Plus Jakarta Sans',system-ui,sans-serif;background:#E8E8F0;-webkit-font-smoothing:antialiased;color:#1A1828}}
a{{text-decoration:none;color:inherit}}
img{{display:block;max-width:100%}}

/* WRAPPER */
.nl{{max-width:700px;margin:0 auto;background:#fff;box-shadow:0 4px 60px rgba(10,0,254,0.1)}}

/* HEADER */
.hdr{{background:linear-gradient(135deg,#261D5F 0%,#18104a 45%,#0A00FE 100%);position:relative;overflow:hidden}}
.hdr-top{{display:flex;justify-content:space-between;align-items:center;padding:14px 40px;border-bottom:1px solid rgba(255,255,255,0.07)}}
.logo-img-hdr{{height:32px;width:auto}}
.logo-fallback{{font-family:'Plus Jakarta Sans',sans-serif;font-size:16px;font-weight:700;color:#fff}}
.hdr-url{{font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(255,255,255,0.35);letter-spacing:1px}}
.hdr-pulse{{position:absolute;right:0;top:50%;transform:translateY(-50%);opacity:0.05;pointer-events:none}}
.hdr-drop{{position:absolute;right:40px;bottom:-30px;width:110px;height:130px;background:radial-gradient(ellipse at 40% 30%,rgba(0,255,255,0.2) 0%,rgba(0,255,255,0) 70%);border-radius:50% 50% 50% 50% / 60% 60% 40% 40%;transform:rotate(-15deg);pointer-events:none}}
.hdr-main{{padding:36px 40px 44px;text-align:center;position:relative;z-index:2}}
.hdr-meta-top{{font-family:'JetBrains Mono',monospace;font-size:8px;color:rgba(0,255,255,0.6);letter-spacing:3px;text-transform:uppercase;margin-bottom:22px;display:block}}
.hdr-name{{font-family:'Playfair Display',serif;font-size:64px;font-weight:900;color:#fff;letter-spacing:-2px;line-height:0.92;display:block}}
.hdr-name-ar{{font-size:30px;color:rgba(0,255,255,0.8);display:block;margin-top:7px;letter-spacing:6px;font-weight:300;direction:rtl}}
.hdr-tagline{{font-family:'Plus Jakarta Sans',sans-serif;font-size:12px;font-weight:400;color:rgba(255,255,255,0.42);letter-spacing:1.5px;margin-top:14px;display:block}}
.hdr-rule{{width:48px;height:1px;background:linear-gradient(90deg,transparent,#00FFFF,transparent);margin:18px auto}}
.hdr-dateline{{font-family:'JetBrains Mono',monospace;font-size:10px;color:rgba(255,255,255,0.38);letter-spacing:0.5px}}

/* EDITION STRIP */
.ed-strip{{background:#271F5C;padding:9px 40px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}}
.ed-strip-lbl{{font-family:'JetBrains Mono',monospace;font-size:8px;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase}}
.ed-chips{{display:flex;gap:8px;flex-wrap:wrap}}
.urg-chip{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:1px;padding:3px 9px;border-radius:2px;text-transform:uppercase;border:1px solid}}

/* EDITORIAL */
.editorial{{padding:48px 40px 44px;border-bottom:3px solid #0A00FE;border-top:4px solid #00FFFF;background:#fff}}
.ed-dropcap{{font-family:'Playfair Display',serif;font-size:62px;font-weight:900;float:left;line-height:0.82;margin:5px 10px 0 0;color:#271F5C}}
.ed-p{{font-size:16px;line-height:1.82;color:#1A1828;margin-bottom:18px;font-weight:400}}
.ed-sig{{font-family:'JetBrains Mono',monospace;font-size:9px;color:#b9bcbe;text-align:right;margin-top:18px;padding-top:14px;border-top:1px solid #eee;clear:left}}
.ed-from-lbl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:3px;color:#0A00FE;text-transform:uppercase;display:block;margin-bottom:20px;opacity:0.55}}
.ed-thesis-wrap{{background:rgba(39,31,92,0.04);border-left:5px solid #00FFFF;padding:20px 24px;margin-bottom:0}}
.ed-thesis-lbl{{font-family:'JetBrains Mono',monospace;font-size:7px;letter-spacing:3px;color:rgba(10,0,254,0.5);text-transform:uppercase;display:block;margin-bottom:12px}}
.ed-kicker{{font-family:'Playfair Display',serif;font-size:24px;font-weight:700;font-style:italic;color:#271F5C;line-height:1.32;margin:0}}
.ed-rule{{height:1px;background:linear-gradient(90deg,rgba(10,0,254,0.15),rgba(10,0,254,0.04),transparent);margin:28px 0 32px;clear:left}}

/* SECTION DIVIDER */
.sec-div{{display:flex;align-items:center;gap:14px;padding:36px 40px 0}}
.sec-line{{flex:1;height:1px;background:#E8E8F2}}
.sec-tag{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:4px;color:#0A00FE;text-transform:uppercase;white-space:nowrap;font-weight:500}}

/* BADGES */
.badges-row{{display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-wrap:wrap}}
.badge-urg{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:1.5px;padding:4px 10px;border-radius:2px;text-transform:uppercase;font-weight:500;border:1px solid}}
.badge-cat{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:1px;padding:3px 9px;border:1px solid #ddd;color:#888;border-radius:2px;text-transform:uppercase}}

/* LEAD STORY */
.lead-wrap{{padding:28px 40px 44px;border-bottom:1px solid #eee}}
.lead-hook{{font-size:13px;font-weight:600;color:#0A00FE;font-style:italic;margin-bottom:14px;line-height:1.45;border-left:3px solid #00FFFF;padding-left:12px}}
.lead-h{{font-family:'Playfair Display',serif;font-size:34px;font-weight:900;color:#271F5C;line-height:1.12;margin-bottom:28px;letter-spacing:-0.5px}}
.lead-h a:hover{{color:#0A00FE}}
.signal-box{{background:#271F5C;border-left:4px solid #00FFFF;padding:20px 24px;margin-bottom:24px}}
.signal-lbl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:2px;color:rgba(0,255,255,0.65);text-transform:uppercase;display:block;margin-bottom:8px}}
.signal-txt{{color:rgba(255,255,255,0.88);font-size:14.5px;line-height:1.72}}
.impl-box{{background:rgba(10,0,254,0.04);border:1px solid rgba(10,0,254,0.1);border-radius:4px;padding:16px 20px;margin-bottom:20px}}
.impl-lbl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:2px;color:#0A00FE;text-transform:uppercase;display:block;margin-bottom:6px}}
.impl-txt{{font-size:13.5px;color:#2C2838;line-height:1.65;font-weight:500}}
.pq-box{{background:linear-gradient(135deg,#F8F6FF,#F0F0FF);border-left:3px solid #0A00FE;padding:18px 22px;margin-bottom:20px;position:relative}}
.pq-mark{{position:absolute;top:-8px;left:14px;font-family:'Playfair Display',serif;font-size:52px;color:#0A00FE;opacity:0.12;line-height:1}}
.pq-txt{{font-size:15px;font-style:italic;color:#271F5C;line-height:1.65;font-weight:500}}
.facts-wrap{{margin-bottom:20px}}
.facts-lbl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:2px;color:#888;text-transform:uppercase;display:block;margin-bottom:10px}}
.fact-row{{display:flex;gap:10px;margin-bottom:8px;align-items:flex-start}}
.fact-dot{{width:6px;height:6px;border-radius:50%;background:#00FFFF;margin-top:5px;flex-shrink:0;box-shadow:0 0 5px rgba(0,255,255,0.5)}}
.fact-text{{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:#2C2838;line-height:1.5}}
.opp-badge{{display:inline-flex;align-items:center;gap:6px;padding:5px 13px;border-radius:3px;font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:1px;text-transform:uppercase;font-weight:500;margin-bottom:20px}}
.lead-foot{{display:flex;justify-content:space-between;align-items:center;padding-top:16px;border-top:1px solid #eee}}
.read-more{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#0A00FE;letter-spacing:0.5px}}
.src-cred{{font-family:'JetBrains Mono',monospace;font-size:9px;color:#b9bcbe}}

/* FEATURED */
.feat-full{{padding:28px 40px;background:#FAFAF8;border-bottom:1px solid #eee}}
.feat-hook{{font-size:12px;font-weight:600;color:#0A00FE;font-style:italic;margin-bottom:10px;line-height:1.4;border-left:2px solid #00FFFF;padding-left:10px}}
.feat-h-lg{{font-family:'Playfair Display',serif;font-size:22px;font-weight:700;color:#271F5C;line-height:1.22;margin-bottom:12px}}
.feat-h-lg a:hover{{color:#0A00FE}}
.feat-sig{{font-size:13px;color:#626468;line-height:1.68;margin-bottom:12px}}
.feat-impl{{font-size:12.5px;color:#2C2838;line-height:1.55;padding:10px 14px;background:#fff;border-left:2px solid #0A00FE;margin-bottom:16px}}
.feat-foot{{display:flex;justify-content:space-between;align-items:center}}
.feat-grid{{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #eee}}
.feat-half{{padding:22px 28px;border-bottom:1px solid #eee}}
.feat-half:first-child{{border-right:1px solid #eee}}
.feat-empty{{background:#fff}}
.feat-hook-sm{{font-size:11px;font-weight:600;color:#0A00FE;font-style:italic;margin-bottom:8px;line-height:1.35;border-left:2px solid #00FFFF;padding-left:8px}}
.feat-h-sm{{font-family:'Playfair Display',serif;font-size:17px;font-weight:700;color:#271F5C;line-height:1.25;margin-bottom:10px}}
.feat-h-sm a:hover{{color:#0A00FE}}
.feat-sig-sm{{font-size:12.5px;color:#626468;line-height:1.6;margin-bottom:10px}}
.feat-impl-sm{{font-size:12px;color:#2C2838;line-height:1.5;padding:8px 12px;background:#FAFAF8;border-left:2px solid #0A00FE;margin-bottom:14px}}

/* BRIEFS */
.briefs-wrap{{background:#F7F7FB;padding:0 40px 32px}}
.brief-item{{padding:16px 0;border-bottom:1px solid #eee;display:grid;grid-template-columns:10px 1fr;gap:14px;align-items:start}}
.brief-dot{{width:8px;height:8px;border-radius:50%;margin-top:5px;flex-shrink:0}}
.brief-urg{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:1px;text-transform:uppercase;display:block;margin-bottom:4px}}
.brief-h{{font-size:14.5px;font-weight:700;color:#271F5C;margin-bottom:5px;line-height:1.35;display:block}}
.brief-h:hover{{color:#0A00FE}}
.brief-sig{{font-size:12px;color:#626468;line-height:1.58;margin-bottom:4px}}
.brief-src{{font-family:'JetBrains Mono',monospace;font-size:9px;color:#b9bcbe}}

/* TRACKER */
.tracker-section{{border-top:3px solid #0A00FE;background:#fff}}
.tracker-header{{background:linear-gradient(135deg,#F8F7FF,#F0F0FF);padding:32px 40px;border-bottom:1px solid rgba(10,0,254,0.1)}}
.tracker-meta-lbl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:3px;color:#0A00FE;text-transform:uppercase;display:block;margin-bottom:10px}}
.tracker-title{{font-family:'Playfair Display',serif;font-size:22px;font-weight:700;color:#271F5C;margin-bottom:8px}}
.tracker-sub{{font-size:13px;color:#626468;line-height:1.6;max-width:500px}}
.tracker-item{{display:grid;grid-template-columns:36px 1fr;gap:16px;padding:22px 40px;border-bottom:1px solid #eee;align-items:start}}
.tracker-num{{font-family:'JetBrains Mono',monospace;font-size:20px;color:#b9bcbe;line-height:1}}
.tracker-status-row{{display:flex;align-items:center;gap:6px;margin-bottom:8px;flex-wrap:wrap}}
.tracker-status-lbl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:2px;text-transform:uppercase;font-weight:600}}
.tracker-status-ar{{font-size:10px;color:#b9bcbe}}
.tracker-updated{{font-family:'JetBrains Mono',monospace;font-size:7px;background:#00FFFF;color:#271F5C;padding:2px 6px;border-radius:2px;letter-spacing:1px;font-weight:600;margin-left:6px}}
.tracker-h{{font-size:14.5px;font-weight:600;color:#271F5C;display:block;margin-bottom:6px;line-height:1.35}}
.tracker-h:hover{{color:#0A00FE}}
.tracker-why{{font-size:12.5px;color:#2C2838;line-height:1.6;margin-bottom:6px}}
.tracker-latest{{font-size:12px;color:#626468;line-height:1.5;margin-bottom:4px}}
.tracker-src{{font-family:'JetBrains Mono',monospace;font-size:9px;color:#b9bcbe}}
.tracker-placeholder{{padding:40px;text-align:center}}
.tracker-placeholder-ar{{font-size:36px;color:#e0e0ee;display:block;margin-bottom:12px}}

/* CLOSING */
.closing-wrap{{background:linear-gradient(135deg,#261D5F,#0A00FE);padding:52px 40px;text-align:center;position:relative;overflow:hidden}}
.closing-ar-bg{{position:absolute;right:16px;bottom:-24px;font-size:110px;color:rgba(255,255,255,0.04);pointer-events:none;direction:rtl}}
.closing-lbl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:3px;color:rgba(0,255,255,0.65);text-transform:uppercase;display:block;margin-bottom:20px}}
.closing-txt{{font-family:'Playfair Display',serif;font-size:20px;font-style:italic;color:rgba(255,255,255,0.9);line-height:1.68;max-width:500px;margin:0 auto 24px}}
.closing-url{{font-family:'JetBrains Mono',monospace;font-size:10px;color:rgba(0,255,255,0.5);letter-spacing:1px}}

/* CITATIONS */
.cit-wrap{{padding:32px 40px;background:#F7F7FB;border-top:1px solid #eee}}
.cit-hdr{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:3px;color:#888;text-transform:uppercase;margin-bottom:18px;display:block}}
.cit-item{{font-size:11px;color:#888;margin-bottom:8px;line-height:1.5;padding-left:26px;position:relative}}
.cit-num{{position:absolute;left:0;font-family:'JetBrains Mono',monospace;font-size:9px;color:#0A00FE}}
.cit-src{{font-weight:600;color:#2C2838}}
.sub-note{{font-family:'JetBrains Mono',monospace;font-size:8px;color:#b9bcbe;margin-left:3px}}
.cit-link{{color:#271F5C}}
.cit-link:hover{{color:#0A00FE;text-decoration:underline}}
.cit-date{{color:#b9bcbe}}

/* FOOTER */
.footer-wrap{{background:#271F5C;padding:36px 40px;text-align:center;position:relative;overflow:hidden}}
.footer-glow{{position:absolute;bottom:-40px;left:-40px;width:180px;height:180px;background:radial-gradient(circle,rgba(0,255,255,0.05) 0%,transparent 70%);border-radius:50%;pointer-events:none}}
.logo-link-ftr{{display:block;margin:0 auto 16px;width:fit-content}}
.logo-img-ftr{{height:28px;width:auto;margin:0 auto}}
.footer-name{{font-family:'Playfair Display',serif;font-size:17px;color:rgba(255,255,255,0.85);display:block;margin-bottom:3px}}
.footer-ar{{font-size:12px;color:rgba(0,255,255,0.38);direction:rtl;display:block;margin-bottom:18px}}
.footer-meta{{font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(255,255,255,0.22);line-height:2;letter-spacing:0.3px}}
.footer-meta a{{color:rgba(255,255,255,0.3)}}
.footer-meta a:hover{{color:rgba(0,255,255,0.5)}}

/* ═══════════════════════════════════════════════════
   MOBILE — Complete redesign below 600px
   Single column, larger touch targets, readable fonts
═══════════════════════════════════════════════════ */
@media(max-width:600px){{

  /* Header */
  .hdr-top{{padding:12px 20px}}
  .logo-img-hdr{{height:26px}}
  .hdr-main{{padding:24px 20px 32px}}
  .hdr-meta-top{{font-size:7px;letter-spacing:2px;margin-bottom:16px}}
  .hdr-name{{font-size:52px;letter-spacing:-1px}}
  .hdr-name-ar{{font-size:24px;letter-spacing:4px}}
  .hdr-tagline{{font-size:11px;letter-spacing:1px}}
  .hdr-rule{{margin:14px auto}}
  .hdr-dateline{{font-size:9px}}

  /* Edition strip */
  .ed-strip{{padding:8px 20px}}
  .ed-chips{{gap:6px}}
  .urg-chip{{font-size:7px;padding:3px 7px}}

  /* Editorial */
  .editorial{{padding:28px 20px 32px}}
  .ed-from-lbl{{font-size:7px;letter-spacing:2px;margin-bottom:16px}}
  .ed-thesis-wrap{{padding:16px 18px}}
  .ed-thesis-lbl{{font-size:7px;letter-spacing:2px;margin-bottom:10px}}
  .ed-kicker{{font-size:20px;line-height:1.35}}
  .ed-rule{{margin:20px 0 24px}}
  .ed-dropcap{{font-size:52px;margin-right:8px}}
  .ed-p{{font-size:16px;line-height:1.78;margin-bottom:16px}}
  .ed-sig{{font-size:8px;margin-top:14px;padding-top:12px}}

  /* Section divider */
  .sec-div{{padding:28px 20px 0}}
  .sec-tag{{font-size:7px;letter-spacing:2.5px}}

  /* Badges */
  .badge-urg{{font-size:7px;padding:4px 9px}}
  .badge-cat{{font-size:7px;padding:3px 7px}}
  .badges-row{{margin-bottom:12px;gap:6px}}

  /* Lead story */
  .lead-wrap{{padding:20px 20px 32px}}
  .lead-hook{{font-size:12px;margin-bottom:12px;padding-left:10px}}
  .lead-h{{font-size:26px;line-height:1.18;margin-bottom:20px;letter-spacing:-0.2px}}
  .signal-box{{padding:16px 18px;margin-bottom:18px}}
  .signal-lbl{{font-size:7px;margin-bottom:6px}}
  .signal-txt{{font-size:14px;line-height:1.68}}
  .impl-box{{padding:14px 16px;margin-bottom:16px}}
  .impl-lbl{{font-size:7px;margin-bottom:5px}}
  .impl-txt{{font-size:13px;line-height:1.6}}
  .pq-box{{padding:14px 18px;margin-bottom:16px}}
  .pq-mark{{font-size:44px;top:-6px;left:12px}}
  .pq-txt{{font-size:14px;line-height:1.6}}
  .facts-lbl{{font-size:7px;margin-bottom:8px}}
  .fact-text{{font-size:11px}}
  .opp-badge{{font-size:7px;padding:5px 10px}}
  .lead-foot{{flex-direction:column;align-items:flex-start;gap:8px;padding-top:14px}}
  .read-more{{font-size:11px;padding:10px 0;display:inline-block;min-height:44px;line-height:44px}}
  .src-cred{{font-size:9px}}

  /* Featured — single column on mobile */
  .feat-full{{padding:20px 20px}}
  .feat-hook{{font-size:11px;padding-left:8px}}
  .feat-h-lg{{font-size:19px}}
  .feat-sig{{font-size:12.5px}}
  .feat-impl{{padding:10px 12px}}
  .feat-grid{{grid-template-columns:1fr !important;border-top:none}}
  .feat-half{{padding:20px 20px;border-right:none !important}}
  .feat-empty{{display:none}}
  .feat-h-sm{{font-size:17px}}
  .feat-sig-sm{{font-size:12px}}
  .feat-impl-sm{{padding:8px 10px}}
  .feat-foot{{flex-direction:column;align-items:flex-start;gap:6px}}

  /* Briefs */
  .briefs-wrap{{padding:0 20px 24px}}
  .brief-item{{padding:18px 0;gap:12px}}
  .brief-h{{font-size:15px;line-height:1.38;min-height:44px;display:flex;align-items:center}}
  .brief-sig{{font-size:12px;line-height:1.6}}
  .brief-src{{font-size:9px}}
  .brief-urg{{font-size:7px;margin-bottom:5px}}

  /* Tracker */
  .tracker-header{{padding:24px 20px 20px}}
  .tracker-meta-lbl{{font-size:7px;letter-spacing:2px}}
  .tracker-title{{font-size:19px}}
  .tracker-sub{{font-size:12px}}
  .tracker-item{{padding:18px 20px;grid-template-columns:28px 1fr;gap:12px}}
  .tracker-num{{font-size:16px}}
  .tracker-status-lbl{{font-size:7px}}
  .tracker-h{{font-size:14px;min-height:44px;display:flex;align-items:center}}
  .tracker-why{{font-size:12px}}
  .tracker-latest{{font-size:11px}}
  .tracker-placeholder{{padding:28px 20px}}

  /* Closing */
  .closing-wrap{{padding:40px 20px}}
  .closing-txt{{font-size:17px;line-height:1.65}}
  .closing-lbl{{font-size:7px;margin-bottom:16px}}

  /* Citations */
  .cit-wrap{{padding:24px 20px}}
  .cit-hdr{{font-size:7px;margin-bottom:14px}}
  .cit-item{{font-size:11px;line-height:1.55;margin-bottom:10px}}

  /* Footer */
  .footer-wrap{{padding:28px 20px}}
  .logo-img-ftr{{height:24px}}
  .footer-name{{font-size:15px}}
  .footer-ar{{font-size:11px}}
  .footer-meta{{font-size:8px;line-height:1.9}}
}}
</style>
</head>
<body>
<div class="nl">

<!-- HEADER -->
<div class="hdr">
  <div class="hdr-top">
    {logo_hdr}
    <a href="https://www.wavesad.com" target="_blank" rel="noopener" class="hdr-url">wavesad.com</a>
  </div>
  <svg class="hdr-pulse" width="260" height="90" viewBox="0 0 260 90">
    <polyline points="0,45 22,45 33,10 44,80 55,22 66,68 77,45 110,45 121,18 132,72 143,35 154,55 165,45 260,45" stroke="#00FFFF" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  </svg>
  <div class="hdr-drop"></div>
  <div class="hdr-main">
    <span class="hdr-meta-top">UAE AI Intelligence Brief &nbsp;&middot;&nbsp; Edition {edition_num} &nbsp;&middot;&nbsp; {TODAY}</span>
    <span class="hdr-name">NABDH</span>
    <span class="hdr-name-ar">نبض</span>
    <span class="hdr-tagline">The Pulse of UAE&rsquo;s AI Economy</span>
    <div class="hdr-rule"></div>
    <span class="hdr-dateline">{TODAY} &nbsp;&middot;&nbsp; {QUARTER} &nbsp;&middot;&nbsp; {len(all_feed)} intelligence items</span>
  </div>
</div>

<!-- EDITION STRIP -->
<div class="ed-strip">
  <span class="ed-strip-lbl">This edition</span>
  <div class="ed-chips">{chips}</div>
</div>

<!-- EDITORIAL -->
<div class="editorial">
  <span class="ed-from-lbl">From the Editor</span>
  {editorial_html}
  <div class="ed-sig">&mdash; NABDH Editorial Team &nbsp;&middot;&nbsp; {TODAY}</div>
</div>

<!-- LEAD -->
{section_divider("Lead Intelligence", "الذكاء الأول")}
{build_lead(lead_kp, hook_lead)}

<!-- FEATURED -->
{section_divider("Featured Intelligence", "الذكاء المميز")}
{feat_html}

<!-- BRIEFS -->
{section_divider("Intelligence Briefs", "موجز الذكاء")}
<div class="briefs-wrap">
  {briefs_html}
</div>

<!-- TRACKER -->
{tracker_html}

<!-- CLOSING -->
<div class="closing-wrap">
  <div class="closing-ar-bg">نبض</div>
  <span class="closing-lbl">The Pulse &nbsp;&middot;&nbsp; النبض</span>
  <p class="closing-txt">{clean(closing)}</p>
  <a href="https://www.wavesad.com" target="_blank" rel="noopener" class="closing-url">wavesad.com</a>
</div>

<!-- CITATIONS -->
<div class="cit-wrap">
  <span class="cit-hdr">Sources &amp; Citations &nbsp;&middot;&nbsp; المصادر</span>
  {cit_html}
</div>

<!-- FOOTER -->
<div class="footer-wrap">
  <div class="footer-glow"></div>
  {logo_ftr}
  <span class="footer-name">NABDH &nbsp;&middot;&nbsp; UAE AI Intelligence Brief</span>
  <span class="footer-ar">نبض الاقتصاد الرقمي</span>
  <div class="footer-meta">
    Edition {edition_num} &nbsp;&middot;&nbsp; {TODAY} &nbsp;&middot;&nbsp; {QUARTER}<br/>
    Published by <a href="https://www.wavesad.com" target="_blank" rel="noopener">Waves AD</a> &nbsp;&middot;&nbsp; Abu Dhabi, United Arab Emirates<br/>
    <a href="mailto:enquiries@wavesad.com">enquiries@wavesad.com</a><br/><br/>
    All sources cited. Intelligence compiled from public domain news.<br/>
    &copy; {YEAR} Wavesad Technologies LLC. All rights reserved.
  </div>
</div>

</div>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════
# PDF EXPORT
# ═══════════════════════════════════════════════════════════

def export_pdf(html_path: Path) -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [PDF] Playwright not installed. Run: pip install playwright && playwright install chromium")
        return None

    pdf_path = html_path.with_suffix(".pdf")
    print(f"  [PDF] Rendering {html_path.name} ...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 800, "height": 1200})
        page.goto(f"file:///{html_path.resolve()}")
        page.wait_for_timeout(2500)   # let Google Fonts load
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        )
        browser.close()
    print(f"  [PDF] Saved → {pdf_path}")
    return pdf_path


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="store_true", help="Export PDF after generating HTML")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  NABDH (نبض) — UAE AI INTELLIGENCE NEWSLETTER")
    print(f"  {TODAY}")
    print(f"{'=' * 60}\n")

    if not DEEPSEEK_API_KEY:
        print("  [ERROR] Set DEEPSEEK_API_KEY or Deepseek_API_Key_1 in .env")
        raise SystemExit(1)
    if not INPUT_FILE.exists():
        print(f"  [ERROR] Not found: {INPUT_FILE}")
        raise SystemExit(1)

    logo_uri    = load_logo()
    edition_num = get_edition_number()

    keypoints, watch_items = load_keypoints()
    tracker = load_tracker()

    print(f"  Model      : {MODEL}")
    print(f"  Main feed  : {len(keypoints)} articles")
    print(f"  Watch      : {len(watch_items)} articles")
    print(f"  Tracker    : {len(tracker)} developments")
    print(f"  Edition    : {edition_num}\n")

    if not keypoints:
        print("  [ERROR] No keypoints. Run uae_ai_keypoints_v2.py first.")
        raise SystemExit(1)

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

    print("  [1/3] Generating editorial...")
    editorial = generate_editorial(keypoints, client)
    print(f"  [EDITORIAL] {len(editorial)} chars")
    print(f"  [PREVIEW]   {repr(editorial[:120])}")

    print("  [2/3] Generating hook lines...")
    hooks = generate_hooks(keypoints, client)

    print("  [3/3] Generating closing...")
    closing = generate_closing(keypoints, client)

    print("\n  Building HTML...")
    html = build_html(keypoints, watch_items, tracker, editorial, hooks, closing, edition_num, logo_uri)

    # Quality gate
    issues = []
    if re.search(r'(?i)paragraph\s*[123]', html):  issues.append("CRITICAL: Paragraph labels in output")
    if "**" in html:                               issues.append("CRITICAL: Markdown ** in output")
    if "ISSUE #" in html:                          issues.append("CRITICAL: 'ISSUE #' in output")
    if "CONFIDENTIAL" in html:                     issues.append("CRITICAL: CONFIDENTIAL in output")
    if "Turnkey" in html:                          issues.append("CRITICAL: Vendor tagline in output")
    if html.count('href="#"') > 3:                 issues.append(f"WARN: {html.count('href=\"#\"')} empty links")

    if issues:
        print("\n  Quality check:")
        for iss in issues:
            tag = "[FAIL]" if "CRITICAL" in iss else "[WARN]"
            print(f"    {tag} {iss}")
        if any("CRITICAL" in i for i in issues):
            print("\n  [ABORT] Critical quality issues. File NOT saved.")
            raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"nabdh_{DATE_STR}.html"
    out_path.write_text(html, encoding="utf-8")
    _increment_edition_number()
    print(f"  [EDITION] Counter updated → next edition will be {get_edition_number()}")

    pdf_path = None
    if args.pdf:
        pdf_path = export_pdf(out_path)

    print(f"""
{'=' * 60}
  NABDH (نبض) — EDITION {edition_num} COMPLETE
{'=' * 60}
  Stories      : {len(keypoints[:12])}
  Tracker items: {len(tracker)}
  Mobile ready : YES
  Quality      : {'All checks passed' if not issues else f'{len(issues)} warnings'}
  HTML         : {out_path}
  PDF          : {pdf_path if pdf_path else 'not exported (use --pdf)'}
{'=' * 60}
""")


if __name__ == "__main__":
    main()
