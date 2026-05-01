"""
UAE AI News — Arabic → English Translator (DeepSeek API Edition)
────────────────────────────────────────────────────────────────────────────
Reads Arabic JSON files from news_output/arabic/
Translates: title, summary, source
Writes translated versions to news_output/arabic_translated/
(mirrors the same directory structure)

Model used: deepseek-v4-flash  (DeepSeek API, OpenAI-compatible)

Translation features:
  - Parallel processing with up to 3 API keys (ThreadPoolExecutor)
  - Batch translation — multiple texts per API call for efficiency
  - Incremental save per file (crash-safe)
  - Original Arabic values preserved alongside translations
  - Known UAE newspaper name map (no API call for common sources)
  - Skip already-translated files on re-run
  - Retry with key fallback on failure
  - Single timestamped log file per run

Install dependencies:
  pip install openai

Set environment variables before running:
  set DeepSeek_API_Key_1=sk-...
  set DeepSeek_API_Key_2=sk-...
  set DeepSeek_API_Key_3=sk-...

Run:
  python uae_ai_translate.py
────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG
# ═════════════════════════════════════════════════════════════════════════════

NEWS_DATE      = os.environ.get("NEWS_DATE", datetime.now().strftime("%Y-%m-%d"))
INPUT_DIR      = os.path.join("news_output", NEWS_DATE, "arabic")
OUTPUT_DIR     = os.path.join("news_output", NEWS_DATE, "arabic_translated")
BATCH_SIZE     = 32      # texts sent per API call (per key)
SKIP_IF_EXISTS = True    # set False to force re-translation of all files

# DeepSeek API — OpenAI-compatible
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-v4-flash"   # swap to "deepseek-v4" for higher quality
MAX_RETRIES       = 2
REQUEST_TIMEOUT   = 60   # seconds per API request

# API keys loaded from environment variables
DEEPSEEK_API_KEYS = [
    os.environ.get("DeepSeek_API_Key_1", ""),
    os.environ.get("DeepSeek_API_Key_2", ""),
    os.environ.get("DeepSeek_API_Key_3", ""),
]

LOGS_DIR = "logs"   # one log file per run is written here

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — KNOWN UAE NEWSPAPER NAME MAP
# The DeepSeek model is unreliable for proper nouns / publication names.
# This map overrides the API output for known Arabic-language UAE sources.
# If a source is not in this map, API translation is used as fallback.
# ═════════════════════════════════════════════════════════════════════════════

UAE_SOURCE_MAP = {
    # UAE Arabic newspapers
    "الخليج":                     "Al Khaleej (UAE)",
    "البيان":                     "Al Bayan (UAE)",
    "الاتحاد":                    "Al Ittihad (UAE)",
    "إمارات اليوم":               "Emarat Al Youm (UAE)",
    "خليج تايمز":                 "Khaleej Times (UAE)",
    "الرؤية":                     "Al Roeya (UAE)",
    "نشرة الإمارات":              "UAE Bulletin",
    "وام":                        "WAM — UAE State News Agency",
    # Regional Arabic outlets commonly covering UAE
    "العربية":                    "Al Arabiya",
    "الجزيرة":                    "Al Jazeera",
    "سكاي نيوز عربية":            "Sky News Arabia",
    "روسيا اليوم":                "RT Arabic",
    "الشرق الأوسط":              "Asharq Al-Awsat",
    "عرب نيوز":                   "Arab News",
    "الأهرام":                    "Al-Ahram (Egypt)",
    "الوطن":                      "Al Watan",
    "البورصة":                    "Al Borsa",
    "CNN عربي":                   "CNN Arabic",
    "بي بي سي عربي":             "BBC Arabic",
    "Forbes Middle East":         "Forbes Middle East",  # already English
    "فوربس الشرق الأوسط":        "Forbes Middle East",
}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOGGING UTILITY
# ═════════════════════════════════════════════════════════════════════════════

# Module-level logger — configured once in run() via setup_logger()
logger: logging.Logger = logging.getLogger("uae_translate")


def setup_logger() -> logging.Logger:
    """
    Create a timestamped log file for this run and configure the module logger.
    Console shows INFO+; log file captures DEBUG+ (every API call).
    """
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    run_ts   = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOGS_DIR) / f"translation_{run_ts}.log"

    lg = logging.getLogger("uae_translate")
    lg.setLevel(logging.DEBUG)
    lg.handlers.clear()   # avoid duplicate handlers on re-runs

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    lg.addHandler(fh)
    lg.addHandler(ch)
    lg.info(f"  Log file : {log_file.resolve()}")
    return lg


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — API KEY MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class APIKeyManager:
    """Thread-safe round-robin distributor across multiple DeepSeek API keys."""

    def __init__(self, keys: list):
        valid = [k for k in keys if k and k.strip()]
        if not valid:
            raise ValueError(
                "No valid DeepSeek API keys found.\n"
                "Set environment variables: DeepSeek_API_Key_1 / _2 / _3"
            )
        self._keys  = valid
        self._index = 0
        self._lock  = threading.Lock()

    def next_key(self) -> str:
        """Return the next key in round-robin order (thread-safe)."""
        with self._lock:
            key = self._keys[self._index % len(self._keys)]
            self._index += 1
            return key

    def all_keys(self) -> list:
        return list(self._keys)

    @property
    def count(self) -> int:
        return len(self._keys)

    @staticmethod
    def mask(key: str) -> str:
        """Partially obscure a key for safe log output."""
        if not key:
            return "<empty>"
        if len(key) <= 12:
            return key[:4] + "***"
        return key[:8] + "..." + key[-4:]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DEEPSEEK API HANDLER
# ═════════════════════════════════════════════════════════════════════════════

def _make_client(api_key: str) -> OpenAI:
    """Instantiate an OpenAI-compatible client pointed at DeepSeek."""
    return OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=REQUEST_TIMEOUT,
    )


def _call_deepseek(texts: list, api_key: str, key_manager: APIKeyManager) -> list:
    """
    Send a numbered-list prompt to DeepSeek and return translations in order.
    On failure, retries up to MAX_RETRIES times cycling through available keys.
    Falls back to returning the original text if all attempts fail.
    """
    if not texts:
        return []

    numbered_input = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

    system_prompt = (
        "You are a professional Arabic-to-English translator. "
        "Translate each numbered Arabic text to fluent English. "
        "Return ONLY the numbered translations in exactly the same numbered format. "
        "Do not add commentary, notes, or change the numbering."
    )
    user_prompt = f"Translate the following Arabic texts to English:\n\n{numbered_input}"

    # Try the assigned key first, then cycle through others for fallback
    all_keys     = key_manager.all_keys()
    remaining    = [k for k in all_keys if k != api_key]
    keys_to_try  = [api_key] + remaining

    for attempt, key in enumerate(keys_to_try[: MAX_RETRIES + 1]):
        masked = APIKeyManager.mask(key)
        try:
            client   = _make_client(key)
            logger.debug(
                f"API call | key={masked} | texts={len(texts)} | attempt={attempt + 1}"
            )
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,   # deterministic output
            )
            raw          = response.choices[0].message.content.strip()
            translations = _parse_numbered_response(raw, len(texts))
            logger.debug(
                f"API success | key={masked} | texts={len(texts)} | attempt={attempt + 1}"
            )
            return translations

        except Exception as exc:
            logger.warning(
                f"API error | key={masked} | attempt={attempt + 1} | {exc}"
            )

    # All attempts exhausted — return originals so pipeline does not crash
    logger.error(f"All API attempts failed for batch of {len(texts)} texts; returning originals")
    return list(texts)


def _parse_numbered_response(raw: str, expected_count: int) -> list:
    """
    Parse a numbered-list API response.
    Handles multi-line translations by joining continuation lines.
    Returns a list of length expected_count; missing entries remain "".
    """
    results     = [""] * expected_count
    current_idx = -1
    buf         = []

    def _flush():
        if current_idx >= 0 and 0 <= current_idx < expected_count:
            results[current_idx] = " ".join(buf).strip()

    for line in raw.splitlines():
        stripped = line.strip()
        m = re.match(r"^(\d+)\.\s*(.*)", stripped)
        if m:
            _flush()
            current_idx = int(m.group(1)) - 1
            buf = [m.group(2).strip()] if m.group(2).strip() else []
        elif current_idx >= 0 and stripped:
            buf.append(stripped)

    _flush()
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRANSLATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def translate_batch(key_manager: APIKeyManager, texts: list) -> list:
    """
    Translate a batch of Arabic strings to English via DeepSeek API.
    Work is distributed across all available API keys in parallel using threads.
    Returns a list of translated strings in the same order as the input.
    Empty / blank entries are passed through without an API call.
    """
    indices_to_translate = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    results = [""] * len(texts)

    if not indices_to_translate:
        return results

    real_texts = [t for _, t in indices_to_translate]
    n_keys     = key_manager.count

    # Divide texts into one chunk per key so all keys work in parallel
    chunk_size = max(1, (len(real_texts) + n_keys - 1) // n_keys)
    chunks     = [real_texts[i : i + chunk_size] for i in range(0, len(real_texts), chunk_size)]
    all_keys   = key_manager.all_keys()

    chunk_results: dict = {}

    with ThreadPoolExecutor(max_workers=n_keys) as executor:
        futures = {
            executor.submit(
                _call_deepseek,
                chunk,
                all_keys[ci % len(all_keys)],
                key_manager,
            ): ci
            for ci, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            ci = futures[future]
            try:
                chunk_results[ci] = future.result()
            except Exception as exc:
                logger.error(f"Chunk {ci} thread error: {exc}")
                chunk_results[ci] = list(chunks[ci])   # return originals

    # Reassemble chunks in original order
    translated_real = []
    for ci in range(len(chunks)):
        translated_real.extend(chunk_results.get(ci, list(chunks[ci])))

    # Map back to original positions (including empty-string slots)
    for (original_index, _), translation in zip(indices_to_translate, translated_real):
        results[original_index] = translation

    return results


def resolve_source(key_manager: APIKeyManager, source_arabic: str) -> tuple:
    """
    Translate a source/publication name.
    Uses known UAE newspaper map first; falls back to DeepSeek API.
    Returns (english_name, method_used).
    """
    if not source_arabic or not source_arabic.strip():
        return "Unknown Source", "empty"

    stripped = source_arabic.strip()

    if stripped in UAE_SOURCE_MAP:
        return UAE_SOURCE_MAP[stripped], "map"

    try:
        translated = translate_batch(key_manager, [stripped])
        return translated[0] if translated[0] else stripped, "model"
    except Exception:
        return stripped, "untranslated"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FILE PROCESSOR
# ═════════════════════════════════════════════════════════════════════════════

def process_file(key_manager: APIKeyManager, input_path: Path, output_path: Path):
    """
    Translate all articles in one JSON file and write to output path.
    Preserves original Arabic values in *_original fields.
    """
    logger.info(f"\n  Processing: {input_path.name}")

    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if not articles:
        logger.info("    No articles found, skipping.")
        write_json(output_path, [])
        return 0

    total = len(articles)
    logger.info(f"    Articles to translate: {total}")

    # ── Sources: resolve unique ones, batch API calls for unknown sources ────
    logger.info("    Translating sources...")
    unique_sources        = list({a.get("source", "") for a in articles})
    source_translation_map = {}

    # Pass 1: map lookups (free, no API call)
    api_needed = []
    for src in unique_sources:
        stripped = src.strip() if src else ""
        if not stripped:
            source_translation_map[src] = {"translated": "Unknown Source", "method": "empty"}
        elif stripped in UAE_SOURCE_MAP:
            source_translation_map[src] = {"translated": UAE_SOURCE_MAP[stripped], "method": "map"}
        else:
            api_needed.append(src)

    # Pass 2: batch-translate unknown sources in one API call
    if api_needed:
        api_translations = translate_batch(key_manager, api_needed)
        for src, translated in zip(api_needed, api_translations):
            source_translation_map[src] = {
                "translated": translated if translated else src,
                "method": "model",
            }

    map_hits   = sum(1 for v in source_translation_map.values() if v["method"] == "map")
    model_hits = sum(1 for v in source_translation_map.values() if v["method"] == "model")
    logger.info(
        f"    Sources: {len(unique_sources)} unique → "
        f"{map_hits} from name map, {model_hits} from API"
    )

    # ── Translate titles and summaries concurrently ──────────────────────────
    # Both fields are submitted at the same time; each internally distributes
    # across all API keys in parallel via translate_batch.
    logger.info("    Translating titles and summaries in parallel...")
    titles    = [a.get("title",   "") for a in articles]
    summaries = [a.get("summary", "") for a in articles]

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_titles    = executor.submit(translate_batch, key_manager, titles)
        future_summaries = executor.submit(translate_batch, key_manager, summaries)
        translated_titles    = future_titles.result()
        translated_summaries = future_summaries.result()

    logger.info(f"    Titles + summaries done ({total} articles)")

    # ── Assemble translated articles ─────────────────────────────────────────
    translated_articles = []
    for i, article in enumerate(articles):
        src_info = source_translation_map.get(article.get("source", ""), {})
        translated_articles.append({
            **article,                                   # all original fields
            # ── Translated fields ──────────────────────────────────────────────
            "title":            translated_titles[i],
            "summary":          translated_summaries[i],
            "source":           src_info.get("translated", article.get("source", "")),
            # ── Original Arabic preserved ─────────────────────────────────────
            "title_original":   article.get("title", ""),
            "summary_original": article.get("summary", ""),
            "source_original":  article.get("source", ""),
            # ── Translation metadata ──────────────────────────────────────────
            "translation": {
                "model":           DEEPSEEK_MODEL,
                "source_method":   src_info.get("method", "unknown"),
                "translated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_language": "Arabic",
                "target_language": "English",
            },
        })

    write_json(output_path, translated_articles)
    logger.info(f"    Done → {output_path}")
    return total


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FILE OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def write_json(filepath: Path, data):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    count = len(data) if isinstance(data, list) else 1
    logger.info(f"  [SAVED] {filepath}  ({count} records)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run():
    global logger
    logger = setup_logger()

    # Initialise key manager — exits early if no keys are configured
    try:
        key_manager = APIKeyManager(DEEPSEEK_API_KEYS)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}\n")
        raise SystemExit(1)

    input_base  = Path(INPUT_DIR)
    output_base = Path(OUTPUT_DIR)

    if not input_base.exists():
        logger.error(f"Input directory not found: {input_base}")
        print("  Run uae_ai_news.py first to generate the Arabic JSON files.\n")
        raise SystemExit(1)

    json_files = sorted(input_base.glob("*.json"))
    if not json_files:
        logger.error(f"No JSON files found in {input_base}")
        raise SystemExit(1)

    output_base.mkdir(parents=True, exist_ok=True)

    print(f"""
{'='*65}
  UAE AI NEWS — ARABIC TO ENGLISH TRANSLATOR
{'='*65}
  Input  : {input_base.resolve()}
  Output : {output_base.resolve()}
  Files  : {len(json_files)} JSON files found
  Model  : {DEEPSEEK_MODEL}
  Batch  : {BATCH_SIZE} articles/batch
  Keys   : {key_manager.count} API key(s) active
{'='*65}
""")

    # ── Check which files still need translation ──────────────────────────────
    files_to_process = []
    for json_file in json_files:
        out_file = output_base / json_file.name
        if SKIP_IF_EXISTS and out_file.exists():
            print(f"  [SKIP] Already translated: {json_file.name}")
        else:
            files_to_process.append(json_file)

    if not files_to_process:
        print("\n  All files already translated. Set SKIP_IF_EXISTS=False to re-run.\n")
        raise SystemExit(0)

    print(f"\n  Files to process: {len(files_to_process)}\n")

    # ── Process each file ────────────────────────────────────────────────────
    total_articles = 0
    run_start      = time.time()

    for json_file in files_to_process:
        out_file = output_base / json_file.name
        count    = process_file(key_manager, json_file, out_file)
        total_articles += count

    elapsed      = time.time() - run_start
    mins, secs   = divmod(int(elapsed), 60)

    print(f"""
{'='*65}
  TRANSLATION COMPLETE
{'='*65}
  Files translated   : {len(files_to_process)}
  Total articles     : {total_articles}
  Time taken         : {mins}m {secs}s
  Output directory   : {output_base.resolve()}/

  NOTE: Each article now contains:
    title           → English translation
    title_original  → Original Arabic (preserved)
    summary         → English translation
    summary_original→ Original Arabic (preserved)
    source          → English name (map or API)
    source_original → Original Arabic (preserved)
    translation.*   → Metadata (model, method, timestamp)
{'='*65}
""")


if __name__ == "__main__":
    run()
