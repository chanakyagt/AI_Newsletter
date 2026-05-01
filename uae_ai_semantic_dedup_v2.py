#!/usr/bin/env python3
"""
uae_ai_semantic_dedup_v2.py — Step 4
Semantic deduplication + dual source selection for UAE AI Newsletter.

Input:  news_output/english/*.json          (English articles)
        news_output/arabic_translated/*.json (translated Arabic articles)
Output: news_output/deduped/distinct_articles.json  (~5000 articles, dual source structure)
"""

import json
import os
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ──────────────────────────────── CONFIG ──────────────────────────────────────
INPUT_DIRS = [
    Path("news_output/english"),
    Path("news_output/arabic_translated"),
]
OUTPUT_DIR           = Path("news_output/deduped")
SIMILARITY_THRESHOLD = 0.88
EMBEDDING_MODEL      = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 500
BLOCK_SIZE           = 500
MIN_CHUNK_CHARS      = 20

OPENAI_API_KEY_1 = os.environ.get("OPENAI_API_KEY_1")
OPENAI_API_KEY_2 = os.environ.get("OPENAI_API_KEY_2")
OPENAI_API_KEY_3 = os.environ.get("OPENAI_API_KEY_3")
# ──────────────────────────────────────────────────────────────────────────────

_PRINT_LOCK = threading.Lock()


def _sep(char="═", width=63):
    print(char * width)


# ───────────────────────── STEP 1: STARTUP VALIDATION ────────────────────────
def validate_setup():
    print()
    _sep()
    print("  UAE AI NEWSLETTER — SEMANTIC DEDUPLICATION")
    print(f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _sep()
    print()

    active_keys = []
    for i, key in enumerate([OPENAI_API_KEY_1, OPENAI_API_KEY_2, OPENAI_API_KEY_3], 1):
        label = f"OPENAI_API_KEY_{i}"
        if key:
            print(f"  [KEY CHECK] {label} ✓")
            active_keys.append((f"KEY-{i}", key))
        else:
            print(f"  [KEY CHECK] {label} ✗  (not set)")

    if not active_keys:
        print("\n  [ERROR] No OpenAI API keys found.")
        print("  Set OPENAI_API_KEY_1, OPENAI_API_KEY_2, OPENAI_API_KEY_3 in .env")
        raise SystemExit(1)

    print(f"  [PARALLEL]  {len(active_keys)} key(s) active — embedding will use parallel threads")
    print()

    try:
        from url_resolution.unified_registry import get_profile, is_paywalled, is_free, get_credibility  # noqa: F401
        print("  [REGISTRY] unified_registry.py loaded ✓")
    except ImportError:
        print("  [ERROR] url_resolution/unified_registry.py not found.")
        print("  Run generate_unified_registry.py first.")
        raise SystemExit(1)

    try:
        import faiss  # noqa: F401
        faiss_available = True
        print("  [SIMILARITY] FAISS available — fast mode")
    except ImportError:
        faiss_available = False
        print("  [SIMILARITY] FAISS not installed — using numpy blocks (~35s for 7000 articles)")
        print("  Optional install: pip install faiss-cpu  OR  conda install -c conda-forge faiss-cpu")

    print()
    return active_keys, faiss_available


# ───────────────────────── STEP 2: LOAD ARTICLES ─────────────────────────────
def load_articles():
    from url_resolution.unified_registry import get_profile

    seen_ids: dict = {}   # article_id → article, for dedup across directories

    for input_dir in INPUT_DIRS:
        label = str(input_dir)
        if not input_dir.exists():
            print(f"  [WARN] Input directory not found, skipping: {input_dir}")
            continue

        json_files = sorted(input_dir.glob("*.json"))
        if not json_files:
            print(f"  [WARN] No JSON files in {input_dir}, skipping")
            continue

        print(f"  Loading from {input_dir}/")
        max_len = max(len(f.name) for f in json_files)

        for fpath in json_files:
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"    [WARN] {fpath.name}: failed to parse ({e}), skipping")
                continue

            if isinstance(data, list):
                batch = data
            elif isinstance(data, dict):
                batch = data.get("articles", [])
            else:
                print(f"    [WARN] {fpath.name}: unexpected format, skipping")
                continue

            new_count = 0
            for article in batch:
                aid = article.get("article_id")
                if aid and aid in seen_ids:
                    continue
                if aid:
                    seen_ids[aid] = article
                else:
                    # No article_id — include but can't dedup
                    seen_ids[id(article)] = article
                new_count += 1

            print(f"    {fpath.name:{max_len}} → {new_count:,} articles  "
                  f"({len(batch) - new_count} duplicates skipped)")

    all_articles = list(seen_ids.values())

    print(f"  {'─' * 50}")
    print(f"  Total loaded: {len(all_articles):,} articles (deduplicated by article_id)")
    print()

    if not all_articles:
        print("  [ERROR] No articles loaded. Exiting.")
        raise SystemExit(1)

    for i, article in enumerate(all_articles):
        article["_index"] = i

    tier_counts     = defaultdict(int)
    paywalled_count = 0
    not_in_registry = 0

    for article in all_articles:
        profile = get_profile(source_name=article.get("source", ""))
        article["_credibility_tier"]  = profile["credibility_tier"]
        article["_is_paywalled"]      = profile["is_paywalled"]
        article["_paywall_status"]    = profile["paywall_status"]
        article["_found_in_registry"] = profile["found_in_registry"]
        tier_counts[profile["credibility_tier"]] += 1
        if profile["is_paywalled"]:
            paywalled_count += 1
        if not profile["found_in_registry"]:
            not_in_registry += 1

    tier_labels = {5: "UAE Govt", 4: "Major UAE", 3: "Regional", 2: "Intl Press", 1: "Unknown"}
    print("  Source enrichment complete:")
    for tier in sorted(tier_counts.keys(), reverse=True):
        label = tier_labels.get(tier, f"Tier {tier}")
        print(f"    Tier {tier} ({label:<10}): {tier_counts[tier]:>5,} articles")
    print(f"    Paywalled sources   : {paywalled_count:>5,} articles")
    print(f"    Not in registry     : {not_in_registry:>5,} articles  (will use Tier 1 default)")
    print()

    return all_articles, tier_counts, paywalled_count, not_in_registry


# ───────────────────── STEP 3: BUILD EMBEDDING CHUNKS ────────────────────────
def build_chunks(articles):
    chunks = []
    skip_count = 0

    for article in articles:
        title   = (article.get("title")   or "").strip()
        summary = (article.get("summary") or "").strip()
        chunk   = title + ". " + summary

        if len(chunk) < MIN_CHUNK_CHARS:
            article["_skip_embedding"] = True
            chunks.append("")
            skip_count += 1
        else:
            article["_skip_embedding"] = False
            chunks.append(chunk)

    if skip_count:
        print(f"  [CHUNKS] {len(chunks):,} built, {skip_count} too short (pass through as unique)")
    else:
        print(f"  [CHUNKS] {len(chunks):,} built")

    return chunks


# ───────────────────── STEP 4: PARALLEL EMBEDDING ────────────────────────────
def _embed_batches(key_label, api_key, batch_assignments, model):
    from openai import OpenAI, RateLimitError
    client = OpenAI(api_key=api_key)
    results = []
    total = len(batch_assignments)
    dim = 1536

    for pos, (batch_global_idx, texts, orig_indices) in enumerate(batch_assignments, 1):
        start_art = (orig_indices[0] + 1) if orig_indices else 0
        end_art   = (orig_indices[-1] + 1) if orig_indices else 0
        success   = False

        for attempt in range(1, 5):
            try:
                t0 = time.time()
                response = client.embeddings.create(model=model, input=texts)
                vectors  = [e.embedding for e in response.data]
                elapsed  = time.time() - t0

                with _PRINT_LOCK:
                    print(f"  [{key_label}] Batch {pos:>2}/{total:<2} "
                          f"(art. {start_art:>5}-{end_art:<5}) ✓  {elapsed:.1f}s")

                results.append((batch_global_idx, vectors))
                success = True
                break

            except RateLimitError:
                with _PRINT_LOCK:
                    print(f"  [{key_label}] Batch {batch_global_idx} attempt {attempt} "
                          f"— rate limited, waiting 10s")
                time.sleep(10)
            except Exception as e:
                with _PRINT_LOCK:
                    print(f"  [{key_label}] Batch {batch_global_idx} attempt {attempt} "
                          f"— {e!r}, waiting 2s")
                time.sleep(2)

        if not success:
            with _PRINT_LOCK:
                print(f"  [{key_label}] Batch {batch_global_idx} permanently failed "
                      f"— using zero vector fallback")
            results.append((batch_global_idx, [[0.0] * dim] * len(texts)))

    return results


def embed_articles(chunks, active_keys):
    n   = len(chunks)
    dim = 1536

    non_empty = [(i, chunks[i]) for i in range(n) if chunks[i]]

    batches = []
    for b_start in range(0, len(non_empty), EMBEDDING_BATCH_SIZE):
        b_slice   = non_empty[b_start : b_start + EMBEDDING_BATCH_SIZE]
        orig_idxs = [pair[0] for pair in b_slice]
        texts     = [pair[1] for pair in b_slice]
        batches.append((len(batches), texts, orig_idxs))

    total_batches = len(batches)
    num_keys      = len(active_keys)

    key_assignments = defaultdict(list)
    for pos, batch_tuple in enumerate(batches):
        key_assignments[pos % num_keys].append(batch_tuple)

    print(f"  Embedding {n:,} articles in {total_batches} batches across {num_keys} parallel key(s):")
    print()

    t0 = time.time()
    all_results = []

    with ThreadPoolExecutor(max_workers=num_keys) as pool:
        futures = {
            pool.submit(_embed_batches, label, key, key_assignments[ki], EMBEDDING_MODEL): ki
            for ki, (label, key) in enumerate(active_keys)
            if key_assignments[ki]
        }
        for future in futures:
            all_results.extend(future.result())

    elapsed = time.time() - t0

    batch_map    = {bgidx: vecs for bgidx, vecs in all_results}
    vectors_list = [[0.0] * dim] * n

    for bgidx, _texts, orig_idxs in batches:
        if bgidx in batch_map:
            for local_i, global_i in enumerate(orig_idxs):
                if local_i < len(batch_map[bgidx]):
                    vectors_list[global_i] = batch_map[bgidx][local_i]

    total_tokens_est = sum(len(c.split()) * 1.3 for _, c in non_empty)
    cost_est = (total_tokens_est / 1_000_000) * 0.02

    print()
    print(f"  Embedding complete:")
    print(f"    Batches     : {total_batches} across {num_keys} parallel key(s)")
    print(f"    Total time  : {elapsed:.1f}s")
    print(f"    Est. cost   : ${cost_est:.4f}")
    print()

    return vectors_list, elapsed, total_batches, cost_est


# ───────────────────── STEP 5: NORMALISE VECTORS ─────────────────────────────
def normalise_vectors(vectors_list):
    vectors = np.array(vectors_list, dtype=np.float32)
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms   = np.where(norms == 0, 1.0, norms)
    return vectors / norms


# ───────────────────── STEP 6: FIND SIMILAR PAIRS ────────────────────────────
def find_similar_pairs_faiss(normalised):
    import faiss
    n   = normalised.shape[0]
    dim = normalised.shape[1]
    k   = min(50, n)

    index = faiss.IndexFlatIP(dim)
    index.add(normalised)

    t0 = time.time()
    distances, indices = index.search(normalised, k=k)
    elapsed = time.time() - t0

    pairs = []
    for i in range(n):
        for rank in range(k):
            j   = int(indices[i][rank])
            sim = float(distances[i][rank])
            if j > i and sim >= SIMILARITY_THRESHOLD:
                pairs.append((i, j, sim))

    print(f"  [FAISS]  Similarity search: {elapsed:.1f}s — {len(pairs):,} duplicate pairs found")
    print(f"  Threshold: {SIMILARITY_THRESHOLD}")
    print()
    return pairs, elapsed


def find_similar_pairs_numpy(normalised):
    n            = normalised.shape[0]
    total_blocks = -(-n // BLOCK_SIZE)
    pairs        = []

    t0 = time.time()
    for blk in range(0, n, BLOCK_SIZE):
        block     = normalised[blk : blk + BLOCK_SIZE]
        sim_block = block @ normalised.T

        rows, cols = np.where(sim_block >= SIMILARITY_THRESHOLD)
        for r, c in zip(rows, cols):
            i = blk + int(r)
            j = int(c)
            if j > i:
                pairs.append((i, j, float(sim_block[r, c])))

        del sim_block
        block_num = blk // BLOCK_SIZE + 1
        print(f"  Block {block_num}/{total_blocks} complete", end="\r")

    elapsed = time.time() - t0
    print(f"  [NUMPY]  Similarity search: {elapsed:.1f}s — {len(pairs):,} duplicate pairs found")
    print(f"  Threshold: {SIMILARITY_THRESHOLD}")
    print()
    return pairs, elapsed


# ───────────────────── STEP 7: UNION-FIND CLUSTERING ─────────────────────────
def cluster_articles(articles, duplicate_pairs):
    n      = len(articles)
    parent = list(range(n))
    rank   = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    for i, j, _sim in duplicate_pairs:
        union(i, j)

    # Per-article adjacency for efficient intra-cluster similarity lookup
    article_sims = defaultdict(dict)
    for i, j, sim in duplicate_pairs:
        article_sims[i][j] = sim
        article_sims[j][i] = sim

    clusters = defaultdict(list)
    for article in articles:
        clusters[find(article["_index"])].append(article)

    n_clusters = len(clusters)
    dup_groups = sum(1 for c in clusters.values() if len(c) > 1)
    merged     = n - n_clusters
    reduction  = (merged / n * 100) if n else 0.0

    print(f"  Clustering complete:")
    print(f"    Input articles    : {n:,}")
    print(f"    Distinct clusters : {n_clusters:,}")
    print(f"    Duplicate groups  : {dup_groups:,}   (clusters with 2+ articles)")
    print(f"    Articles to merge : {merged:,}")
    print(f"    Reduction         : {reduction:.1f}%")
    print()

    return clusters, article_sims, n_clusters, dup_groups, merged, reduction


# ───────────────────── STEP 8: DUAL SOURCE SELECTION ─────────────────────────
def _safe_date(article):
    try:
        return datetime.fromisoformat(
            article["published_date"].replace("Z", "+00:00")
        )
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def select_dual_sources(cluster, article_sims):
    if len(cluster) == 1:
        art        = cluster[0]
        base_score = float(art["_credibility_tier"] * 100)
        result     = dict(art)
        result["content_source"] = {
            "article_id":       art["article_id"],
            "source_name":      art.get("source", ""),
            "rss_url":          art["url"],
            "credibility_tier": art["_credibility_tier"],
            "is_paywalled":     art["_is_paywalled"],
            "summary":          art.get("summary", "") or "",
            "content_score":    base_score,
        }
        result["citation_differs_from_content"]   = False
        result["content_fallback_to_rss_summary"] = art["_is_paywalled"]
        result["dedup_metadata"] = {
            "cluster_size":         1,
            "cluster_sources":      [art.get("source", "")],
            "all_paywalled":        art["_is_paywalled"],
            "had_free_alternative": not art["_is_paywalled"],
            "citation_score":       base_score,
            "content_score":        base_score,
            "selection_reason":     "Single article — no deduplication needed.",
            "similarity_scores":    [],
        }
        return result

    # ── Multi-article cluster ─────────────────────────────────────────────────
    sorted_by_date = sorted(cluster, key=_safe_date, reverse=True)
    recency = {}
    if len(sorted_by_date) >= 1:
        recency[sorted_by_date[0]["_index"]] = 5
    if len(sorted_by_date) >= 2:
        recency[sorted_by_date[1]["_index"]] = 3

    def citation_score(a):
        base = a["_credibility_tier"] * 100 + (10 if a["_is_paywalled"] else 0)
        return base + recency.get(a["_index"], 0)

    def content_score(a):
        free_bonus    = 50 if not a["_is_paywalled"] else 0
        summary_bonus = min(len(a.get("summary", "") or ""), 200) / 10
        return a["_credibility_tier"] * 100 + free_bonus + summary_bonus

    citation_art  = max(cluster, key=citation_score)
    c_score       = citation_score(citation_art)
    all_paywalled = all(a["_is_paywalled"] for a in cluster)

    if all_paywalled:
        content_art  = citation_art
        cnt_score    = content_score(content_art)
        fallback_rss = True
    else:
        content_art  = max(cluster, key=content_score)
        cnt_score    = content_score(content_art)
        fallback_rss = False

    differs  = (citation_art["_index"] != content_art["_index"])
    had_free = any(not a["_is_paywalled"] for a in cluster)

    # Collect intra-cluster similarity scores efficiently
    cluster_indices = {a["_index"] for a in cluster}
    seen_pairs  = set()
    cluster_sims_list = []
    for a in cluster:
        idx = a["_index"]
        for other_idx, sim in article_sims.get(idx, {}).items():
            if other_idx in cluster_indices:
                pair_key = (min(idx, other_idx), max(idx, other_idx))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    cluster_sims_list.append(round(sim, 4))

    # Build selection reason
    if all_paywalled:
        reason = (
            f"All {len(cluster)} articles paywalled. "
            f"Citation+Content: {citation_art.get('source', '?')} "
            f"(T{citation_art['_credibility_tier']}). RSS summary fallback flagged."
        )
    elif differs:
        cite_bonus = " + paywall prestige bonus" if citation_art["_is_paywalled"] else ""
        reason = (
            f"Citation: {citation_art.get('source', '?')} "
            f"(T{citation_art['_credibility_tier']}{cite_bonus}). "
            f"Content: {content_art.get('source', '?')} "
            f"(T{content_art['_credibility_tier']} + free bonus, best Firecrawl target)."
        )
    else:
        reason = (
            f"Single best article: {citation_art.get('source', '?')} "
            f"(T{citation_art['_credibility_tier']}) serves both roles."
        )

    result = dict(citation_art)
    result["content_source"] = {
        "article_id":       content_art["article_id"],
        "source_name":      content_art.get("source", ""),
        "rss_url":          content_art["url"],
        "credibility_tier": content_art["_credibility_tier"],
        "is_paywalled":     content_art["_is_paywalled"],
        "summary":          content_art.get("summary", "") or "",
        "content_score":    round(cnt_score, 2),
    }
    result["citation_differs_from_content"]   = differs
    result["content_fallback_to_rss_summary"] = fallback_rss
    result["dedup_metadata"] = {
        "cluster_size":         len(cluster),
        "cluster_sources":      [a.get("source", "") for a in cluster],
        "all_paywalled":        all_paywalled,
        "had_free_alternative": had_free,
        "citation_score":       round(c_score, 2),
        "content_score":        round(cnt_score, 2),
        "selection_reason":     reason,
        "similarity_scores":    cluster_sims_list,
    }
    return result


def strip_internal(article):
    return {k: v for k, v in article.items() if not k.startswith("_")}


# ───────────────────────────── MAIN ──────────────────────────────────────────
def main():
    run_start = time.time()
    run_at    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    active_keys, faiss_available = validate_setup()

    articles, tier_counts, paywalled_count, not_in_registry = load_articles()

    chunks = build_chunks(articles)

    vectors_list, embed_time, total_batches, cost_est = embed_articles(chunks, active_keys)

    normalised = normalise_vectors(vectors_list)

    if faiss_available:
        duplicate_pairs, sim_time = find_similar_pairs_faiss(normalised)
        sim_method = "faiss"
    else:
        duplicate_pairs, sim_time = find_similar_pairs_numpy(normalised)
        sim_method = "numpy"

    t0 = time.time()
    clusters, article_sims, n_clusters, dup_groups, merged, reduction = cluster_articles(
        articles, duplicate_pairs
    )
    cluster_time = time.time() - t0

    t0 = time.time()
    distinct_articles = []
    dedup_report      = []
    cluster_counter   = 0

    for _root, cluster in clusters.items():
        result = select_dual_sources(cluster, article_sims)
        distinct_articles.append(result)

        if len(cluster) > 1:
            cluster_counter += 1
            meta = result["dedup_metadata"]
            dedup_report.append({
                "cluster_id":   cluster_counter,
                "cluster_size": len(cluster),
                "citation_source": {
                    "source":           result.get("source", ""),
                    "credibility_tier": result["_credibility_tier"],
                    "is_paywalled":     result["_is_paywalled"],
                    "citation_score":   meta["citation_score"],
                    "rss_url":          result.get("url", ""),
                },
                "content_source": {
                    "source":           result["content_source"]["source_name"],
                    "credibility_tier": result["content_source"]["credibility_tier"],
                    "is_paywalled":     result["content_source"]["is_paywalled"],
                    "content_score":    meta["content_score"],
                    "rss_url":          result["content_source"]["rss_url"],
                },
                "citation_differs_from_content": result["citation_differs_from_content"],
                "all_paywalled": meta["all_paywalled"],
                "all_articles": [
                    {
                        "source":    a.get("source", ""),
                        "tier":      a["_credibility_tier"],
                        "paywalled": a["_is_paywalled"],
                        "title":     a.get("title", ""),
                    }
                    for a in cluster
                ],
                "selection_reason": meta["selection_reason"],
            })

    selection_time = time.time() - t0

    # Sort by date descending before stripping
    distinct_articles.sort(key=_safe_date, reverse=True)

    # Compute stats while internal fields are still present
    n_input    = len(articles)
    n_output   = len(distinct_articles)
    n_differs  = sum(1 for a in distinct_articles if a.get("citation_differs_from_content"))
    n_fallback = sum(1 for a in distinct_articles if a.get("content_fallback_to_rss_summary"))
    n_had_free = sum(1 for a in distinct_articles
                     if a.get("dedup_metadata", {}).get("had_free_alternative"))
    n_single   = n_output - len(dedup_report)

    # Strip internal fields
    distinct_articles = [strip_internal(a) for a in distinct_articles]

    total_time = time.time() - run_start

    # ── Write output ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "distinct_articles.json").write_text(
        json.dumps(distinct_articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "dedup_report.json").write_text(
        json.dumps(dedup_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_summary = {
        "run_at":                          run_at,
        "input_articles":                  n_input,
        "distinct_articles":               n_output,
        "reduction_pct":                   round(reduction, 1),
        "duplicate_clusters":              dup_groups,
        "single_article_clusters":         n_single,
        "citation_differs_from_content":   n_differs,
        "all_paywalled_clusters":          n_fallback,
        "content_fallback_to_rss_summary": n_fallback,
        "had_free_alternative":            n_had_free,
        "similarity_method":               sim_method,
        "faiss_available":                 faiss_available,
        "embedding_keys_used":             len(active_keys),
        "embedding_batches":               total_batches,
        "embedding_time_seconds":          round(embed_time, 1),
        "similarity_time_seconds":         round(sim_time, 1),
        "clustering_time_seconds":         round(cluster_time, 1),
        "selection_time_seconds":          round(selection_time, 1),
        "total_time_seconds":              round(total_time, 1),
        "estimated_embedding_cost_usd":    round(cost_est, 4),
        "similarity_threshold":            SIMILARITY_THRESHOLD,
        "embedding_model":                 EMBEDDING_MODEL,
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Log file ──────────────────────────────────────────────────────────────
    n_files = sum(len(list(d.glob("*.json"))) for d in INPUT_DIRS if d.exists())
    tier_label_map = {
        5: "UAE Govt sources      ",
        4: "Major UAE sources     ",
        3: "Regional sources      ",
        2: "Intl Press sources    ",
        1: "Unknown/Tier 1 sources",
    }
    log = [
        "═" * 63,
        "  UAE AI NEWSLETTER — SEMANTIC DEDUPLICATION",
        f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "═" * 63,
        "",
        "  SETUP",
        "  ─────",
        f"  API keys active     : {len(active_keys)} (parallel embedding enabled)",
        f"  FAISS               : {'available (fast mode)' if faiss_available else 'not installed (numpy blocks)'}",
        "  Registry            : loaded",
        "",
        "  INPUT",
        "  ─────",
        f"  Sources             : news_output/english/  +  news_output/arabic_translated/",
        f"  Total articles      : {n_input:,}",
        f"  Files loaded        : {n_files}",
    ]
    for tier in sorted(tier_counts.keys(), reverse=True):
        label = tier_label_map.get(tier, f"Tier {tier}              ")
        log.append(f"  Tier {tier} ({label}): {tier_counts[tier]:>5,} articles")
    log.extend([
        f"  Paywalled articles  : {paywalled_count:>5,} articles",
        "",
        "  EMBEDDING",
        "  ─────────",
        f"  Model               : {EMBEDDING_MODEL}",
        f"  Batches             : {total_batches} ({EMBEDDING_BATCH_SIZE} articles each)",
        f"  Parallel keys       : {len(active_keys)}",
        f"  Time                : {embed_time:.1f} seconds",
        f"  Estimated cost      : ${cost_est:.4f}",
        "",
        "  SIMILARITY SEARCH",
        "  ─────────────────",
        f"  Method              : {'FAISS IndexFlatIP' if faiss_available else 'Numpy block matrix'}",
        f"  Threshold           : {SIMILARITY_THRESHOLD}",
        f"  Time                : {sim_time:.1f} seconds",
        f"  Pairs found         : {len(duplicate_pairs):,}",
        "",
        "  CLUSTERING",
        "  ──────────",
        "  Algorithm           : Union-Find (path compression + rank)",
        f"  Input articles      : {n_input:,}",
        f"  Distinct clusters   : {n_clusters:,}",
        f"  Duplicate groups    : {dup_groups:,}",
        f"  Articles merged     : {merged:,}",
        f"  Reduction           : {reduction:.1f}%",
        "",
        "  DUAL SOURCE SELECTION",
        "  ─────────────────────",
        f"  Single source       : {n_single:,}  (same article = citation and content)",
        f"  Split sources       : {n_differs:>5,}  (different articles for citation vs content)",
        "    → Prestigious paywalled cited, free RSS URL sent to Firecrawl",
        f"  All-paywalled       : {n_fallback:>5,}  (RSS summary fallback flagged)",
        "",
        "  OUTPUT",
        "  ──────",
        f"  distinct_articles.json   : {n_output:,} articles ← feed to LLM scorer",
        f"  dedup_report.json        : {dup_groups:,} cluster records",
        "  run_summary.json         : stats",
        "  dedup_log.txt            : this file",
        "",
        "  NEXT STEP",
        "  ─────────",
        "  Feed distinct_articles.json to LLM scorer (Step 5)",
        "  After scoring, top 20 per subcategory → Firecrawl (Step 7)",
        '  Firecrawl uses article["content_source"]["rss_url"] for each article',
        '  Newsletter displays article["source"] and article["url"] (citation fields)',
        "",
        f"  Total time          : {total_time:.1f} seconds",
        f"  Embedding cost      : ${cost_est:.4f}",
        "═" * 63,
    ])
    (OUTPUT_DIR / "dedup_log.txt").write_text("\n".join(log), encoding="utf-8")

    # ── Console final output ──────────────────────────────────────────────────
    print()
    _sep()
    print("  SEMANTIC DEDUP COMPLETE")
    _sep()
    print(f"  Input articles              : {n_input:,}")
    print(f"  Distinct stories            : {n_output:,}  ({reduction:.1f}% reduction)")
    print()
    print("  DUAL SOURCE BREAKDOWN")
    print("  ─────────────────────")
    print(f"  Same source (no split)      : {n_single:,}")
    print(f"  Split citation/content      : {n_differs:>5,}")
    print("    Prestigious paywalled cited, free RSS for Firecrawl")
    print(f"  All-paywalled (RSS fallback): {n_fallback:>5,}")
    print()
    print("  PERFORMANCE")
    print("  ───────────")
    print(f"  Embedding  : {embed_time:.1f}s  ({len(active_keys)} parallel key(s))")
    print(f"  Similarity : {sim_time:.1f}s   ({'FAISS' if faiss_available else 'NumPy'})")
    print(f"  Total      : {total_time:.1f}s")
    print(f"  Cost       : ${cost_est:.4f}")
    print()
    print(f"  OUTPUT → {OUTPUT_DIR}/")
    print("    distinct_articles.json  ← next: LLM scorer")
    print("    dedup_report.json")
    print("    run_summary.json")
    print("    dedup_log.txt")
    _sep()
    print()


if __name__ == "__main__":
    main()
