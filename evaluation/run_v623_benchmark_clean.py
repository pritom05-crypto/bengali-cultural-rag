import os
import sys
import json
import time
import math
import csv
import inspect
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

print("=" * 70)
print("Loading V6.26 retrieval engine...")
print("=" * 70)

import v6_rag

print("=" * 70)
print("V6.26 retrieval engine imported successfully.")
print("=" * 70)

print("\nV6.26 function signatures:")
print("correct_query:", inspect.signature(v6_rag.correct_query))
print("detect_concepts:", inspect.signature(v6_rag.detect_concepts))
print("expand_query:", inspect.signature(v6_rag.expand_query))
print("retrieve:", inspect.signature(v6_rag.retrieve))


# ============================================================
# PATHS
# ============================================================

BENCHMARK_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "benchmark_queries.json"
)

RESULT_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "benchmark_results_clean.json"
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "paper_evaluation_results.csv"
)

SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "evaluation_summary.json"
)

ERROR_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "error_analysis.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip()

    replacements = {
        "কুড়ের": "কুঁড়ের",
        "কুঁড়ের": "কুঁড়ের",
        "পড়েছে": "পড়েছে",
        "পড়ে": "পড়ে",
        "ভেংগে": "ভেঙ্গে",
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    return " ".join(text.split())


def split_acceptable(value):
    if value is None:
        return []

    if isinstance(value, list):
        items = value
    else:
        items = str(value).split("|")

    result = []

    for item in items:
        item = normalize_text(item)

        if not item:
            continue

        # slash-separated alternatives
        for part in item.split("/"):
            part = normalize_text(part)

            if part and part not in result:
                result.append(part)

    return result


def get_phrase(item):
    if isinstance(item, dict):

        for key in [
            "phrase",
            "expression",
            "idiom",
            "name",
            "title",
            "cultural_expression",
        ]:
            if key in item:
                value = item[key]

                if value is not None:
                    return normalize_text(value)

    return normalize_text(item)


def extract_results(raw):

    if raw is None:
        return []

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):

        for key in [
            "results",
            "candidates",
            "retrieved",
            "items",
            "top_results",
        ]:

            value = raw.get(key)

            if isinstance(value, list):
                return value

        # single result
        if any(
            key in raw
            for key in [
                "phrase",
                "expression",
                "idiom",
            ]
        ):
            return [raw]

    return []


# ============================================================
# CORRECT NDCG
# ============================================================

def dcg_at_k(relevances, k=5):

    relevances = list(relevances)[:k]

    score = 0.0

    for rank, rel in enumerate(relevances, start=1):

        rel = float(rel)

        gain = (2 ** rel) - 1

        discount = math.log2(rank + 1)

        score += gain / discount

    return score


def ndcg_at_k(relevances, k=5):

    relevances = list(relevances)[:k]

    if not relevances:
        return 0.0

    actual = dcg_at_k(relevances, k)

    ideal = sorted(
        relevances,
        reverse=True
    )

    ideal_score = dcg_at_k(
        ideal,
        k
    )

    if ideal_score <= 0:
        return 0.0

    value = actual / ideal_score

    return max(
        0.0,
        min(1.0, value)
    )


# ============================================================
# GOLD MATCH
# ============================================================

def rank_relevance(phrase, acceptable):

    phrase = normalize_text(phrase)

    acceptable_norm = [
        normalize_text(x)
        for x in acceptable
    ]

    if phrase in acceptable_norm:
        return 1.0

    return 0.0


# ============================================================
# MAIN
# ============================================================

print("\n")
print("=" * 75)
print("BENGALI CULTURAL RAG V6.26")
print("CLEAN SEMANTIC BENCHMARK")
print("REAL RETRIEVE() EVALUATION")
print("MULTI-ACCEPTABLE GOLD MATCHING")
print("CORRECT NDCG@5")
print("NO API / NO GEMINI / NO GROQ")
print("=" * 75)


# ============================================================
# LOAD BENCHMARK
# ============================================================

print("\nLoading benchmark...")

with open(
    BENCHMARK_PATH,
    "r",
    encoding="utf-8"
) as f:

    benchmark = json.load(f)


if isinstance(benchmark, dict):

    records = (
        benchmark.get("queries")
        or benchmark.get("benchmark")
        or benchmark.get("records")
        or []
    )

else:

    records = benchmark


verified = []

for record in records:

    if not isinstance(record, dict):
        continue

    if record.get("verified", True):
        verified.append(record)


print(
    f"Verified benchmark queries: {len(verified)}"
)


if not verified:

    raise RuntimeError(
        "No verified benchmark queries found."
    )


# ============================================================
# STORAGE
# ============================================================

all_results = []
errors = []

top1_hits = 0
top3_hits = 0
top5_hits = 0

mrr_values = []
ndcg_values = []

latencies = []
confidences = []


# ============================================================
# EVALUATION
# ============================================================

print("\n")
print("=" * 75)
print("STARTING REAL RETRIEVAL EVALUATION")
print("=" * 75)


for index, record in enumerate(
    verified,
    start=1
):

    query = (
        record.get("query")
        or record.get("natural_query")
        or ""
    ).strip()

    gold = normalize_text(
        record.get("gold_phrase")
        or record.get("gold")
        or record.get("phrase")
        or ""
    )

    acceptable = split_acceptable(
        record.get("acceptable")
        or record.get("acceptable_phrases")
        or record.get("acceptable_answers")
        or gold
    )

    if gold and gold not in acceptable:
        acceptable.insert(0, gold)

    print("\n")
    print("-" * 70)

    print(
        f"[{index}/{len(verified)}] {query}"
    )

    print(
        f"Gold: {gold}"
    )

    print(
        "Acceptable: "
        + " | ".join(acceptable)
    )

    start = time.perf_counter()

    try:

        corrected = v6_rag.correct_query(
            query
        )

        concepts = v6_rag.detect_concepts(
            corrected
        )

        expanded = v6_rag.expand_query(
            corrected,
            concepts
        )

        raw = v6_rag.retrieve(
            query,
            corrected,
            concepts,
            expanded
        )

        latency = (
            time.perf_counter()
            - start
        ) * 1000

    except Exception as e:

        latency = (
            time.perf_counter()
            - start
        ) * 1000

        print(
            f"RETRIEVAL ERROR: {e}"
        )

        errors.append({
            "query": query,
            "gold": gold,
            "error": str(e)
        })

        raw = []

    results = extract_results(raw)

    ranked = []

    for item in results:

        phrase = get_phrase(item)

        if not phrase:
            continue

        ranked.append({
            "phrase": phrase,
            "raw": item
        })

    top5 = ranked[:5]

    predicted = (
        top5[0]["phrase"]
        if top5
        else "NONE"
    )


    # ========================================================
    # RELEVANCE
    # ========================================================

    relevance = []

    for item in top5:

        phrase = item["phrase"]

        relevance.append(
            rank_relevance(
                phrase,
                acceptable
            )
        )


    # ========================================================
    # TOP K
    # ========================================================

    top1 = (
        1
        if any(
            rank_relevance(
                item["phrase"],
                acceptable
            ) > 0
            for item in ranked[:1]
        )
        else 0
    )

    top3 = (
        1
        if any(
            rank_relevance(
                item["phrase"],
                acceptable
            ) > 0
            for item in ranked[:3]
        )
        else 0
    )

    top5_hit = (
        1
        if any(
            rank_relevance(
                item["phrase"],
                acceptable
            ) > 0
            for item in ranked[:5]
        )
        else 0
    )


    # ========================================================
    # MRR
    # ========================================================

    reciprocal_rank = 0.0

    for rank, item in enumerate(
        ranked,
        start=1
    ):

        if rank_relevance(
            item["phrase"],
            acceptable
        ) > 0:

            reciprocal_rank = 1.0 / rank

            break


    # ========================================================
    # NDCG
    # ========================================================

    ndcg = ndcg_at_k(
        relevance,
        5
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = 0.0

    if isinstance(raw, dict):

        for key in [
            "confidence",
            "final_confidence",
            "score",
        ]:

            value = raw.get(key)

            if isinstance(
                value,
                (int, float)
            ):

                confidence = float(value)

                break


    if not confidence and ranked:

        item = ranked[0]["raw"]

        if isinstance(item, dict):

            for key in [
                "confidence",
                "final",
                "final_score",
                "score",
                "similarity",
            ]:

                value = item.get(key)

                if isinstance(
                    value,
                    (int, float)
                ):

                    confidence = float(value)

                    break


    confidence = max(
        0.0,
        min(1.0, confidence)
    )


    # ========================================================
    # UPDATE METRICS
    # ========================================================

    top1_hits += top1
    top3_hits += top3
    top5_hits += top5_hit

    mrr_values.append(
        reciprocal_rank
    )

    ndcg_values.append(
        ndcg
    )

    latencies.append(
        latency
    )

    confidences.append(
        confidence
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        f"Prediction: {predicted}"
    )

    print(
        f"Top-1: {top1} | "
        f"Top-3: {top3} | "
        f"Top-5: {top5_hit}"
    )

    print(
        f"MRR: {reciprocal_rank:.3f}"
    )

    print(
        f"NDCG@5: {ndcg:.3f}"
    )

    print(
        f"Latency: {latency:.2f} ms"
    )

    print(
        f"Confidence: {confidence:.3f}"
    )

    if top5:

        print(
            "Top-5: "
            + " | ".join(
                x["phrase"]
                for x in top5
            )
        )

    else:

        print(
            "Top-5: NONE"
        )


    # ========================================================
    # SAVE RECORD
    # ========================================================

    all_results.append({

        "index": index,

        "query": query,

        "corrected_query": corrected,

        "concepts": concepts,

        "gold": gold,

        "acceptable": acceptable,

        "prediction": predicted,

        "top1": top1,

        "top3": top3,

        "top5": top5_hit,

        "mrr": reciprocal_rank,

        "ndcg_at_5": ndcg,

        "latency_ms": latency,

        "confidence": confidence,

        "top5_results": [
            x["phrase"]
            for x in top5
        ]

    })


# ============================================================
# FINAL METRICS
# ============================================================

n = len(verified)

top1_accuracy = (
    top1_hits / n
)

top3_accuracy = (
    top3_hits / n
)

top5_accuracy = (
    top5_hits / n
)

mean_mrr = (
    float(np.mean(mrr_values))
    if mrr_values
    else 0.0
)

mean_ndcg = (
    float(np.mean(ndcg_values))
    if ndcg_values
    else 0.0
)

mean_latency = (
    float(np.mean(latencies))
    if latencies
    else 0.0
)

p95_latency = (
    float(np.percentile(
        latencies,
        95
    ))
    if latencies
    else 0.0
)

mean_confidence = (
    float(np.mean(confidences))
    if confidences
    else 0.0
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 75)
print("CLEAN BENCHMARK COMPLETE")
print("=" * 75)

print(
    f"Queries          : {n}"
)

print(
    f"Top-1 Accuracy   : {top1_accuracy:.4f}"
)

print(
    f"Top-3 Accuracy   : {top3_accuracy:.4f}"
)

print(
    f"Top-5 Accuracy   : {top5_accuracy:.4f}"
)

print(
    f"MRR              : {mean_mrr:.4f}"
)

print(
    f"NDCG@5           : {mean_ndcg:.4f}"
)

print(
    f"Mean Latency     : {mean_latency:.2f} ms"
)

print(
    f"P95 Latency      : {p95_latency:.2f} ms"
)

print(
    f"Mean Confidence  : {mean_confidence:.4f}"
)


# ============================================================
# SAVE JSON
# ============================================================

summary = {

    "system": "Bengali Cultural RAG V6.26",

    "dataset_records": 408,

    "benchmark_queries": n,

    "top1_accuracy": top1_accuracy,

    "top3_accuracy": top3_accuracy,

    "top5_accuracy": top5_accuracy,

    "mrr": mean_mrr,

    "ndcg_at_5": mean_ndcg,

    "mean_latency_ms": mean_latency,

    "p95_latency_ms": p95_latency,

    "mean_confidence": mean_confidence,

    "retriever": (
        "E5 + BM25 + Concept "
        "+ Meaning Grounding "
        "+ Primary Intent"
    ),

    "api_used": False,

    "results": all_results
}


with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            key: value
            for key, value in summary.items()
            if key != "results"
        },
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "query",
        "gold",
        "prediction",
        "top1",
        "top3",
        "top5",
        "mrr",
        "ndcg_at_5",
        "latency_ms",
        "confidence"
    ])

    for item in all_results:

        writer.writerow([
            item["query"],
            item["gold"],
            item["prediction"],
            item["top1"],
            item["top3"],
            item["top5"],
            f'{item["mrr"]:.6f}',
            f'{item["ndcg_at_5"]:.6f}',
            f'{item["latency_ms"]:.3f}',
            f'{item["confidence"]:.6f}'
        ])


# ============================================================
# ERROR ANALYSIS
# ============================================================

with open(
    ERROR_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "query",
        "gold",
        "prediction",
        "top1",
        "top3",
        "top5",
        "mrr",
        "error_type"
    ])

    for item in all_results:

        if item["top1"] == 1:

            error_type = "correct"

        elif item["top3"] == 1:

            error_type = "acceptable_in_top3"

        elif item["top5"] == 1:

            error_type = "acceptable_in_top5"

        elif item["prediction"] == "NONE":

            error_type = "no_reliable_candidate"

        else:

            error_type = "incorrect_retrieval"

        writer.writerow([
            item["query"],
            item["gold"],
            item["prediction"],
            item["top1"],
            item["top3"],
            item["top5"],
            f'{item["mrr"]:.6f}',
            error_type
        ])


print("\n")
print("Saved:")
print(RESULT_PATH)
print(CSV_PATH)
print(SUMMARY_PATH)
print(ERROR_PATH)

print("=" * 75)