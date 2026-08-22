import sys
import json
import time
import csv
from pathlib import Path

import numpy as np


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SCRIPTS_DIR = BASE_DIR / "scripts"

sys.path.insert(
    0,
    str(SCRIPTS_DIR)
)

import v6_rag


BENCHMARK_PATH = (
    BASE_DIR
    / "evaluation"
    / "benchmark_queries.json"
)

RESULT_JSON = (
    BASE_DIR
    / "evaluation"
    / "benchmark_results.json"
)

RESULT_CSV = (
    BASE_DIR
    / "evaluation"
    / "paper_evaluation_results.csv"
)

SUMMARY_JSON = (
    BASE_DIR
    / "evaluation"
    / "evaluation_summary.json"
)


# ============================================================
# METRICS
# ============================================================

def reciprocal_rank(
    predicted,
    gold
):

    for rank, phrase in enumerate(
        predicted,
        start=1
    ):

        if phrase == gold:

            return 1.0 / rank

    return 0.0


def hit_at_k(
    predicted,
    gold,
    k
):

    return int(
        gold in predicted[:k]
    )


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("BENGALI CULTURAL RAG V6.23")
print("REAL BENCHMARK EVALUATION")
print("=" * 70)

with open(
    BENCHMARK_PATH,
    "r",
    encoding="utf-8"
) as f:

    benchmark = json.load(f)


# ============================================================
# VERIFIED ONLY
# ============================================================

benchmark = [

    x for x in benchmark

    if x.get("verified") is True

    and str(
        x.get("query", "")
    ).strip()

]


print()
print(
    f"Verified benchmark queries: "
    f"{len(benchmark)}"
)


if not benchmark:

    raise RuntimeError(
        "No verified benchmark queries found."
    )


# ============================================================
# EVALUATION
# ============================================================

results = []


for counter, item in enumerate(
    benchmark,
    start=1
):

    query = str(
        item["query"]
    ).strip()

    gold = str(
        item["gold_phrase"]
    ).strip()

    difficulty = item.get(
        "difficulty",
        ""
    )

    print()
    print(
        f"[{counter}/{len(benchmark)}] "
        f"{query}"
    )

    print(
        f"Gold: {gold}"
    )

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # V6.23 pipeline
    # --------------------------------------------------------

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

    retrieved = v6_rag.retrieve(
        query,
        corrected,
        concepts,
        expanded
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = [

        str(
            x["phrase"]
        ).strip()

        for x in retrieved

    ]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    top1 = hit_at_k(
        predictions,
        gold,
        1
    )

    top3 = hit_at_k(
        predictions,
        gold,
        3
    )

    top5 = hit_at_k(
        predictions,
        gold,
        5
    )

    mrr = reciprocal_rank(
        predictions,
        gold
    )

    rank = (
        predictions.index(gold) + 1
        if gold in predictions
        else 0
    )

    answer = v6_rag.choose_answer(
        retrieved,
        concepts
    )

    answer_phrase = (
        answer["phrase"]
        if answer is not None
        else ""
    )

    confidence = (
        float(answer["final"])
        if answer is not None
        else 0.0
    )

    correct = int(
        answer_phrase == gold
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    row = {

        "query_id":
            counter,

        "query":
            query,

        "corrected_query":
            corrected,

        "gold_phrase":
            gold,

        "predicted_phrase":
            answer_phrase,

        "top1":
            top1,

        "top3":
            top3,

        "top5":
            top5,

        "mrr":
            mrr,

        "gold_rank":
            rank,

        "confidence":
            confidence,

        "latency_ms":
            elapsed * 1000,

        "difficulty":
            difficulty,

        "concepts":
            "|".join(
                concepts
            ),

        "num_candidates":
            len(retrieved),

        "retrieved_phrases":
            " | ".join(
                predictions[:5]
            ),

        "correct":
            correct
    }

    results.append(
        row
    )

    print(
        f"Prediction: "
        f"{answer_phrase or 'NONE'}"
    )

    print(
        f"Top-1: {top1} | "
        f"Top-3: {top3} | "
        f"Top-5: {top5} | "
        f"MRR: {mrr:.3f}"
    )


# ============================================================
# SUMMARY
# ============================================================

n = len(results)

top1_accuracy = (
    sum(x["top1"] for x in results)
    / n
)

top3_accuracy = (
    sum(x["top3"] for x in results)
    / n
)

top5_accuracy = (
    sum(x["top5"] for x in results)
    / n
)

mrr = (
    sum(x["mrr"] for x in results)
    / n
)

mean_latency = (
    sum(x["latency_ms"] for x in results)
    / n
)

latencies = sorted(
    x["latency_ms"]
    for x in results
)

p95_index = min(
    n - 1,
    int(
        0.95 * n
    )
)

p95_latency = latencies[
    p95_index
]

mean_confidence = (
    sum(
        x["confidence"]
        for x in results
    )
    / n
)

correct_count = sum(
    x["correct"]
    for x in results
)


# ============================================================
# ERROR ANALYSIS
# ============================================================

errors = [

    x for x in results

    if x["correct"] == 0

]


error_rows = []

for x in errors:

    if not x["predicted_phrase"]:

        error_type = (
            "NO_ANSWER"
        )

    elif x["gold_rank"] == 0:

        error_type = (
            "WRONG_RETRIEVAL"
        )

    else:

        error_type = (
            "RANKING_ERROR"
        )

    error_rows.append({

        "query_id":
            x["query_id"],

        "query":
            x["query"],

        "gold_phrase":
            x["gold_phrase"],

        "predicted_phrase":
            x["predicted_phrase"],

        "error_type":
            error_type,

        "confidence":
            x["confidence"],

        "difficulty":
            x["difficulty"],

        "retrieved_phrases":
            x["retrieved_phrases"]

    })


# ============================================================
# SAVE JSON
# ============================================================

summary = {

    "system":
        "Bengali Cultural RAG V6.23",

    "dataset_records":
        len(v6_rag.dataset),

    "benchmark_queries":
        n,

    "verified_queries":
        n,

    "top1_accuracy":
        top1_accuracy,

    "top3_accuracy":
        top3_accuracy,

    "top5_accuracy":
        top5_accuracy,

    "mrr":
        mrr,

    "mean_latency_ms":
        mean_latency,

    "p95_latency_ms":
        p95_latency,

    "mean_confidence":
        mean_confidence,

    "correct_predictions":
        correct_count,

    "errors":
        len(errors)

}


with open(
    RESULT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = list(
    results[0].keys()
)

with open(
    RESULT_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# ERROR CSV
# ============================================================

ERROR_CSV = (
    BASE_DIR
    / "evaluation"
    / "error_analysis.csv"
)

if error_rows:

    with open(
        ERROR_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                error_rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            error_rows
        )


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

print(
    f"Queries       : {n}"
)

print(
    f"Top-1 Accuracy: {top1_accuracy:.4f}"
)

print(
    f"Top-3 Accuracy: {top3_accuracy:.4f}"
)

print(
    f"Top-5 Accuracy: {top5_accuracy:.4f}"
)

print(
    f"MRR           : {mrr:.4f}"
)

print(
    f"Mean Latency  : {mean_latency:.2f} ms"
)

print(
    f"P95 Latency   : {p95_latency:.2f} ms"
)

print(
    f"Mean Confidence: {mean_confidence:.4f}"
)

print()
print("Saved:")

print(
    RESULT_JSON
)

print(
    RESULT_CSV
)

print(
    SUMMARY_JSON
)

if error_rows:

    print(
        ERROR_CSV
    )