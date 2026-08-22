"""
======================================================================
BENGALI CULTURAL RAG V6.26
FINAL PUBLICATION RESULT GENERATOR
======================================================================

Reads ONLY actual project outputs.

Inputs:
    evaluation/benchmark_results_clean.json
    evaluation/paper_evaluation_results.csv
    evaluation/evaluation_summary.json
    evaluation/error_analysis.csv
    data/processed/final_cultural_dataset.json

Outputs:
    figures/paper/final/
    tables/paper/

NO FABRICATED RESULTS.
======================================================================
"""

import os
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

FIG_DIR = os.path.join(BASE_DIR, "figures", "paper", "final")
TABLE_DIR = os.path.join(BASE_DIR, "tables", "paper")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

RESULT_JSON = os.path.join(
    EVAL_DIR,
    "benchmark_results_clean.json"
)

RESULT_CSV = os.path.join(
    EVAL_DIR,
    "paper_evaluation_results.csv"
)

SUMMARY_JSON = os.path.join(
    EVAL_DIR,
    "evaluation_summary.json"
)

ERROR_CSV = os.path.join(
    EVAL_DIR,
    "error_analysis.csv"
)

DATASET_JSON = os.path.join(
    DATA_DIR,
    "final_cultural_dataset.json"
)


# ----------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

# Try Bengali-compatible fonts

plt.rcParams["font.family"] = "DejaVu Sans"


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def save_fig(filename):
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def load_json(path):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_list_of_records(obj):
    """
    Try to locate a list of query/result dictionaries
    inside arbitrary JSON structures.
    """

    if isinstance(obj, list):
        if len(obj) > 0 and all(isinstance(x, dict) for x in obj):
            return obj

        for item in obj:
            result = find_list_of_records(item)
            if result is not None:
                return result

    elif isinstance(obj, dict):
        preferred_keys = [
            "results",
            "benchmark_results",
            "queries",
            "records",
            "items",
            "data",
        ]

        for key in preferred_keys:
            if key in obj:
                result = find_list_of_records(obj[key])
                if result is not None:
                    return result

        for value in obj.values():
            result = find_list_of_records(value)
            if result is not None:
                return result

    return None


def numeric_value(record, keys):
    for key in keys:
        if key in record:
            try:
                return float(record[key])
            except Exception:
                pass
    return None


def text_value(record, keys, default=""):
    for key in keys:
        if key in record and record[key] is not None:
            return str(record[key])
    return default


# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------

print("=" * 75)
print("BENGALI CULTURAL RAG V6.26")
print("FINAL PUBLICATION RESULT GENERATOR")
print("=" * 75)

print("\nLoading actual project results...")

summary = load_json(SUMMARY_JSON)
results_json = load_json(RESULT_JSON)

dataset = None

if os.path.exists(DATASET_JSON):
    dataset = load_json(DATASET_JSON)

csv_df = None

if os.path.exists(RESULT_CSV):
    try:
        csv_df = pd.read_csv(RESULT_CSV)
    except Exception as e:
        print("CSV could not be loaded:", e)


# ----------------------------------------------------------------------
# DATASET SUMMARY
# ----------------------------------------------------------------------

dataset_records = []

if isinstance(dataset, list):
    dataset_records = dataset

elif isinstance(dataset, dict):

    for key in [
        "records",
        "data",
        "dataset",
        "items",
    ]:
        if isinstance(dataset.get(key), list):
            dataset_records = dataset[key]
            break


dataset_count = len(dataset_records)

print(f"Dataset records detected: {dataset_count}")


# ----------------------------------------------------------------------
# BENCHMARK RECORDS
# ----------------------------------------------------------------------

records = find_list_of_records(results_json)

if records is None and csv_df is not None:
    records = csv_df.to_dict("records")

if records is None:
    records = []

print(f"Benchmark result records detected: {len(records)}")


# ----------------------------------------------------------------------
# EXTRACT QUERY METRICS
# ----------------------------------------------------------------------

query_rows = []

for i, rec in enumerate(records, start=1):

    query = text_value(
        rec,
        [
            "query",
            "original_query",
            "user_query",
        ],
        f"Query {i}"
    )

    gold = text_value(
        rec,
        [
            "gold",
            "gold_phrase",
            "expected",
            "target",
        ],
        ""
    )

    prediction = text_value(
        rec,
        [
            "prediction",
            "predicted",
            "answer",
            "phrase",
        ],
        ""
    )

    top1 = numeric_value(
        rec,
        ["top1", "top_1", "top1_correct"]
    )

    top3 = numeric_value(
        rec,
        ["top3", "top_3", "top3_correct"]
    )

    top5 = numeric_value(
        rec,
        ["top5", "top_5", "top5_correct"]
    )

    mrr = numeric_value(
        rec,
        ["mrr", "MRR"]
    )

    ndcg = numeric_value(
        rec,
        ["ndcg@5", "ndcg5", "NDCG@5", "ndcg"]
    )

    latency = numeric_value(
        rec,
        [
            "latency_ms",
            "latency",
            "Latency",
        ]
    )

    confidence = numeric_value(
        rec,
        [
            "confidence",
            "Confidence",
        ]
    )

    row = {
        "query_id": i,
        "query": query,
        "gold": gold,
        "prediction": prediction,
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "mrr": mrr,
        "ndcg5": ndcg,
        "latency_ms": latency,
        "confidence": confidence,
    }

    query_rows.append(row)


results_df = pd.DataFrame(query_rows)


# ----------------------------------------------------------------------
# IF SUMMARY EXISTS
# ----------------------------------------------------------------------

def summary_value(keys):

    if not isinstance(summary, dict):
        return None

    for key in keys:
        if key in summary:
            try:
                return float(summary[key])
            except Exception:
                pass

    # Nested summary
    for value in summary.values():
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    try:
                        return float(value[key])
                    except Exception:
                        pass

    return None


top1_summary = summary_value(
    ["top1_accuracy", "top1", "Top-1 Accuracy"]
)

top3_summary = summary_value(
    ["top3_accuracy", "top3", "Top-3 Accuracy"]
)

top5_summary = summary_value(
    ["top5_accuracy", "top5", "Top-5 Accuracy"]
)

mrr_summary = summary_value(
    ["mrr", "MRR"]
)

ndcg_summary = summary_value(
    ["ndcg5", "ndcg@5", "NDCG@5"]
)

latency_summary = summary_value(
    ["mean_latency", "mean_latency_ms", "Mean Latency"]
)

p95_summary = summary_value(
    ["p95_latency", "p95_latency_ms", "P95 Latency"]
)

confidence_summary = summary_value(
    ["mean_confidence", "Mean Confidence"]
)


# ----------------------------------------------------------------------
# FIGURE 1 — SYSTEM PERFORMANCE
# ----------------------------------------------------------------------

metrics = {
    "Top-1": top1_summary,
    "Top-3": top3_summary,
    "Top-5": top5_summary,
    "MRR": mrr_summary,
    "NDCG@5": ndcg_summary,
}

metrics = {
    k: v for k, v in metrics.items()
    if v is not None
}

if metrics:

    plt.figure(figsize=(8, 5))

    names = list(metrics.keys())
    values = [metrics[x] * 100 for x in names]

    bars = plt.bar(names, values)

    plt.ylabel("Score (%)")
    plt.xlabel("Evaluation Metric")
    plt.title("Retrieval Performance of Bengali Cultural RAG V6.26")
    plt.ylim(0, 105)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_fig("08_retrieval_performance.png")


# ----------------------------------------------------------------------
# FIGURE 2 — LATENCY
# ----------------------------------------------------------------------

if len(results_df) > 0:

    latency = results_df["latency_ms"].dropna()

    if len(latency) > 0:

        plt.figure(figsize=(8, 5))

        plt.hist(
            latency,
            bins=min(8, max(3, len(latency) // 2)),
        )

        plt.axvline(
            latency.mean(),
            linestyle="--",
            linewidth=2,
            label=f"Mean = {latency.mean():.2f} ms"
        )

        plt.xlabel("Latency (ms)")
        plt.ylabel("Number of Queries")
        plt.title("Query Latency Distribution")
        plt.legend()

        save_fig("09_latency_distribution.png")


# ----------------------------------------------------------------------
# FIGURE 3 — CONFIDENCE
# ----------------------------------------------------------------------

if len(results_df) > 0:

    confidence = results_df["confidence"].dropna()

    if len(confidence) > 0:

        plt.figure(figsize=(8, 5))

        x = np.arange(1, len(confidence) + 1)

        plt.plot(
            x,
            confidence,
            marker="o",
            linewidth=2,
        )

        plt.axhline(
            confidence.mean(),
            linestyle="--",
            linewidth=2,
            label=f"Mean = {confidence.mean():.3f}"
        )

        plt.xlabel("Benchmark Query")
        plt.ylabel("Confidence")
        plt.title("Prediction Confidence Across Benchmark Queries")
        plt.ylim(0, 1.05)
        plt.legend()

        save_fig("10_confidence_by_query.png")


# ----------------------------------------------------------------------
# FIGURE 4 — PER-QUERY RETRIEVAL
# ----------------------------------------------------------------------

if len(results_df) > 0:

    valid_top1 = results_df["top1"].fillna(0)
    valid_top3 = results_df["top3"].fillna(0)
    valid_top5 = results_df["top5"].fillna(0)

    x = np.arange(len(results_df))

    width = 0.25

    plt.figure(figsize=(12, 5))

    plt.bar(
        x - width,
        valid_top1,
        width,
        label="Top-1"
    )

    plt.bar(
        x,
        valid_top3,
        width,
        label="Top-3"
    )

    plt.bar(
        x + width,
        valid_top5,
        width,
        label="Top-5"
    )

    plt.xlabel("Benchmark Query")
    plt.ylabel("Correct Retrieval")
    plt.title("Per-Query Retrieval Success")
    plt.xticks(
        x,
        [str(i + 1) for i in range(len(results_df))]
    )
    plt.yticks([0, 1], ["Incorrect", "Correct"])
    plt.legend()

    save_fig("11_per_query_retrieval.png")


# ----------------------------------------------------------------------
# FIGURE 5 — MRR / NDCG PER QUERY
# ----------------------------------------------------------------------

if len(results_df) > 0:

    if (
        results_df["mrr"].notna().any()
        or results_df["ndcg5"].notna().any()
    ):

        plt.figure(figsize=(10, 5))

        x = np.arange(len(results_df))

        if results_df["mrr"].notna().any():

            plt.plot(
                x,
                results_df["mrr"],
                marker="o",
                linewidth=2,
                label="MRR"
            )

        if results_df["ndcg5"].notna().any():

            plt.plot(
                x,
                results_df["ndcg5"],
                marker="s",
                linewidth=2,
                label="NDCG@5"
            )

        plt.xlabel("Benchmark Query")
        plt.ylabel("Ranking Score")
        plt.title("Ranking Quality Across Benchmark Queries")
        plt.xticks(
            x,
            [str(i + 1) for i in range(len(results_df))]
        )
        plt.ylim(0, 1.05)
        plt.legend()

        save_fig("12_ranking_quality.png")


# ----------------------------------------------------------------------
# FIGURE 6 — LATENCY VS CONFIDENCE
# ----------------------------------------------------------------------

if len(results_df) > 0:

    temp = results_df.dropna(
        subset=["latency_ms", "confidence"]
    )

    if len(temp) > 0:

        plt.figure(figsize=(7, 5))

        plt.scatter(
            temp["latency_ms"],
            temp["confidence"],
            s=60,
        )

        plt.xlabel("Latency (ms)")
        plt.ylabel("Confidence")
        plt.title("Latency–Confidence Relationship")

        save_fig("13_latency_vs_confidence.png")


# ----------------------------------------------------------------------
# FIGURE 7 — DATASET SIZE
# ----------------------------------------------------------------------

if dataset_count > 0:

    plt.figure(figsize=(6, 5))

    plt.bar(
        ["Cultural Dataset"],
        [dataset_count]
    )

    plt.ylabel("Number of Records")
    plt.title("Final Bengali Cultural Dataset")
    plt.text(
        0,
        dataset_count,
        str(dataset_count),
        ha="center",
        va="bottom",
        fontsize=14,
    )

    save_fig("14_dataset_size.png")


# ----------------------------------------------------------------------
# FIGURE 8 — ERROR ANALYSIS
# ----------------------------------------------------------------------

if len(results_df) > 0:

    errors = results_df[
        results_df["top1"].fillna(0) == 0
    ]

    successes = results_df[
        results_df["top1"].fillna(0) == 1
    ]

    values = [
        len(successes),
        len(errors)
    ]

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        ["Top-1 Correct", "Top-1 Error"],
        values
    )

    plt.ylabel("Number of Queries")
    plt.title("Top-1 Retrieval Error Analysis")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.1,
            str(value),
            ha="center",
            va="bottom"
        )

    save_fig("15_error_analysis.png")


# ----------------------------------------------------------------------
# TABLE 1 — OVERALL METRICS
# ----------------------------------------------------------------------

overall_table = pd.DataFrame({
    "Metric": [
        "Dataset Records",
        "Benchmark Queries",
        "Top-1 Accuracy",
        "Top-3 Accuracy",
        "Top-5 Accuracy",
        "MRR",
        "NDCG@5",
        "Mean Confidence",
        "Mean Latency (ms)",
        "P95 Latency (ms)",
    ],

    "Value": [
        dataset_count,
        len(results_df),
        top1_summary,
        top3_summary,
        top5_summary,
        mrr_summary,
        ndcg_summary,
        confidence_summary,
        latency_summary,
        p95_summary,
    ]
})

overall_table.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_1_overall_performance.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print(
    "[OK] table_1_overall_performance.csv"
)


# ----------------------------------------------------------------------
# TABLE 2 — QUERY-LEVEL RESULTS
# ----------------------------------------------------------------------

if len(results_df) > 0:

    query_table = results_df[
        [
            "query_id",
            "query",
            "gold",
            "prediction",
            "top1",
            "top3",
            "top5",
            "mrr",
            "ndcg5",
            "latency_ms",
            "confidence",
        ]
    ].copy()

    query_table.to_csv(
        os.path.join(
            TABLE_DIR,
            "table_2_query_level_results.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "[OK] table_2_query_level_results.csv"
    )


# ----------------------------------------------------------------------
# TABLE 3 — LATENCY SUMMARY
# ----------------------------------------------------------------------

if len(results_df) > 0:

    latency = results_df["latency_ms"].dropna()

    if len(latency) > 0:

        latency_table = pd.DataFrame({
            "Statistic": [
                "Mean",
                "Median",
                "Minimum",
                "Maximum",
                "P95",
            ],

            "Latency_ms": [
                latency.mean(),
                latency.median(),
                latency.min(),
                latency.max(),
                np.percentile(latency, 95),
            ]
        })

        latency_table.to_csv(
            os.path.join(
                TABLE_DIR,
                "table_3_latency_statistics.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "[OK] table_3_latency_statistics.csv"
        )


# ----------------------------------------------------------------------
# SAVE DATASET SUMMARY
# ----------------------------------------------------------------------

dataset_summary = {
    "dataset_records": dataset_count,
    "benchmark_queries": len(results_df),
    "generated_from_actual_results": True,
}

with open(
    os.path.join(
        TABLE_DIR,
        "dataset_summary.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dataset_summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ----------------------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------------------

print("\n" + "=" * 75)
print("FINAL PUBLICATION FIGURE GENERATION COMPLETE")
print("=" * 75)

print("\nFigures:")
print(FIG_DIR)

print("\nTables:")
print(TABLE_DIR)

print("\nNo fabricated results were generated.")

if top1_summary is not None:
    print(f"\nTop-1 Accuracy : {top1_summary:.4f}")

if top3_summary is not None:
    print(f"Top-3 Accuracy : {top3_summary:.4f}")

if top5_summary is not None:
    print(f"Top-5 Accuracy : {top5_summary:.4f}")

if mrr_summary is not None:
    print(f"MRR            : {mrr_summary:.4f}")

if ndcg_summary is not None:
    print(f"NDCG@5         : {ndcg_summary:.4f}")

if latency_summary is not None:
    print(f"Mean Latency   : {latency_summary:.2f} ms")

if p95_summary is not None:
    print(f"P95 Latency    : {p95_summary:.2f} ms")

if confidence_summary is not None:
    print(f"Mean Confidence: {confidence_summary:.4f}")

print("\nDONE")