"""
generate_paper_figures.py
=========================
Generate publication figures DIRECTLY from the Bengali Cultural RAG project.

Project root:
    E:\bengali-cultural-rag

The script reads project files; it does NOT invent experimental numbers.

Generated figures:
01_system_architecture.png
02_dataset_expression_type.png
03_dataset_field_completeness.png
04_phrase_length_distribution.png
05_tone_distribution.png
06_validation_decisions.png          (only if validation report exists)
07_topk_performance.png              (only if evaluation CSV exists)
08_main_metrics.png                  (only if evaluation CSV exists)
09_latency_distribution.png          (only if evaluation CSV exists)
10_confidence_distribution.png       (only if evaluation CSV exists)
11_query_category_accuracy.png      (only if evaluation CSV exists)
12_error_breakdown.png              (only if evaluation CSV exists)

Optional evaluation file:
    evaluation/paper_evaluation_results.csv

IMPORTANT:
- Dataset figures come from final_cultural_dataset.json.
- Validation figure comes from groq_validation_report.json if present.
- Performance figures come ONLY from actual evaluation CSV.
- No fake accuracy, MRR, Recall, etc. are inserted.
"""

from pathlib import Path
import json
import csv
import re
import math
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_CANDIDATES = [
    ROOT / "data" / "processed" / "final_cultural_dataset.json",
    ROOT / "data" / "final_cultural_dataset.json",
]

VALIDATION_CANDIDATES = [
    ROOT / "data" / "processed" / "groq_validation_report.json",
]

EVAL_CANDIDATES = [
    ROOT / "evaluation" / "paper_evaluation_results.csv",
    ROOT / "data" / "processed" / "paper_evaluation_results.csv",
]

FIG_DIR = ROOT / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def text(v):
    if v is None:
        return ""
    return str(v).strip()


def safe_list(v):
    if isinstance(v, list):
        return v
    if v in (None, ""):
        return []
    return [v]


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def clean_label(x, max_len=35):
    x = text(x)
    return x if len(x) <= max_len else x[:max_len - 1] + "…"


def setup():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    path = first_existing(DATASET_CANDIDATES)

    if path is None:
        raise FileNotFoundError(
            "final_cultural_dataset.json not found.\n"
            "Expected: data/processed/final_cultural_dataset.json"
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list of records.")

    print(f"\nDataset loaded: {path}")
    print(f"Records: {len(data)}")
    return data, path


# ============================================================
# FIGURE 1 — SYSTEM ARCHITECTURE
# ============================================================

def architecture():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, title, body=""):
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.4,
        )
        ax.add_patch(p)
        ax.text(
            x + w/2, y + h*0.68, title,
            ha="center", va="center",
            fontweight="bold", fontsize=10.5
        )
        if body:
            ax.text(
                x + w/2, y + h*0.30, body,
                ha="center", va="center",
                fontsize=8.2
            )

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->",
            mutation_scale=13,
            linewidth=1.2
        ))

    ax.text(
        8, 7.55,
        "Bengali Cultural RAG V6.23 — Proposed System Architecture",
        ha="center", fontsize=16, fontweight="bold"
    )

    box(0.4, 5.8, 2.1, 1.0, "User Query",
        "Natural Bengali sentence")
    box(3.0, 5.8, 2.2, 1.0, "Pre-processing",
        "Correction • normalization")
    box(5.7, 6.15, 2.3, 1.0, "Concept Detection",
        "Intent / semantic concepts")
    box(5.7, 4.8, 2.3, 1.0, "Query Expansion",
        "Related Bengali terms")
    box(8.6, 5.8, 2.3, 1.0, "E5 Retrieval",
        "Semantic similarity")
    box(8.6, 4.15, 2.3, 1.0, "BM25 Retrieval",
        "Lexical matching")
    box(11.8, 5.0, 2.8, 1.45, "Meaning-Grounded Reranking",
        "Semantic + BM25 + Concept\n+ Meaning + Pattern + Severity")
    box(11.8, 2.85, 2.8, 1.2, "Safety / Decision Gate",
        "Confidence + threshold\n+ no-answer handling")
    box(3.0, 1.85, 3.0, 1.3, "Cultural Dataset",
        "408 records\nPhrase + meaning + tone + variants")
    box(7.8, 1.65, 2.8, 1.3, "Top-K Evidence",
        "Ranked candidates + scores")
    box(11.8, 1.0, 2.8, 1.2, "Final Response",
        "Expression + meaning + tone\nor safe abstention")

    arrow(2.5, 6.3, 3.0, 6.3)
    arrow(5.2, 6.3, 5.7, 6.65)
    arrow(5.2, 6.0, 5.7, 5.3)
    arrow(8.0, 6.65, 8.6, 6.3)
    arrow(8.0, 5.3, 8.6, 4.65)
    arrow(10.9, 6.3, 11.8, 5.8)
    arrow(10.9, 4.65, 11.8, 5.35)
    arrow(13.2, 5.0, 13.2, 4.05)
    arrow(13.2, 2.85, 13.2, 2.2)
    arrow(6.0, 2.5, 7.8, 2.3)
    arrow(9.2, 2.95, 11.8, 3.45)
    arrow(4.5, 3.15, 5.7, 4.8)
    arrow(4.5, 3.15, 8.6, 5.0)

    ax.text(
        8, 0.35,
        "Fully local retrieval pipeline • E5 + BM25 + Concept + Meaning Grounding",
        ha="center", fontsize=10.5, fontweight="bold"
    )

    save(fig, "01_system_architecture.png")


# ============================================================
# FIGURE 2 — EXPRESSION TYPE
# ============================================================

def expression_type(data):
    keys = ["expression_type", "type", "category", "expression_category"]
    counter = Counter()

    for r in data:
        value = ""
        for k in keys:
            if text(r.get(k)):
                value = text(r.get(k))
                break
        if not value:
            value = "Unspecified"
        counter[value] += 1

    labels = list(counter.keys())
    values = list(counter.values())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values)
    ax.set_title("Distribution of Bengali Cultural Expression Types")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", rotation=30)

    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width()/2,
            v,
            str(v),
            ha="center", va="bottom", fontsize=9
        )

    save(fig, "02_dataset_expression_type.png")


# ============================================================
# FIGURE 3 — FIELD COMPLETENESS
# ============================================================

def field_completeness(data):
    fields = [
        "bengali_phrase",
        "variants",
        "literal_meaning",
        "cultural_meaning",
        "intended_emotion_tone",
        "primary_tone",
        "secondary_tone",
        "expression_type",
    ]

    labels = []
    percentages = []

    for field in fields:
        count = 0
        for r in data:
            v = r.get(field)
            if isinstance(v, list):
                ok = len(v) > 0
            else:
                ok = bool(text(v))
            count += int(ok)

        labels.append(field.replace("_", " ").title())
        percentages.append(100 * count / len(data))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(labels, percentages)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Completeness (%)")
    ax.set_title("Dataset Field Completeness")

    for b, v in zip(bars, percentages):
        ax.text(
            min(v + 1, 99), b.get_y() + b.get_height()/2,
            f"{v:.1f}%", va="center", fontsize=9
        )

    save(fig, "03_dataset_field_completeness.png")


# ============================================================
# FIGURE 4 — PHRASE LENGTH
# ============================================================

def phrase_length(data):
    lengths = []

    for r in data:
        phrase = text(r.get("bengali_phrase"))
        if phrase:
            # Bengali whitespace-based word count
            lengths.append(len(phrase.split()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=range(1, max(lengths) + 2), rwidth=0.85)
    ax.set_title("Distribution of Bengali Expression Length")
    ax.set_xlabel("Number of words")
    ax.set_ylabel("Number of records")

    save(fig, "04_phrase_length_distribution.png")


# ============================================================
# FIGURE 5 — TONE DISTRIBUTION
# ============================================================

def tone_distribution(data):
    counter = Counter()

    for r in data:
        tone = text(r.get("primary_tone"))
        if not tone:
            tone = text(r.get("intended_emotion_tone"))
        if not tone:
            tone = "Unspecified"
        counter[tone] += 1

    # Keep graph readable: top 12 only, but save full counts separately.
    all_counts = dict(counter.most_common())

    with open(FIG_DIR / "tone_distribution_full.json", "w", encoding="utf-8") as f:
        json.dump(all_counts, f, ensure_ascii=False, indent=2)

    top = counter.most_common(12)
    labels = [clean_label(x[0], 28) for x in top]
    values = [x[1] for x in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], values[::-1])
    ax.set_title("Most Frequent Primary/Intended Tone Categories")
    ax.set_xlabel("Number of records")

    save(fig, "05_tone_distribution.png")


# ============================================================
# FIGURE 6 — VALIDATION DECISIONS
# ============================================================

def validation_decisions():
    path = first_existing(VALIDATION_CANDIDATES)
    if path is None:
        print("\nValidation report not found — skipping Figure 6.")
        return

    with open(path, encoding="utf-8") as f:
        obj = json.load(f)

    records = obj
    if isinstance(obj, dict):
        # Support common report structures.
        for key in ["results", "validation_results", "records", "report"]:
            if isinstance(obj.get(key), list):
                records = obj[key]
                break

    counter = Counter()
    if isinstance(records, list):
        for r in records:
            d = text(r.get("decision")).upper()
            if d:
                counter[d] += 1

    if not counter:
        print("No decision records found — skipping Figure 6.")
        return

    labels = ["KEEP", "FIX", "REVIEW"]
    values = [counter.get(x, 0) for x in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values)
    ax.set_title("Dataset Validation Decisions")
    ax.set_ylabel("Number of records")

    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width()/2,
            v,
            str(v),
            ha="center", va="bottom"
        )

    save(fig, "06_validation_decisions.png")


# ============================================================
# LOAD REAL EVALUATION RESULTS
# ============================================================

def load_eval():
    path = first_existing(EVAL_CANDIDATES)
    if path is None:
        print("\nNo paper_evaluation_results.csv found.")
        print("Performance figures will be skipped.")
        print("Run the benchmark evaluator after creating the annotated test set.")
        return None

    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"\nEvaluation results loaded: {path}")
    print(f"Evaluation queries: {len(rows)}")
    return rows


def num(r, key):
    try:
        return float(r.get(key, 0))
    except Exception:
        return 0.0


# ============================================================
# FIGURE 7 — TOP-K
# ============================================================

def topk(rows):
    ans = [r for r in rows if int(num(r, "answerable")) == 1]
    if not ans:
        return

    vals = [
        100 * sum(int(num(r, "top1_correct")) for r in ans) / len(ans),
        100 * sum(int(num(r, "top3_hit")) for r in ans) / len(ans),
        100 * sum(int(num(r, "top5_hit")) for r in ans) / len(ans),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Top-1", "Top-3", "Top-5"], vals)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Retrieval Performance at K")

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width()/2,
            v + 1,
            f"{v:.2f}%",
            ha="center", fontweight="bold"
        )

    save(fig, "07_topk_performance.png")


# ============================================================
# FIGURE 8 — MAIN METRICS
# ============================================================

def main_metrics(rows):
    ans = [r for r in rows if int(num(r, "answerable")) == 1]
    if not ans:
        return

    metrics = {
        "Top-1": sum(int(num(r, "top1_correct")) for r in ans) / len(ans),
        "MRR": sum(num(r, "reciprocal_rank") for r in ans) / len(ans),
        "Recall@5": sum(int(num(r, "top5_hit")) for r in ans) / len(ans),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(list(metrics.keys()), list(metrics.values()))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Main Retrieval Metrics")

    for b, v in zip(bars, metrics.values()):
        ax.text(
            b.get_x() + b.get_width()/2,
            v + 0.02,
            f"{v:.3f}",
            ha="center", fontweight="bold"
        )

    save(fig, "08_main_metrics.png")


# ============================================================
# FIGURE 9 — LATENCY
# ============================================================

def latency(rows):
    values = [num(r, "latency_ms") for r in rows if num(r, "latency_ms") > 0]
    if not values:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=20)
    ax.set_title("Query Processing Latency")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Number of queries")

    save(fig, "09_latency_distribution.png")


# ============================================================
# FIGURE 10 — CONFIDENCE
# ============================================================

def confidence(rows):
    values = [num(r, "confidence") for r in rows]
    if not values:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=20, range=(0, 1))
    ax.set_title("Final Confidence Score Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Number of queries")

    save(fig, "10_confidence_distribution.png")


# ============================================================
# FIGURE 11 — CATEGORY ACCURACY
# ============================================================

def category_accuracy(rows):
    groups = defaultdict(list)

    for r in rows:
        if int(num(r, "answerable")) == 1:
            groups[text(r.get("category")) or "Unknown"].append(
                int(num(r, "top1_correct"))
            )

    if not groups:
        return

    labels = list(groups.keys())
    vals = [
        100 * sum(groups[k]) / len(groups[k])
        for k in labels
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, vals)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Top-1 Accuracy by Query Category")
    ax.tick_params(axis="x", rotation=30)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width()/2,
            v + 1,
            f"{v:.1f}%",
            ha="center", fontsize=8
        )

    save(fig, "11_query_category_accuracy.png")


# ============================================================
# FIGURE 12 — ERROR BREAKDOWN
# ============================================================

def error_breakdown(rows):
    ans = [r for r in rows if int(num(r, "answerable")) == 1]
    if not ans:
        return

    errors = Counter()

    for r in ans:
        if int(num(r, "top1_correct")) == 1:
            continue

        rank = num(r, "rank_of_gold")

        if rank and rank <= 5:
            errors["Gold in Top-5\n(not Top-1)"] += 1
        elif text(r.get("predicted_phrase")):
            errors["Wrong expression"] += 1
        else:
            errors["Incorrect abstention"] += 1

    if not errors:
        print("No Top-1 errors found — skipping error figure.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(list(errors.keys()), list(errors.values()))
    ax.set_title("Top-1 Error Breakdown")
    ax.set_ylabel("Number of queries")

    for b, v in zip(bars, errors.values()):
        ax.text(
            b.get_x() + b.get_width()/2,
            v,
            str(v),
            ha="center", va="bottom"
        )

    save(fig, "12_error_breakdown.png")


# ============================================================
# DATASET SUMMARY
# ============================================================

def write_dataset_summary(data, path):
    summary = {
        "dataset_path": str(path),
        "record_count": len(data),
        "fields": sorted({
            k for r in data for k in r.keys()
        }),
    }

    variants = 0
    for r in data:
        variants += len(safe_list(r.get("variants")))

    summary["total_variant_entries"] = variants

    with open(FIG_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDataset summary saved.")


# ============================================================
# MAIN
# ============================================================

def main():
    setup()

    print("=" * 72)
    print("BENGALI CULTURAL RAG V6.23 — PAPER FIGURE GENERATOR")
    print("PROJECT-DATASET-FIRST / NO FABRICATED RESULTS")
    print("=" * 72)

    data, dataset_path = load_dataset()

    write_dataset_summary(data, dataset_path)

    print("\nGenerating dataset figures...")
    architecture()
    expression_type(data)
    field_completeness(data)
    phrase_length(data)
    tone_distribution(data)
    validation_decisions()

    print("\nChecking for real evaluation results...")
    rows = load_eval()

    if rows:
        topk(rows)
        main_metrics(rows)
        latency(rows)
        confidence(rows)
        category_accuracy(rows)
        error_breakdown(rows)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Figures directory:\n{FIG_DIR}")
    print("\nOnly figures supported by actual project files were generated.")


if __name__ == "__main__":
    main()