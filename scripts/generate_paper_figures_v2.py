"""
generate_paper_figures_v2.py
Bengali Cultural RAG V6.23
---------------------------------
Generates publication-ready figures directly from the project files.

Run from:
    E:\bengali-cultural-rag

Command:
    python scripts\generate_paper_figures_v2.py

The script NEVER invents evaluation numbers.
Dataset figures are generated from:
    data/processed/final_cultural_dataset.json

Validation figures are generated from any available:
    groq_validation_report.json
    audit_report.json
    final_review_log.json
    review_queue.json

Performance figures are generated only when:
    evaluation/paper_evaluation_results.csv
exists.

Output:
    figures/paper/
"""

from pathlib import Path
import json
import csv
from collections import Counter, defaultdict
import statistics

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "processed" / "final_cultural_dataset.json"
PROCESSED = ROOT / "data" / "processed"
EVAL = ROOT / "evaluation" / "paper_evaluation_results.csv"

OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def save(fig, filename):
    path = OUT / filename
    fig.savefig(
        path,
        dpi=400,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)
    print(f"[OK] {filename}")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def txt(v):
    return "" if v is None else str(v).strip()


def has_value(v):
    if isinstance(v, list):
        return len(v) > 0
    return bool(txt(v))


def number(v):
    try:
        return float(v)
    except Exception:
        return 0.0


# ============================================================
# LOAD DATASET
# ============================================================

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET}"
    )

dataset = load_json(DATASET)

print("=" * 75)
print("BENGALI CULTURAL RAG V6.23")
print("PUBLICATION FIGURE GENERATOR V2")
print("=" * 75)
print(f"Dataset: {DATASET}")
print(f"Records: {len(dataset)}")


# ============================================================
# FIGURE 1 — PROFESSIONAL SYSTEM ARCHITECTURE
# ============================================================

def create_architecture():

    fig, ax = plt.subplots(figsize=(17, 9))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ---- title ----
    ax.text(
        8.5, 9.55,
        "Proposed Bengali Cultural RAG V6.23 Architecture",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold"
    )

    ax.text(
        8.5, 9.05,
        "Dataset-first • Fully local • E5 + BM25 • Meaning-grounded reranking • Safe abstention",
        ha="center",
        fontsize=10.5
    )

    # ---- colors ----
    C = {
        "input": "#DCEEFF",
        "pre": "#E9E2FF",
        "retrieval": "#DDF4E4",
        "ground": "#FFF0C9",
        "decision": "#FFE0E0",
        "output": "#E2F4F5",
        "data": "#F1F1F1",
    }

    def box(x, y, w, h, title, body, color):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.4,
            edgecolor="#333333",
            facecolor=color
        )
        ax.add_patch(patch)

        ax.text(
            x + w / 2,
            y + h * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold"
        )

        ax.text(
            x + w / 2,
            y + h * 0.30,
            body,
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.35
        )

    def arrow(x1, y1, x2, y2, curved=False):
        style = "Simple,tail_width=0.6,head_width=7,head_length=8"
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=style,
                linewidth=0.8,
                color="#444444",
                connectionstyle="arc3,rad=0.0"
            )
        )

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    box(
        0.35, 6.55, 2.15, 1.25,
        "1. User Query",
        "Natural Bengali sentence\n(e.g., \"সে খুব অলস\")",
        C["input"]
    )

    box(
        2.95, 6.55, 2.35, 1.25,
        "2. Pre-processing",
        "Spelling correction\nNormalization",
        C["pre"]
    )

    box(
        5.80, 7.05, 2.35, 1.25,
        "3A. Concept Detection",
        "Intent / concept\nidentification",
        C["pre"]
    )

    box(
        5.80, 5.35, 2.35, 1.25,
        "3B. Query Expansion",
        "Synonyms • variants\nrelated concepts",
        C["pre"]
    )

    box(
        8.75, 7.05, 2.35, 1.25,
        "4A. E5 Retrieval",
        "Semantic similarity\nmeaning-level evidence",
        C["retrieval"]
    )

    box(
        8.75, 5.35, 2.35, 1.25,
        "4B. BM25 Retrieval",
        "Lexical overlap\nexact/near-exact evidence",
        C["retrieval"]
    )

    box(
        11.65, 6.15, 2.75, 1.55,
        "5. Meaning-Grounded\nReranking",
        "Semantic + BM25\nConcept + Meaning\nPattern + Severity",
        C["ground"]
    )

    box(
        11.65, 3.95, 2.75, 1.35,
        "6. Safety / Decision Gate",
        "Confidence threshold\nEvidence + margin",
        C["decision"]
    )

    box(
        14.65, 4.75, 2.0, 1.35,
        "7. Final Answer",
        "Expression\nMeaning + Tone",
        C["output"]
    )

    # ---- abstention path ----
    box(
        14.65, 2.35, 2.0, 1.35,
        "Safe Abstention",
        "No reliable\ncandidate found",
        C["decision"]
    )

    # ---- arrows main ----
    arrow(2.50, 7.18, 2.95, 7.18)
    arrow(5.30, 7.18, 5.80, 7.65)
    arrow(5.30, 7.18, 5.80, 5.95)

    arrow(8.15, 7.65, 8.75, 7.65)
    arrow(8.15, 5.95, 8.75, 5.95)

    arrow(11.10, 7.65, 11.65, 7.0)
    arrow(11.10, 5.95, 11.65, 6.8)

    arrow(13.02, 6.15, 13.02, 5.30)
    arrow(14.40, 4.62, 14.65, 5.42)

    # ---- safety split ----
    arrow(14.40, 4.35, 14.65, 3.05)

    # ========================================================
    # DATASET BLOCK
    # ========================================================

    box(
        2.8, 1.15, 5.0, 2.15,
        "Bengali Cultural Dataset",
        "408 records\n"
        "Phrase • Variants • Literal Meaning\n"
        "Cultural Meaning • Tone • Expression Type\n"
        "Validated / reviewed metadata",
        C["data"]
    )

    # dataset feeds retrieval and meaning grounding
    arrow(6.6, 3.30, 9.20, 5.35)
    arrow(7.45, 2.35, 12.00, 6.15)

    # ========================================================
    # SCORE FORMULA PANEL
    # ========================================================

    panel = FancyBboxPatch(
        (8.25, 0.85), 5.9, 1.65,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#444444",
        facecolor="white"
    )
    ax.add_patch(panel)

    ax.text(
        11.2, 2.15,
        "Meaning-Grounded Ranking",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

    ax.text(
        11.2, 1.55,
        "Final Score = semantic evidence + lexical evidence\n"
        "+ concept/meaning compatibility + linguistic signals\n"
        "− contradiction / mismatch penalties",
        ha="center",
        va="center",
        fontsize=9
    )

    # ========================================================
    # FOOTER
    # ========================================================

    ax.text(
        8.5, 0.35,
        "No Gemini • No Groq • No external LLM dependency • Local retrieval and decision making",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

    save(fig, "01_system_architecture.png")


# ============================================================
# FIGURE 2 — EXPRESSION TYPES
# ============================================================

def expression_type():

    counter = Counter(
        txt(r.get("expression_type")) or "Unspecified"
        for r in dataset
    )

    items = counter.most_common()

    labels = [x[0] for x in items]
    values = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(labels, values)

    ax.set_title(
        "Distribution of Bengali Cultural Expression Types",
        fontweight="bold"
    )
    ax.set_ylabel("Number of records")
    ax.set_xlabel("Expression type")

    ax.tick_params(axis="x", rotation=25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + max(values)*0.015,
            str(value),
            ha="center",
            fontsize=9
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "02_dataset_expression_type.png")


# ============================================================
# FIGURE 3 — FIELD COMPLETENESS
# ============================================================

def field_completeness():

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
    values = []

    for field in fields:
        count = sum(
            has_value(r.get(field))
            for r in dataset
        )

        labels.append(field.replace("_", " ").title())
        values.append(100 * count / len(dataset))

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.barh(labels[::-1], values[::-1])

    ax.set_xlim(0, 105)
    ax.set_xlabel("Completeness (%)")
    ax.set_title(
        "Completeness of Dataset Information Fields",
        fontweight="bold"
    )

    for bar, value in zip(bars, values[::-1]):
        ax.text(
            value + 1,
            bar.get_y() + bar.get_height()/2,
            f"{value:.1f}%",
            va="center",
            fontsize=9
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "03_dataset_field_completeness.png")


# ============================================================
# FIGURE 4 — PHRASE LENGTH
# ============================================================

def phrase_length():

    values = [
        len(txt(r.get("bengali_phrase")).split())
        for r in dataset
        if txt(r.get("bengali_phrase"))
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bins = range(
        min(values),
        max(values) + 2
    )

    ax.hist(values, bins=bins, rwidth=0.82)

    ax.set_title(
        "Distribution of Bengali Expression Length",
        fontweight="bold"
    )
    ax.set_xlabel("Number of whitespace-separated words")
    ax.set_ylabel("Number of records")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "04_phrase_length_distribution.png")


# ============================================================
# FIGURE 5 — TONE
# ============================================================

def tone_distribution():

    counter = Counter()

    for r in dataset:
        tone = (
            txt(r.get("primary_tone"))
            or txt(r.get("intended_emotion_tone"))
            or "Unspecified"
        )
        counter[tone] += 1

    full = dict(counter.most_common())

    with open(
        OUT / "tone_distribution_full.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    top = counter.most_common(12)

    labels = [x[0] for x in top][::-1]
    values = [x[1] for x in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(labels, values)

    ax.set_title(
        "Most Frequent Tone Categories",
        fontweight="bold"
    )
    ax.set_xlabel("Number of records")

    for bar, value in zip(bars, values):
        ax.text(
            value + max(values)*0.015,
            bar.get_y() + bar.get_height()/2,
            str(value),
            va="center",
            fontsize=9
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "05_tone_distribution.png")


# ============================================================
# FIGURE 6A — REVIEW STATUS
# ============================================================

def review_status():

    counter = Counter()

    for r in dataset:
        status = (
            txt(r.get("review_status"))
            or txt(r.get("status"))
            or "unspecified"
        )
        counter[status] += 1

    labels = list(counter.keys())
    values = list(counter.values())

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, values)

    ax.set_title(
        "Dataset Review Status",
        fontweight="bold"
    )
    ax.set_ylabel("Number of records")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + max(values)*0.015,
            str(value),
            ha="center"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "06_dataset_review_status.png")


# ============================================================
# FIGURE 6B — VARIANTS
# ============================================================

def variants():

    counts = [
        len(r.get("variants", []))
        if isinstance(r.get("variants"), list)
        else 0
        for r in dataset
    ]

    counter = Counter(counts)

    labels = [str(x) for x in sorted(counter)]
    values = [counter[int(x)] for x in labels]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, values)

    ax.set_title(
        "Distribution of Variant Entries per Expression",
        fontweight="bold"
    )
    ax.set_xlabel("Number of variants")
    ax.set_ylabel("Number of records")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + max(values)*0.015,
            str(value),
            ha="center",
            fontsize=9
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "07_variant_distribution.png")


# ============================================================
# FIND VALIDATION REPORTS
# ============================================================

def find_validation_records():

    candidates = [
        PROCESSED / "groq_validation_report.json",
        PROCESSED / "audit_report.json",
        PROCESSED / "final_review_log.json",
        PROCESSED / "review_queue.json",
        PROCESSED / "groq_review_queue.json",
    ]

    for path in candidates:

        if not path.exists():
            continue

        try:
            obj = load_json(path)
        except Exception:
            continue

        records = None

        if isinstance(obj, list):
            records = obj

        elif isinstance(obj, dict):

            possible = [
                "results",
                "validation_results",
                "records",
                "report",
                "items",
                "review_queue",
            ]

            for key in possible:
                if isinstance(obj.get(key), list):
                    records = obj[key]
                    break

        if not records:
            continue

        # Detect whether decision fields actually exist
        if any(
            txt(r.get("decision")).upper()
            for r in records
            if isinstance(r, dict)
        ):
            return path, records

    return None, None


# ============================================================
# FIGURE 8 — VALIDATION DECISIONS
# ============================================================

def validation_decisions():

    path, records = find_validation_records()

    if not records:
        print(
            "[SKIP] No validation decision report detected."
        )
        return

    counter = Counter()

    for r in records:
        decision = txt(r.get("decision")).upper()

        if decision:
            counter[decision] += 1

    if not counter:
        print("[SKIP] Validation report has no decisions.")
        return

    preferred = ["KEEP", "FIX", "REVIEW"]

    labels = [
        x for x in preferred
        if x in counter
    ]

    labels += [
        x for x in counter
        if x not in labels
    ]

    values = [counter[x] for x in labels]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, values)

    ax.set_title(
        "Dataset Validation Decisions",
        fontweight="bold"
    )
    ax.set_ylabel("Number of validation records")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + max(values)*0.015,
            str(value),
            ha="center"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "08_validation_decisions.png")

    print(f"Validation source: {path}")


# ============================================================
# EVALUATION
# ============================================================

def load_evaluation():

    if not EVAL.exists():
        print(
            "\n[INFO] Evaluation CSV not found:"
            f"\n{EVAL}"
        )
        print(
            "Performance figures will be generated after "
            "the benchmark is run."
        )
        return None

    with open(EVAL, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(
        f"\nEvaluation results loaded: {len(rows)} queries"
    )

    return rows


# ============================================================
# FIGURE 9 — TOP K
# ============================================================

def topk(rows):

    ans = [
        r for r in rows
        if number(r.get("answerable")) == 1
    ]

    if not ans:
        return

    vals = [
        100 * sum(
            int(number(r.get("top1_correct")))
            for r in ans
        ) / len(ans),

        100 * sum(
            int(number(r.get("top3_hit")))
            for r in ans
        ) / len(ans),

        100 * sum(
            int(number(r.get("top5_hit")))
            for r in ans
        ) / len(ans),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        ["Top-1", "Top-3", "Top-5"],
        vals
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Retrieval success (%)")
    ax.set_title(
        "Retrieval Performance at Different K",
        fontweight="bold"
    )

    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontweight="bold"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "09_topk_performance.png")


# ============================================================
# FIGURE 10 — MAIN METRICS
# ============================================================

def main_metrics(rows):

    ans = [
        r for r in rows
        if number(r.get("answerable")) == 1
    ]

    if not ans:
        return

    metrics = {
        "Top-1 Accuracy":
            sum(
                int(number(r.get("top1_correct")))
                for r in ans
            ) / len(ans),

        "MRR":
            sum(
                number(r.get("reciprocal_rank"))
                for r in ans
            ) / len(ans),

        "Recall@5":
            sum(
                int(number(r.get("top5_hit")))
                for r in ans
            ) / len(ans),
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(
        list(metrics.keys()),
        list(metrics.values())
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(
        "Main Retrieval Performance Metrics",
        fontweight="bold"
    )

    ax.tick_params(axis="x", rotation=15)

    for bar, value in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontweight="bold"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "10_main_metrics.png")


# ============================================================
# FIGURE 11 — LATENCY
# ============================================================

def latency(rows):

    vals = [
        number(r.get("latency_ms"))
        for r in rows
        if number(r.get("latency_ms")) > 0
    ]

    if not vals:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(vals, bins=20)

    ax.axvline(
        statistics.mean(vals),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {statistics.mean(vals):.1f} ms"
    )

    ax.set_title(
        "Query Processing Latency Distribution",
        fontweight="bold"
    )
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Number of queries")
    ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "11_latency_distribution.png")


# ============================================================
# FIGURE 12 — CONFIDENCE
# ============================================================

def confidence(rows):

    vals = [
        number(r.get("confidence"))
        for r in rows
    ]

    if not vals:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        vals,
        bins=20,
        range=(0, 1)
    )

    ax.set_title(
        "Distribution of Final Confidence Scores",
        fontweight="bold"
    )
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Number of queries")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "12_confidence_distribution.png")


# ============================================================
# FIGURE 13 — CATEGORY ACCURACY
# ============================================================

def category_accuracy(rows):

    groups = defaultdict(list)

    for r in rows:

        if number(r.get("answerable")) == 1:

            category = (
                txt(r.get("category"))
                or "Unknown"
            )

            groups[category].append(
                int(number(r.get("top1_correct")))
            )

    if not groups:
        return

    labels = list(groups.keys())

    vals = [
        100 * sum(groups[x]) / len(groups[x])
        for x in labels
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.bar(labels, vals)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(
        "Top-1 Accuracy by Query Category",
        fontweight="bold"
    )

    ax.tick_params(
        axis="x",
        rotation=25
    )

    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            fontsize=8
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "13_category_accuracy.png")


# ============================================================
# FIGURE 14 — ERROR BREAKDOWN
# ============================================================

def error_breakdown(rows):

    ans = [
        r for r in rows
        if number(r.get("answerable")) == 1
    ]

    errors = Counter()

    for r in ans:

        if int(number(r.get("top1_correct"))) == 1:
            continue

        rank = number(r.get("rank_of_gold"))

        if rank and rank <= 5:
            errors["Gold in Top-5\nbut not Top-1"] += 1

        elif txt(r.get("predicted_phrase")):
            errors["Wrong expression"] += 1

        else:
            errors["Incorrect abstention"] += 1

    if not errors:
        return

    labels = list(errors.keys())
    values = list(errors.values())

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, values)

    ax.set_title(
        "Top-1 Error Breakdown",
        fontweight="bold"
    )
    ax.set_ylabel("Number of queries")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + 0.1,
            str(value),
            ha="center"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "14_error_breakdown.png")


# ============================================================
# SUMMARY FILE
# ============================================================

def save_summary():

    summary = {
        "dataset_records": len(dataset),
        "dataset_file": str(DATASET),
        "output_directory": str(OUT),
        "evaluation_available": EVAL.exists(),
    }

    with open(
        OUT / "figure_generation_summary.json",
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
# MAIN
# ============================================================

def main():

    print("\nGenerating publication figures...\n")

    create_architecture()
    expression_type()
    field_completeness()
    phrase_length()
    tone_distribution()
    review_status()
    variants()
    validation_decisions()

    rows = load_evaluation()

    if rows:
        topk(rows)
        main_metrics(rows)
        latency(rows)
        confidence(rows)
        category_accuracy(rows)
        error_breakdown(rows)

    save_summary()

    print("\n" + "=" * 75)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 75)
    print(f"Output folder:\n{OUT}")
    print("\nImportant:")
    print("09-14 are generated ONLY after actual benchmark results exist.")
    print("No fabricated accuracy or performance values are used.")


if __name__ == "__main__":
    main()