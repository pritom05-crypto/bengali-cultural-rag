import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_cultural_dataset.json"
)

MODEL_NAME = "nahidstaq/bge-m3-bangla"


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [

    {
        "query": "আমি খুব বিপদে পড়েছি",
        "expected": [
            "অথৈ জলে পড়া",
            "অকূল পাথার",
            "আতান্তরে পড়া",
            "অথৈ জল",
        ],
    },

    {
        "query": "সে খুব রেগে গেছে",
        "expected": [
            "অগ্নিশর্মা",
        ],
    },

    {
        "query": "সে খুব অলস",
        "expected": [
            "অকর্মার ধাড়ি",
            "অজগর বৃত্তি",
            "ইতুনিদকুঁড়ে",
        ],
    },

    {
        "query": "হঠাৎ করে অনেক ধনী হয়ে গেল",
        "expected": [
            "আঙুল ফুলে কলাগাছ",
        ],
    },

    {
        "query": "এক কাজ করে দুইটা উদ্দেশ্য পূরণ করা",
        "expected": [
            "এক ঢিলে দু’পাখি",
        ],
    },

    {
        "query": "নিজের কাজের কারণে নিজেরই ক্ষতি করা",
        "expected": [
            "আপন পায়ে কুড়াল মারা",
        ],
    },

    {
        "query": "সে সিদ্ধান্ত নিতে পারছে না",
        "expected": [
            "ইতস্তত করা",
            "অস্থির পঞ্চক",
        ],
    },

    {
        "query": "হালকা গুঁড়ি গুঁড়ি বৃষ্টি হচ্ছে",
        "expected": [
            "ইলশে গুঁড়ি",
        ],
    },

    {
        "query": "অনেক বন্ধু একসাথে",
        "expected": [
            "ইয়ারবকসি",
        ],
    },

    {
        "query": "সে খুব দ্বিধায় আছে",
        "expected": [
            "ইতস্তত করা",
            "আমতা আমতা করা",
        ],
    },

    {
        "query": "অবাস্তব বা অসম্ভব কিছু কল্পনা করা",
        "expected": [
            "আকাশ কুসুম",
            "কপোল-কল্পনা",
        ],
    },

    {
        "query": "অতিরিক্ত গর্বের কারণে পতন হওয়া",
        "expected": [
            "অতি দর্পে হত লঙ্কা",
        ],
    },
]


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("BENGALI CULTURAL RAG - BGE-M3 BANGLA BENCHMARK")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\nDataset records: {len(data)}")


# ============================================================
# LOAD MODEL
# ============================================================

print(f"\nLoading model: {MODEL_NAME}")
print("Device: CPU")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Model loaded successfully.")


# ============================================================
# BUILD RICH PASSAGES
# ============================================================

phrases = []
passages = []

for item in data:

    phrase = item["bengali_phrase"].strip()

    cultural_meaning = (
        item.get("cultural_meaning") or ""
    ).strip()

    tone = (
        item.get("intended_emotion_tone") or ""
    ).strip()

    passage = (
        f"বাংলা সাংস্কৃতিক অভিব্যক্তি: {phrase}. "
        f"অর্থ: {cultural_meaning}. "
        f"ভাব বা প্রয়োগের ধরন: {tone}."
    )

    phrases.append(phrase)
    passages.append(passage)


# ============================================================
# EMBED PASSAGES
# ============================================================

print("\nEncoding rich passages...")

passage_embeddings = model.encode(
    passages,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=16,
    show_progress_bar=True,
)

print(
    "\nEmbedding matrix:",
    passage_embeddings.shape
)


# ============================================================
# METRICS
# ============================================================

recall_1 = 0
recall_3 = 0
recall_5 = 0
rr_total = 0.0


# ============================================================
# QUERY EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("QUERY RESULTS")
print("=" * 75)


for idx, test in enumerate(TEST_QUERIES, start=1):

    query = test["query"]
    expected = set(test["expected"])

    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    scores = np.dot(
        passage_embeddings,
        query_embedding
    )

    ranked_indices = np.argsort(scores)[::-1]

    top5_indices = ranked_indices[:5]

    top5_phrases = [
        phrases[i]
        for i in top5_indices
    ]

    # --------------------------------------------------------
    # Find first relevant rank
    # --------------------------------------------------------

    relevant_ranks = []

    for rank, i in enumerate(
        ranked_indices,
        start=1
    ):
        if phrases[i] in expected:
            relevant_ranks.append(rank)

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if any(
        phrases[i] in expected
        for i in ranked_indices[:1]
    ):
        recall_1 += 1

    if any(
        phrases[i] in expected
        for i in ranked_indices[:3]
    ):
        recall_3 += 1

    if any(
        phrases[i] in expected
        for i in ranked_indices[:5]
    ):
        recall_5 += 1

    # --------------------------------------------------------
    # Reciprocal Rank
    # --------------------------------------------------------

    if relevant_ranks:
        rr_total += 1 / min(relevant_ranks)

    # --------------------------------------------------------
    # Print query
    # --------------------------------------------------------

    print(f"\n[{idx}] QUERY:")
    print(query)

    print("\nExpected:")
    print(", ".join(test["expected"]))

    print("\nTop-5:")

    for rank, i in enumerate(
        top5_indices,
        start=1
    ):

        marker = (
            "✓"
            if phrases[i] in expected
            else " "
        )

        print(
            f"{rank}. {marker} "
            f"{phrases[i]:<30} "
            f"{scores[i]:.4f}"
        )


# ============================================================
# FINAL METRICS
# ============================================================

total = len(TEST_QUERIES)

recall_1_score = recall_1 / total
recall_3_score = recall_3 / total
recall_5_score = recall_5 / total
mrr_score = rr_total / total


print("\n" + "=" * 75)
print("BGE-M3 BANGLA RETRIEVAL METRICS")
print("=" * 75)

print(
    f"\nRecall@1 : {recall_1_score:.4f}"
)

print(
    f"Recall@3 : {recall_3_score:.4f}"
)

print(
    f"Recall@5 : {recall_5_score:.4f}"
)

print(
    f"MRR      : {mrr_score:.4f}"
)

print("\n" + "=" * 75)
print("BENCHMARK COMPLETE")
print("=" * 75)