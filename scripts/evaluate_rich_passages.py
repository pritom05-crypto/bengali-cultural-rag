import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_cultural_dataset.json"
)

MODEL_NAME = "intfloat/multilingual-e5-base"


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


print("=" * 75)
print("BENGALI CULTURAL RAG - RICH PASSAGE BENCHMARK")
print("=" * 75)


with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


print(f"\nDataset records: {len(data)}")


model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)


# ------------------------------------------------------------
# Rich passages
# ------------------------------------------------------------

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
        f"passage: বাংলা সাংস্কৃতিক অভিব্যক্তি: {phrase}. "
        f"অর্থ: {cultural_meaning}. "
        f"ভাব বা প্রয়োগের ধরন: {tone}."
    )

    phrases.append(phrase)
    passages.append(passage)


print("\nEncoding rich passages...")

passage_embeddings = model.encode(
    passages,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=16,
    show_progress_bar=True,
)


print(
    "Embedding matrix:",
    passage_embeddings.shape
)


recall_1 = 0
recall_3 = 0
recall_5 = 0
rr_total = 0.0


print("\n" + "=" * 75)
print("QUERY RESULTS")
print("=" * 75)


for idx, test in enumerate(TEST_QUERIES, start=1):

    query = test["query"]
    expected = set(test["expected"])

    query_embedding = model.encode(
        ["query: " + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

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

    ranks = []

    for rank, i in enumerate(
        ranked_indices,
        start=1
    ):

        if phrases[i] in expected:
            ranks.append(rank)

    if any(
        p in expected
        for p in top5_phrases[:1]
    ):
        recall_1 += 1

    if any(
        p in expected
        for p in top5_phrases[:3]
    ):
        recall_3 += 1

    if any(
        p in expected
        for p in top5_phrases[:5]
    ):
        recall_5 += 1

    if ranks:
        rr_total += 1 / min(ranks)


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


total = len(TEST_QUERIES)


print("\n" + "=" * 75)
print("RICH PASSAGE RETRIEVAL METRICS")
print("=" * 75)

print(
    f"\nRecall@1 : {recall_1 / total:.4f}"
)

print(
    f"Recall@3 : {recall_3 / total:.4f}"
)

print(
    f"Recall@5 : {recall_5 / total:.4f}"
)

print(
    f"MRR      : {rr_total / total:.4f}"
)

print("\n" + "=" * 75)
print("BENCHMARK COMPLETE")
print("=" * 75)