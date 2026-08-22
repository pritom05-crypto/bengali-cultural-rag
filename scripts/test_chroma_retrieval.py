import json
from pathlib import Path

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR = (
    PROJECT_ROOT
    / "data"
    / "chroma_db"
)

MODEL_NAME = "intfloat/multilingual-e5-base"

COLLECTION_NAME = "bengali_cultural_expressions"


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
print("BENGALI CULTURAL RAG - CHROMADB RETRIEVAL TEST")
print("=" * 75)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Model loaded.")


# ============================================================
# CONNECT CHROMADB
# ============================================================

print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=None
)

print(
    "Collection:",
    collection.name
)

print(
    "Record count:",
    collection.count()
)


# ============================================================
# TEST RETRIEVAL
# ============================================================

print("\n" + "=" * 75)
print("RETRIEVAL RESULTS")
print("=" * 75)


recall_1 = 0
recall_3 = 0
recall_5 = 0
rr_total = 0.0


for idx, test in enumerate(
    TEST_QUERIES,
    start=1
):

    query = test["query"]

    expected = set(
        test["expected"]
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # E5 query format
    # --------------------------------------------------------

    query_text = (
        "query: " + query
    )


    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


    # --------------------------------------------------------
    # ChromaDB query
    # --------------------------------------------------------

    result = collection.query(

        query_embeddings=[
            query_embedding.tolist()
        ],

        n_results=5,

        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )


    retrieved_metadata = (
        result["metadatas"][0]
    )

    distances = (
        result["distances"][0]
    )


    retrieved_phrases = [

        item["bengali_phrase"]

        for item in retrieved_metadata
    ]


    # --------------------------------------------------------
    # Find relevant rank
    # --------------------------------------------------------

    relevant_ranks = []

    for rank, phrase in enumerate(
        retrieved_phrases,
        start=1
    ):

        if phrase in expected:

            relevant_ranks.append(
                rank
            )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    if relevant_ranks:

        first_rank = min(
            relevant_ranks
        )

        if first_rank <= 1:
            recall_1 += 1

        if first_rank <= 3:
            recall_3 += 1

        if first_rank <= 5:
            recall_5 += 1

        rr_total += (
            1.0 / first_rank
        )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\n[{idx}] {query}"
    )

    print(
        "Expected:",
        ", ".join(test["expected"])
    )

    print("\nTop-5:")

    for rank, (
        phrase,
        distance
    ) in enumerate(
        zip(
            retrieved_phrases,
            distances
        ),
        start=1
    ):

        marker = (
            "✓"
            if phrase in expected
            else " "
        )

        # Chroma distance for normalized
        # embeddings is approximately
        # cosine distance.

        similarity = (
            1.0 - distance
        )

        print(
            f"{rank}. {marker} "
            f"{phrase:<30} "
            f"similarity={similarity:.4f}"
        )


# ============================================================
# FINAL METRICS
# ============================================================

total = len(TEST_QUERIES)


recall_1_score = (
    recall_1 / total
)

recall_3_score = (
    recall_3 / total
)

recall_5_score = (
    recall_5 / total
)

mrr_score = (
    rr_total / total
)


print("\n" + "=" * 75)
print("CHROMADB RETRIEVAL METRICS")
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
print("CHROMADB RETRIEVAL TEST COMPLETE")
print("=" * 75)