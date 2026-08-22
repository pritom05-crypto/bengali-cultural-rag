import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent)
)

import chromadb
from sentence_transformers import SentenceTransformer

from relevance_selector import (
    select_relevant_context
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-base"

TOP_K = 5


# ============================================================
# BENCHMARK QUERIES
# ============================================================

BENCHMARKS = [

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
# NORMALIZATION
# ============================================================

def normalize(text):

    return (
        text
        .strip()
        .replace("/", " ")
        .lower()
    )


# ============================================================
# MATCH
# ============================================================

def is_expected(
    phrase,
    expected
):

    phrase = normalize(
        phrase
    )

    for item in expected:

        if normalize(item) == phrase:

            return True

    return False


# ============================================================
# METRICS
# ============================================================

def recall_at_k(
    ranked,
    expected,
    k
):

    top_k = ranked[:k]

    for item in top_k:

        if is_expected(
            item["bengali_phrase"],
            expected
        ):

            return 1.0

    return 0.0


def reciprocal_rank(
    ranked,
    expected
):

    for index, item in enumerate(
        ranked,
        start=1
    ):

        if is_expected(
            item["bengali_phrase"],
            expected
        ):

            return 1.0 / index

    return 0.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("BENGALI CULTURAL RAG - RERANKER BENCHMARK")
    print("=" * 75)


    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding model loaded."
    )


    # --------------------------------------------------------
    # ChromaDB
    # --------------------------------------------------------

    print(
        "\nConnecting to ChromaDB..."
    )

    client = chromadb.PersistentClient(
        path="data/chroma_db"
    )

    collection = client.get_collection(
        name="bengali_cultural_expressions"
    )

    print(
        f"Collection: {collection.name}"
    )

    print(
        f"Records: {collection.count()}"
    )


    # --------------------------------------------------------
    # Metric storage
    # --------------------------------------------------------

    r1 = []
    r3 = []
    r5 = []
    mrr = []


    # --------------------------------------------------------
    # Run benchmark
    # --------------------------------------------------------

    for index, benchmark in enumerate(
        BENCHMARKS,
        start=1
    ):

        query = benchmark["query"]

        expected = benchmark["expected"]


        # ----------------------------------------------------
        # Encode query
        # ----------------------------------------------------

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True
        )[0].tolist()


        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=TOP_K
        )


        documents = (
            results["documents"][0]
        )

        metadatas = (
            results["metadatas"][0]
        )

        distances = (
            results["distances"][0]
        )


        retrieved = []


        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            similarity = 1 - distance

            retrieved.append({

                "document":
                    document,

                "bengali_phrase":
                    metadata[
                        "bengali_phrase"
                    ],

                "cultural_meaning":
                    metadata[
                        "cultural_meaning"
                    ],

                "intended_emotion_tone":
                    metadata[
                        "intended_emotion_tone"
                    ],

                "similarity":
                    similarity
            })


        # ----------------------------------------------------
        # Rerank
        # ----------------------------------------------------

        scored, selected = (
            select_relevant_context(
                query=query,
                retrieved_items=retrieved,
                max_results=TOP_K,
                min_score=0.0
            )
        )


        # ----------------------------------------------------
        # Metrics use reranked list
        # ----------------------------------------------------

        r1.append(
            recall_at_k(
                scored,
                expected,
                1
            )
        )

        r3.append(
            recall_at_k(
                scored,
                expected,
                3
            )
        )

        r5.append(
            recall_at_k(
                scored,
                expected,
                5
            )
        )

        mrr.append(
            reciprocal_rank(
                scored,
                expected
            )
        )


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print("\n")
        print(
            f"[{index}] QUERY:"
        )

        print(query)

        print(
            "\nExpected:"
        )

        print(
            ", ".join(expected)
        )

        print(
            "\nRERANKED TOP-5:"
        )


        for rank, item in enumerate(
            scored[:5],
            start=1
        ):

            marker = (
                "✓"
                if is_expected(
                    item[
                        "bengali_phrase"
                    ],
                    expected
                )
                else " "
            )

            print(
                f"{rank}. {marker} "
                f"{item['bengali_phrase']:<30} "
                f"score="
                f"{item['relevance_score']:.4f}"
            )


    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    mean_r1 = sum(r1) / len(r1)
    mean_r3 = sum(r3) / len(r3)
    mean_r5 = sum(r5) / len(r5)
    mean_mrr = sum(mrr) / len(mrr)


    print("\n")
    print("=" * 75)
    print("RERANKER RETRIEVAL METRICS")
    print("=" * 75)

    print(
        f"\nRecall@1 : "
        f"{mean_r1:.4f}"
    )

    print(
        f"Recall@3 : "
        f"{mean_r3:.4f}"
    )

    print(
        f"Recall@5 : "
        f"{mean_r5:.4f}"
    )

    print(
        f"MRR      : "
        f"{mean_mrr:.4f}"
    )


    print("\n")
    print("=" * 75)
    print("BENCHMARK COMPLETE")
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()