# scripts/expanded_retrieval.py

import os
import json
import numpy as np

from sentence_transformers import SentenceTransformer

from query_expansion import expand_query


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_cultural_dataset.json"
)

EMBEDDING_PATH = os.path.join(
    BASE_DIR,
    "data",
    "embeddings",
    "e5_rich_embeddings.npy"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-base"

TOP_K_PER_QUERY = 10
FINAL_TOP_K = 10


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("BENGALI CULTURAL RAG - QUERY EXPANSION RETRIEVAL")
print("=" * 70)

print("\nLoading final dataset...")

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:
    dataset = json.load(f)

print(f"Dataset records: {len(dataset)}")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading E5 embeddings...")

embeddings = np.load(EMBEDDING_PATH)

print(f"Embedding matrix: {embeddings.shape}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

def normalize_matrix(matrix):

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    return matrix / norms


document_embeddings = normalize_matrix(embeddings)


# ============================================================
# CREATE RICH PASSAGE
# ============================================================

def build_passage(record):

    phrase = record.get(
        "bengali_phrase",
        ""
    )

    meaning = record.get(
        "cultural_meaning",
        ""
    )

    tone = record.get(
        "intended_emotion_tone",
        ""
    )

    return (
        f"বাংলা সাংস্কৃতিক অভিব্যক্তি: {phrase}. "
        f"অর্থ: {meaning}. "
        f"ভাব বা প্রয়োগের ধরন: {tone}."
    )


# ============================================================
# QUERY EMBEDDING
# ============================================================

def encode_query(query):

    # E5 query prefix
    text = "query: " + query

    vector = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    return vector


# ============================================================
# SINGLE QUERY RETRIEVAL
# ============================================================

def retrieve_single_query(query, top_k=TOP_K_PER_QUERY):

    vector = encode_query(query)

    scores = np.dot(
        document_embeddings,
        vector
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, 1):

        record = dataset[int(idx)]

        results.append({
            "index": int(idx),
            "rank": rank,
            "score": float(scores[idx]),
            "record": record
        })

    return results


# ============================================================
# EXPANDED RETRIEVAL
# ============================================================

def expanded_retrieve(query):

    expansion = expand_query(query)

    expanded_queries = expansion[
        "expanded_queries"
    ]

    print("\n" + "=" * 70)
    print("QUERY EXPANSION")
    print("=" * 70)

    for i, q in enumerate(
        expanded_queries,
        1
    ):
        print(f"{i}. {q}")

    # --------------------------------------------------------
    # Retrieve from every expanded query
    # --------------------------------------------------------

    candidate_scores = {}

    candidate_evidence = {}

    for query_number, expanded_query in enumerate(
        expanded_queries
    ):

        results = retrieve_single_query(
            expanded_query,
            TOP_K_PER_QUERY
        )

        for result in results:

            idx = result["index"]
            score = result["score"]

            if idx not in candidate_scores:
                candidate_scores[idx] = []

            candidate_scores[idx].append(score)

            if idx not in candidate_evidence:
                candidate_evidence[idx] = []

            candidate_evidence[idx].append({
                "query": expanded_query,
                "score": score,
                "rank": result["rank"]
            })


    # --------------------------------------------------------
    # Aggregate scores
    # --------------------------------------------------------

    final_candidates = []

    for idx, scores in candidate_scores.items():

        scores_array = np.array(
            scores,
            dtype=np.float32
        )

        # Best evidence gets highest importance
        max_score = float(
            np.max(scores_array)
        )

        # Mean evidence
        mean_score = float(
            np.mean(scores_array)
        )

        # Number of expanded queries retrieving this record
        coverage = len(scores)

        # Coverage bonus
        coverage_bonus = min(
            coverage / len(expanded_queries),
            1.0
        )

        # Final expansion score
        final_score = (
            0.70 * max_score
            + 0.20 * mean_score
            + 0.10 * coverage_bonus
        )

        final_candidates.append({
            "index": idx,
            "score": final_score,
            "max_score": max_score,
            "mean_score": mean_score,
            "coverage": coverage,
            "evidence": candidate_evidence[idx],
            "record": dataset[idx]
        })


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    final_candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return expansion, final_candidates[:FINAL_TOP_K]


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(
    expansion,
    results
):

    print("\n" + "=" * 70)
    print("EXPANDED RETRIEVAL RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        1
    ):

        record = result["record"]

        print(
            f"\n[{rank}] "
            f"{record.get('bengali_phrase', '')}"
        )

        print(
            f"Expansion Score : "
            f"{result['score']:.4f}"
        )

        print(
            f"Best Similarity  : "
            f"{result['max_score']:.4f}"
        )

        print(
            f"Coverage         : "
            f"{result['coverage']}"
        )

        print(
            f"Meaning          : "
            f"{record.get('cultural_meaning', '')}"
        )

        print(
            f"Tone             : "
            f"{record.get('intended_emotion_tone', '')}"
        )


# ============================================================
# INTERACTIVE LOOP
# ============================================================

if __name__ == "__main__":

    while True:

        print("\nআপনার প্রশ্ন লিখুন:")

        try:
            query = input("> ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break

        if not query:
            continue

        if query.lower() in [
            "exit",
            "quit",
            "q"
        ]:
            print("Exiting...")
            break

        expansion, results = expanded_retrieve(
            query
        )

        print_results(
            expansion,
            results
        )