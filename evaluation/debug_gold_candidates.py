import os
import sys
import json
import time
import numpy as np

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SCRIPTS_DIR = os.path.join(
    BASE_DIR,
    "scripts"
)

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


# ============================================================
# IMPORT RAG
# ============================================================

import v6_rag


# ============================================================
# PATHS
# ============================================================

BENCHMARK_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "benchmark_queries.json"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    return (
        str(text)
        .strip()
        .replace("–", "-")
        .replace("—", "-")
        .replace("৷", "")
    )


def get_phrase(record):

    try:
        return v6_rag.get_record_phrase(record)
    except Exception:
        pass

    for key in [
        "phrase",
        "expression",
        "idiom",
        "cultural_expression",
        "title",
        "name"
    ]:

        if isinstance(record, dict):

            value = record.get(key)

            if value:
                return str(value).strip()

    return ""


def get_meaning(record):

    try:
        return v6_rag.get_record_meaning(record)
    except Exception:
        pass

    for key in [
        "meaning",
        "cultural_meaning",
        "definition",
        "description"
    ]:

        if isinstance(record, dict):

            value = record.get(key)

            if value:
                return str(value).strip()

    return ""


def phrase_match(phrase, acceptable):

    phrase_n = normalize_text(
        phrase
    )

    if not phrase_n:
        return False

    if isinstance(
        acceptable,
        str
    ):

        acceptable = [
            x.strip()
            for x in acceptable.split(",")
        ]

    for gold in acceptable:

        gold_n = normalize_text(
            gold
        )

        if not gold_n:
            continue

        if phrase_n == gold_n:
            return True

    return False


# ============================================================
# LOAD BENCHMARK
# ============================================================

print("=" * 78)
print("BENGALI CULTURAL RAG V6.25")
print("GOLD CANDIDATE DIAGNOSTIC")
print("=" * 78)

print("\nLoading benchmark...")

with open(
    BENCHMARK_PATH,
    "r",
    encoding="utf-8"
) as f:

    benchmark = json.load(f)


# Handle different benchmark structures

if isinstance(
    benchmark,
    dict
):

    records = (
        benchmark.get("queries")
        or benchmark.get("benchmark")
        or benchmark.get("records")
        or []
    )

else:

    records = benchmark


verified = [
    x for x in records
    if x.get("verified", True)
]


print(
    f"Benchmark records : {len(records)}"
)

print(
    f"Verified records  : {len(verified)}"
)

print(
    f"Dataset records   : {len(v6_rag.dataset)}"
)


# ============================================================
# DATASET PHRASES
# ============================================================

dataset_phrases = []

for record in v6_rag.dataset:

    dataset_phrases.append(
        get_phrase(record)
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

all_rows = []


for i, item in enumerate(
    verified,
    1
):

    query = (
        item.get("query")
        or item.get("natural_query")
        or item.get("text")
        or ""
    ).strip()


    gold = (
        item.get("gold_phrase")
        or item.get("gold")
        or ""
    ).strip()


    acceptable = item.get(
        "acceptable",
        []
    )


    if isinstance(
        acceptable,
        str
    ):

        acceptable_list = [
            x.strip()
            for x in acceptable.split(",")
            if x.strip()
        ]

    else:

        acceptable_list = [
            str(x).strip()
            for x in acceptable
            if str(x).strip()
        ]


    if gold and gold not in acceptable_list:

        acceptable_list.insert(
            0,
            gold
        )


    print("\n")
    print("-" * 78)
    print(
        f"[{i}/{len(verified)}]"
    )

    print(
        "Query:",
        query
    )

    print(
        "Gold:",
        gold
    )

    print(
        "Acceptable:",
        " | ".join(
            acceptable_list
        )
    )


    # ========================================================
    # FIND GOLD RECORD
    # ========================================================

    gold_indices = []

    for idx, phrase in enumerate(
        dataset_phrases
    ):

        if phrase_match(
            phrase,
            acceptable_list
        ):

            gold_indices.append(
                idx
            )


    if not gold_indices:

        print(
            "\n!!! GOLD NOT FOUND IN DATASET !!!"
        )

        all_rows.append({

            "query": query,
            "gold": gold,
            "gold_found": False

        })

        continue


    print(
        "\nGold dataset rows:",
        gold_indices
    )


    # ========================================================
    # QUERY
    # ========================================================

    try:

        corrected = v6_rag.correct_query(
            query
        )

    except Exception:

        corrected = query


    try:

        concepts = v6_rag.detect_concepts(
            corrected
        )

    except Exception:

        concepts = []


    try:

        expanded = v6_rag.expand_query(
            corrected
        )

        if isinstance(
            expanded,
            dict
        ):

            expanded_queries = (
                expanded.get(
                    "expanded_queries",
                    [corrected]
                )
            )

        else:

            expanded_queries = [
                corrected
            ]

    except Exception:

        expanded_queries = [
            corrected
        ]


    print(
        "\nCorrected:",
        corrected
    )

    print(
        "Concepts:",
        concepts
    )


    # ========================================================
    # SEMANTIC
    # ========================================================

    try:

        qvec = v6_rag.encode_query(
            corrected
        )

        semantic_raw = np.dot(
            v6_rag.embeddings,
            qvec
        )

        semantic = (
            semantic_raw + 1.0
        ) / 2.0

    except Exception as e:

        print(
            "Semantic error:",
            e
        )

        semantic = np.zeros(
            len(v6_rag.dataset)
        )


    # ========================================================
    # BM25
    # ========================================================

    try:

        bm25_query = (
            v6_rag.build_bm25_query(
                corrected,
                concepts
            )
        )

        bm25_raw = v6_rag.bm25.get_scores(
            bm25_query
        )

        bm25 = (
            v6_rag.normalize_scores(
                bm25_raw
            )
        )

    except Exception as e:

        print(
            "BM25 error:",
            e
        )

        bm25 = np.zeros(
            len(v6_rag.dataset)
        )


    # ========================================================
    # GOLD SCORES
    # ========================================================

    gold_scores = []


    for idx in gold_indices:

        record = v6_rag.dataset[idx]


        try:

            meaning_score = float(
                v6_rag.meaning_concept_score(
                    record,
                    concepts
                )
            )

        except Exception:

            meaning_score = 0.0


        try:

            concept_score = float(
                v6_rag.concept_score(
                    record,
                    concepts
                )
            )

        except Exception:

            concept_score = 0.0


        try:

            lexical = float(
                v6_rag.lexical_score(
                    record,
                    corrected
                )
            )

        except Exception:

            lexical = 0.0


        gold_scores.append({

            "idx": idx,

            "phrase":
                get_phrase(record),

            "meaning":
                get_meaning(record),

            "semantic":
                float(
                    semantic[idx]
                ),

            "bm25":
                float(
                    bm25[idx]
                ),

            "concept":
                concept_score,

            "meaning_score":
                meaning_score,

            "lexical":
                lexical

        })


    # ========================================================
    # GLOBAL RANKS
    # ========================================================

    best_gold = max(
        gold_scores,
        key=lambda x:
            (
                x["meaning_score"],
                x["semantic"]
            )
    )


    semantic_order = np.argsort(
        semantic
    )[::-1]

    bm25_order = np.argsort(
        bm25
    )[::-1]


    semantic_rank = (
        np.where(
            semantic_order
            == best_gold["idx"]
        )[0][0] + 1
    )

    bm25_rank = (
        np.where(
            bm25_order
            == best_gold["idx"]
        )[0][0] + 1
    )


    # ========================================================
    # MEANING RANK
    # ========================================================

    meaning_all = np.zeros(
        len(v6_rag.dataset)
    )


    for idx, record in enumerate(
        v6_rag.dataset
    ):

        try:

            meaning_all[idx] = float(
                v6_rag.meaning_concept_score(
                    record,
                    concepts
                )
            )

        except Exception:

            meaning_all[idx] = 0.0


    meaning_order = np.argsort(
        meaning_all
    )[::-1]


    meaning_rank = (
        np.where(
            meaning_order
            == best_gold["idx"]
        )[0][0] + 1
    )


    # ========================================================
    # RETRIEVE RESULT
    # ========================================================

    try:

        retrieved = v6_rag.retrieve(
            query,
            corrected,
            concepts,
            expanded_queries
        )

    except Exception as e:

        print(
            "\nRetrieve error:",
            e
        )

        retrieved = []


    retrieved_phrases = []

    for r in retrieved:

        if isinstance(
            r,
            dict
        ):

            phrase = (
                r.get("phrase")
                or get_phrase(
                    r.get(
                        "record",
                        {}
                    )
                )
            )

        else:

            phrase = str(r)

        retrieved_phrases.append(
            phrase
        )


    gold_in_retrieval = any(
        phrase_match(
            phrase,
            acceptable_list
        )
        for phrase
        in retrieved_phrases
    )


    # ========================================================
    # PRINT
    # ========================================================

    print("\nBEST GOLD RECORD")

    print(
        "Phrase:",
        best_gold["phrase"]
    )

    print(
        "Meaning:",
        best_gold["meaning"]
    )

    print(
        f"Semantic score : "
        f"{best_gold['semantic']:.4f}"
    )

    print(
        f"Semantic rank  : "
        f"{semantic_rank}"
    )

    print(
        f"BM25 score     : "
        f"{best_gold['bm25']:.4f}"
    )

    print(
        f"BM25 rank      : "
        f"{bm25_rank}"
    )

    print(
        f"Concept score  : "
        f"{best_gold['concept']:.4f}"
    )

    print(
        f"Meaning score  : "
        f"{best_gold['meaning_score']:.4f}"
    )

    print(
        f"Meaning rank   : "
        f"{meaning_rank}"
    )

    print(
        f"Lexical score  : "
        f"{best_gold['lexical']:.4f}"
    )

    print(
        "\nRetrieved:",
        " | ".join(
            retrieved_phrases[:5]
        )
    )

    print(
        "\nGold in retrieve():",
        gold_in_retrieval
    )


    all_rows.append({

        "query":
            query,

        "gold":
            gold,

        "gold_found":
            True,

        "semantic_rank":
            semantic_rank,

        "bm25_rank":
            bm25_rank,

        "meaning_rank":
            meaning_rank,

        "semantic_score":
            best_gold["semantic"],

        "bm25_score":
            best_gold["bm25"],

        "concept_score":
            best_gold["concept"],

        "meaning_score":
            best_gold["meaning_score"],

        "lexical_score":
            best_gold["lexical"],

        "gold_in_retrieve":
            gold_in_retrieval

    })


# ============================================================
# SAVE
# ============================================================

output_path = os.path.join(
    BASE_DIR,
    "evaluation",
    "gold_candidate_diagnostic.json"
)


with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_rows,
        f,
        ensure_ascii=False,
        indent=2
        default=lambda x: x.item() if hasattr(x, "item") else str(x)
    )


# ============================================================
# SUMMARY
# ============================================================

found = [
    x for x in all_rows
    if x.get("gold_found")
]

in_retrieve = [
    x for x in found
    if x.get("gold_in_retrieve")
]


print("\n")
print("=" * 78)
print("DIAGNOSTIC SUMMARY")
print("=" * 78)

print(
    "Benchmark queries:",
    len(all_rows)
)

print(
    "Gold found in dataset:",
    len(found)
)

print(
    "Gold found in retrieve():",
    len(in_retrieve)
)


if found:

    print(
        "\nAverage semantic rank:",
        round(
            np.mean([
                x["semantic_rank"]
                for x in found
            ]),
            2
        )
    )

    print(
        "Average BM25 rank:",
        round(
            np.mean([
                x["bm25_rank"]
                for x in found
            ]),
            2
        )
    )

    print(
        "Average meaning rank:",
        round(
            np.mean([
                x["meaning_rank"]
                for x in found
            ]),
            2
        )
    )


print(
    "\nSaved:",
    output_path
)

print("=" * 78)