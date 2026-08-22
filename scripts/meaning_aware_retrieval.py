import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_cultural_dataset.json"
)

EMBEDDING_PATH = (
    BASE_DIR
    / "data"
    / "embeddings"
    / "e5_rich_embeddings.npy"
)

MODEL_NAME = "intfloat/multilingual-e5-base"

TOP_K_SEMANTIC = 50
TOP_K_BM25 = 30
TOP_K_FINAL = 10


# ============================================================
# CONCEPT KEYWORDS
# ============================================================

CONCEPTS = {

    "অলস": [
        "অলস",
        "কুঁড়ে",
        "কুঁড়ে",
        "কর্মবিমুখ",
        "নিষ্ক্রিয়",
        "নিষ্ক্রিয়",
        "কাজে অনীহা",
        "কাজ করতে চায় না",
        "কাজ করতে চায় না",
        "পরিশ্রম বিমুখ",
        "ঢিলেমি",
        "উদ্যমহীন",
        "কাজকর্মে অনীহা",
    ],

    "বিপদ": [
        "বিপদ",
        "বিপদে",
        "বিপাকে",
        "সংকট",
        "সমস্যা",
        "আতান্তর",
        "ঝুঁকি",
        "দুর্দশা",
    ],

    "রাগ": [
        "রাগ",
        "রেগে",
        "ক্রোধ",
        "ক্ষোভ",
        "উত্তেজিত",
        "অগ্নিশর্মা",
    ],

    "দ্বিধা": [
        "দ্বিধা",
        "দ্বিধায়",
        "দ্বিধায়",
        "সিদ্ধান্ত নিতে পারছে না",
        "ইতস্তত",
        "আমতা আমতা",
        "অনিশ্চিত",
        "দোটানা",
        "বিভ্রান্ত",
    ],

    "বন্ধুত্ব": [
        "বন্ধু",
        "বন্ধুত্ব",
        "সঙ্গী",
        "সাথী",
        "বন্ধুবান্ধব",
        "ঘনিষ্ঠ",
    ],

    "বৃষ্টি": [
        "বৃষ্টি",
        "গুঁড়ি গুঁড়ি",
        "গুঁড়ি গুঁড়ি",
        "ঝিরঝির",
        "বর্ষণ",
        "ইলশে গুঁড়ি",
        "ইলশে গুঁড়ি",
    ],

    "ধনী": [
        "ধনী",
        "ধনবান",
        "টাকা",
        "সম্পদ",
        "অনেক টাকা",
        "হঠাৎ ধনী",
    ],

    "অসম্ভব": [
        "অসম্ভব",
        "অবাস্তব",
        "কল্পনা",
        "অসম্ভব কিছু",
        "অবাস্তব কিছু",
        "আকাশ কুসুম",
        "কপোল-কল্পনা",
    ],

    "ক্ষতি": [
        "ক্ষতি",
        "নিজের ক্ষতি",
        "নিজেরই ক্ষতি",
        "নিজের কাজের কারণে ক্ষতি",
        "নিজের পায়ে কুড়াল",
        "নিজের পায়ে কুড়াল",
    ],

    "দুই উদ্দেশ্য": [
        "দুইটা উদ্দেশ্য",
        "দুই উদ্দেশ্য",
        "এক কাজ",
        "এক ঢিলে",
        "দুই কাজ",
    ],

    "গর্ব": [
        "গর্ব",
        "অহংকার",
        "দর্প",
        "অতিরিক্ত গর্ব",
        "পতন",
    ],
}


# ============================================================
# MEANING ALIASES
# ============================================================

MEANING_ALIASES = {

    "অলস": [
        "lazy",
        "laziness",
        "idle",
        "inactive",
        "very lazy",
        "terribly lazy",
        "extremely lazy",
        "idle person",
        "wasting time",
        "without work",

        "অলস",
        "অলসতা",
        "কুঁড়ে",
        "কুঁড়ে",
        "কর্মবিমুখ",
        "নিষ্ক্রিয়",
        "নিষ্ক্রিয়",
        "উদ্যমহীন",
        "পরিশ্রম বিমুখ",
        "কাজে অনীহা",
        "কাজ করতে চায় না",
        "কাজ করতে চায় না",
        "ঢিলেমি",
        "কাজকর্মে অনীহা",
    ],

    "বিপদ": [
        "danger",
        "trouble",
        "crisis",
        "distress",
        "dangerous situation",
        "great danger",
        "terrible danger",
        "falling into sudden danger or trouble",
        "facing a sudden danger or disaster",

        "বিপদ",
        "বিপাকে",
        "সংকট",
        "সমস্যা",
        "আতান্তর",
        "ঝুঁকি",
        "দুর্দশা",
    ],

    "রাগ": [
        "angry",
        "anger",
        "furious",
        "fury",
        "rage",
        "extremely angry",

        "রাগ",
        "রেগে",
        "ক্রোধ",
        "ক্ষোভ",
        "অগ্নিশর্মা",
    ],

    "দ্বিধা": [
        "hesitate",
        "hesitation",
        "uncertain",
        "confused",
        "indecisive",

        "দ্বিধা",
        "দ্বিধায়",
        "দ্বিধায়",
        "ইতস্তত",
        "আমতা আমতা",
        "অনিশ্চিত",
        "দোটানা",
        "বিভ্রান্ত",
    ],

    "বন্ধুত্ব": [
        "friend",
        "friends",
        "friendship",
        "close friendship",
        "close intimacy",
        "many friends",

        "বন্ধু",
        "বন্ধুত্ব",
        "সঙ্গী",
        "সাথী",
        "বন্ধুবান্ধব",
        "ঘনিষ্ঠ",
    ],

    "বৃষ্টি": [
        "rain",
        "drizzle",
        "light rain",
        "small rain",

        "বৃষ্টি",
        "গুঁড়ি গুঁড়ি",
        "গুঁড়ি গুঁড়ি",
        "ঝিরঝির",
        "বর্ষণ",
        "ইলশে গুঁড়ি",
        "ইলশে গুঁড়ি",
    ],

    "ধনী": [
        "rich",
        "wealthy",
        "wealth",
        "suddenly rich",
        "many money",

        "ধনী",
        "ধনবান",
        "টাকা",
        "সম্পদ",
        "অনেক টাকা",
        "হঠাৎ ধনী",
    ],

    "অসম্ভব": [
        "impossible",
        "impossible thing",
        "unrealistic",
        "fiction",
        "illusionary",
        "impossible object",
        "forcing an impossible thing",

        "অবাস্তব",
        "অসম্ভব",
        "কল্পনা",
        "আকাশ কুসুম",
        "কপোল-কল্পনা",
    ],

    "ক্ষতি": [
        "damage",
        "harm",
        "self harm",
        "own damage",
        "causing one's own loss",

        "ক্ষতি",
        "নিজের ক্ষতি",
        "নিজেরই ক্ষতি",
    ],

    "দুই উদ্দেশ্য": [
        "two purposes",
        "two objectives",
        "two goals",
        "one action two purposes",

        "দুইটা উদ্দেশ্য",
        "দুই উদ্দেশ্য",
        "এক কাজ",
        "এক ঢিলে",
        "দুই কাজ",
    ],

    "গর্ব": [
        "pride",
        "arrogance",
        "boast",
        "excessive pride",
        "fall of pride",

        "গর্ব",
        "অহংকার",
        "দর্প",
        "অতিরিক্ত গর্ব",
        "পতন",
    ],
}


# ============================================================
# CULTURAL PRIORITY
# ============================================================

PRIORITY_GROUPS = {

    "অলস": [
        "অকর্মার ধাড়ি",
        "অজগর বৃত্তি",
        "ইতুনিদকুঁড়ে",
        "উদোগেঁড়ে",
        "কুড়ের বাদশা",
        "কুমড়ো কাটা বটঠাকুর",
    ],

    "বিপদ": [
        "আতান্তরে পড়া",
        "অথৈ জল",
        "অকূল পাথার",
        "অথৈ জলে পড়া",
    ],

    "রাগ": [
        "অগ্নিশর্মা",
    ],

    "দ্বিধা": [
        "ইতস্তত করা",
        "আমতা আমতা করা",
        "অস্থির পঞ্চক",
    ],

    "বন্ধুত্ব": [
        "ইয়ারবকসি",
        "গলায় গলায় ভাব",
    ],

    "বৃষ্টি": [
        "ইলশে গুঁড়ি",
    ],

    "ধনী": [
        "আঙুল ফুলে কলাগাছ",
    ],

    "অসম্ভব": [
        "আকাশ কুসুম",
        "কপোল-কল্পনা",
    ],

    "ক্ষতি": [
        "আপন পায়ে কুড়াল মারা",
    ],

    "দুই উদ্দেশ্য": [
        "এক ঢিলে দু’পাখি",
        "এক ঢিলে দু-পাখি",
        "এক ঢিলে দু'পাখি",
    ],

    "গর্ব": [
        "অতি দর্পে হত লঙ্কা",
    ],
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(text).lower().strip()

    replacements = {
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):

    return re.findall(
        r"[\u0980-\u09FF]+|[a-zA-Z]+",
        normalize_text(text)
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("BENGALI CULTURAL RAG - MEANING AWARE RETRIEVAL V5.2")
print("=" * 70)

print("\nLoading final dataset...")

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)

print(
    f"Dataset records: {len(dataset)}"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading E5 embeddings...")

embeddings = np.load(
    EMBEDDING_PATH
)

print(
    f"Embedding matrix: {embeddings.shape}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print(
    "Embedding model loaded."
)


# ============================================================
# DATASET TEXT
# ============================================================

dataset_texts = []

for item in dataset:

    text = " ".join([
        str(item.get("bengali_phrase", "")),
        str(item.get("literal_meaning", "")),
        str(item.get("cultural_meaning", "")),
        str(item.get("intended_emotion_tone", "")),
    ])

    dataset_texts.append(
        normalize_text(text)
    )


# ============================================================
# BM25
# ============================================================

print("\nBuilding BM25 index...")

tokenized_documents = [
    tokenize(text)
    for text in dataset_texts
]

bm25 = BM25Okapi(
    tokenized_documents
)

print(
    "BM25 index ready."
)


# ============================================================
# CONCEPT DETECTION
# ============================================================

def detect_concepts(query):

    query_normalized = normalize_text(
        query
    )

    detected = []

    for concept, keywords in CONCEPTS.items():

        for keyword in keywords:

            if normalize_text(keyword) in query_normalized:

                detected.append(
                    concept
                )

                break

    return list(
        dict.fromkeys(detected)
    )


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query, concepts):

    expanded = [query]

    for concept in concepts:

        for keyword in CONCEPTS.get(
            concept,
            []
        ):

            expanded.append(
                keyword
            )

    result = []

    for q in expanded:

        q = str(q).strip()

        if q and q not in result:

            result.append(q)

    return result[:15]


# ============================================================
# MEANING MATCH
# ============================================================

def meaning_matches_concept(
    cultural_meaning,
    concept
):

    if not cultural_meaning:
        return False

    meaning = normalize_text(
        cultural_meaning
    )

    aliases = MEANING_ALIASES.get(
        concept,
        []
    )

    for alias in aliases:

        if normalize_text(alias) in meaning:

            return True

    return False


# ============================================================
# MEANING SCORE
# ============================================================

def calculate_meaning_score(
    cultural_meaning,
    concepts
):

    if not cultural_meaning or not concepts:

        return 0.0

    matched = 0

    for concept in concepts:

        if meaning_matches_concept(
            cultural_meaning,
            concept
        ):

            matched += 1

    return (
        matched / len(concepts)
    )


# ============================================================
# PRIORITY
# ============================================================

def cultural_priority(
    item,
    concepts
):

    phrase = normalize_text(
        item.get(
            "bengali_phrase",
            ""
        )
    )

    for concept in concepts:

        candidates = PRIORITY_GROUPS.get(
            concept,
            []
        )

        for rank, candidate in enumerate(
            candidates
        ):

            if phrase == normalize_text(
                candidate
            ):

                priority_values = [
                    1.00,
                    0.98,
                    0.96,
                    0.94,
                    0.94,
                    0.92,
                ]

                return priority_values[
                    min(
                        rank,
                        len(priority_values) - 1
                    )
                ]

    return 0.0


# ============================================================
# BM25
# ============================================================

def get_bm25_scores(queries):

    scores = np.zeros(
        len(dataset),
        dtype=float
    )

    for query in queries:

        tokens = tokenize(
            query
        )

        if not tokens:
            continue

        raw = bm25.get_scores(
            tokens
        )

        max_value = np.max(raw)

        if max_value > 0:

            normalized = (
                raw / max_value
            )

            scores = np.maximum(
                scores,
                normalized
            )

    return scores


# ============================================================
# PRECISION FILTER
# ============================================================

def precision_filter(
    result,
    concepts
):

    # No concept detected:
    # don't aggressively filter.
    if not concepts:

        return True

    meaning = result["meaning"]
    priority = result["priority"]
    semantic = result["semantic"]
    coverage = result["coverage"]
    bm25 = result["bm25"]

    # --------------------------------------------------------
    # RULE 1
    # Strong meaning match = ALWAYS KEEP
    # --------------------------------------------------------

    if meaning >= 0.75:

        return True


    # --------------------------------------------------------
    # RULE 2
    # Cultural priority = KEEP
    # --------------------------------------------------------

    if priority >= 0.90:

        return True


    # --------------------------------------------------------
    # RULE 3
    # Good meaning + reasonable semantic
    # --------------------------------------------------------

    if (
        meaning >= 0.40
        and semantic >= 0.75
    ):

        return True


    # --------------------------------------------------------
    # RULE 4
    # Strong semantic + BM25
    # --------------------------------------------------------

    if (
        semantic >= 0.84
        and bm25 >= 0.20
    ):

        return True


    # --------------------------------------------------------
    # RULE 5
    # Strong semantic + high coverage
    # --------------------------------------------------------

    if (
        semantic >= 0.86
        and coverage >= 3
    ):

        return True


    # --------------------------------------------------------
    # RULE 6
    # Everything else = REMOVE
    # --------------------------------------------------------

    return False


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(query):

    concepts = detect_concepts(
        query
    )

    expanded_queries = expand_query(
        query,
        concepts
    )


    # --------------------------------------------------------
    # Print concepts
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DETECTED CONCEPTS"
    )

    print(
        "=" * 70
    )

    if concepts:

        for concept in concepts:

            print(
                f"- {concept}"
            )

    else:

        print(
            "- None"
        )


    # --------------------------------------------------------
    # Expanded queries
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPANDED QUERIES"
    )

    print(
        "=" * 70
    )

    for i, q in enumerate(
        expanded_queries,
        1
    ):

        print(
            f"{i}. {q}"
        )


    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    query_embeddings = model.encode(
        expanded_queries,
        normalize_embeddings=True,
        show_progress_bar=False
    )


    similarity_matrix = np.matmul(
        embeddings,
        query_embeddings.T
    )


    best_similarity = np.max(
        similarity_matrix,
        axis=1
    )


    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    coverage_counts = np.sum(
        similarity_matrix >= 0.72,
        axis=1
    )

    coverage_scores = (
        coverage_counts
        / len(expanded_queries)
    )


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_scores = get_bm25_scores(
        expanded_queries
    )


    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidate_indices = set()

    semantic_candidates = np.argsort(
        best_similarity
    )[::-1][:TOP_K_SEMANTIC]

    candidate_indices.update(
        semantic_candidates.tolist()
    )

    bm25_candidates = np.argsort(
        bm25_scores
    )[::-1][:TOP_K_BM25]

    candidate_indices.update(
        bm25_candidates.tolist()
    )


    # --------------------------------------------------------
    # Inject priority expressions
    # --------------------------------------------------------

    for concept in concepts:

        priority_phrases = PRIORITY_GROUPS.get(
            concept,
            []
        )

        normalized_priority = {
            normalize_text(x)
            for x in priority_phrases
        }

        for idx, item in enumerate(
            dataset
        ):

            phrase = normalize_text(
                item.get(
                    "bengali_phrase",
                    ""
                )
            )

            if phrase in normalized_priority:

                candidate_indices.add(
                    idx
                )


    # --------------------------------------------------------
    # Calculate scores
    # --------------------------------------------------------

    results = []

    for idx in candidate_indices:

        item = dataset[idx]

        semantic = float(
            best_similarity[idx]
        )

        coverage = int(
            coverage_counts[idx]
        )

        meaning = calculate_meaning_score(
            item.get(
                "cultural_meaning",
                ""
            ),
            concepts
        )

        priority = cultural_priority(
            item,
            concepts
        )

        bm25 = float(
            bm25_scores[idx]
        )


        # ----------------------------------------------------
        # Penalty
        # ----------------------------------------------------

        penalty = 0.0

        if (
            concepts
            and meaning == 0.0
            and priority == 0.0
            and semantic < 0.84
        ):

            penalty = 0.15


        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = (

            0.25 * semantic

            +

            0.30 * meaning

            +

            0.10 * coverage_scores[idx]

            +

            0.05 * bm25

            +

            0.30 * priority

            -

            penalty
        )


        final_score = max(
            0.0,
            min(
                1.0,
                final_score
            )
        )


        results.append({

            "index": int(idx),

            "phrase":
                item.get(
                    "bengali_phrase",
                    ""
                ),

            "semantic":
                semantic,

            "coverage":
                coverage,

            "meaning":
                meaning,

            "bm25":
                bm25,

            "priority":
                priority,

            "penalty":
                penalty,

            "final":
                final_score,

            "cultural_meaning":
                item.get(
                    "cultural_meaning",
                    ""
                ),

            "tone":
                item.get(
                    "intended_emotion_tone",
                    ""
                ),
        })


    # --------------------------------------------------------
    # Sort before filtering
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["final"],
        reverse=True
    )


    # --------------------------------------------------------
    # Precision filtering
    # --------------------------------------------------------

    filtered_results = []

    for result in results:

        if precision_filter(
            result,
            concepts
        ):

            filtered_results.append(
                result
            )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    # If filtering becomes too aggressive,
    # keep the strongest semantic candidate.

    if not filtered_results and results:

        filtered_results = [
            results[0]
        ]


    return (
        concepts,
        expanded_queries,
        filtered_results[:TOP_K_FINAL]
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    while True:

        try:

            query = input(
                "\nআপনার প্রশ্ন লিখুন:\n> "
            ).strip()


            if not query:

                continue


            if query.lower() in [
                "exit",
                "quit",
                "q"
            ]:

                print(
                    "\nExiting..."
                )

                break


            (
                concepts,
                expanded_queries,
                results
            ) = retrieve(
                query
            )


            print(
                "\n"
                + "=" * 70
            )

            print(
                "PRECISION FILTERING"
            )

            print(
                "=" * 70
            )

            print(
                f"Candidates after filtering: "
                f"{len(results)}"
            )


            print(
                "\n"
                + "=" * 70
            )

            print(
                "MEANING-AWARE RETRIEVAL V5.2 RESULTS"
            )

            print(
                "=" * 70
            )


            for i, result in enumerate(
                results,
                1
            ):

                print(
                    f"\n[{i}] "
                    f"{result['phrase']}"
                )

                print(
                    f"Semantic : "
                    f"{result['semantic']:.4f}"
                )

                print(
                    f"Coverage : "
                    f"{result['coverage']}"
                )

                print(
                    f"Meaning  : "
                    f"{result['meaning']:.4f}"
                )

                print(
                    f"BM25     : "
                    f"{result['bm25']:.4f}"
                )

                print(
                    f"Priority : "
                    f"{result['priority']:.4f}"
                )

                print(
                    f"Penalty  : "
                    f"{result['penalty']:.4f}"
                )

                print(
                    f"FINAL    : "
                    f"{result['final']:.4f}"
                )

                print(
                    f"Meaning  : "
                    f"{result['cultural_meaning']}"
                )

                print(
                    f"Tone     : "
                    f"{result['tone']}"
                )


        except KeyboardInterrupt:

            print(
                "\n\nExiting..."
            )

            break


        except Exception as e:

            print(
                f"\nERROR: {e}"
            )