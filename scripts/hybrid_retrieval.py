import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


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

SEMANTIC_TOP_K = 15
BM25_TOP_K = 15

FINAL_TOP_K = 10


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Basic Bengali-friendly text normalization.
    """

    if not text:
        return ""

    text = str(text).strip().lower()

    # Normalize common punctuation
    text = re.sub(
        r"[।,!?;:()\[\]{}\"“”‘’]",
        " ",
        text
    )

    # Remove multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Simple whitespace tokenizer for BM25.
    """

    text = normalize_text(
        text
    )

    if not text:
        return []

    return text.split()


# ============================================================
# RICH PASSAGE
# ============================================================

def build_rich_passage(record):
    """
    Build a rich textual representation of
    a Bengali cultural expression.
    """

    phrase = record.get(
        "bengali_phrase",
        ""
    )

    literal = record.get(
        "literal_meaning",
        ""
    )

    cultural = record.get(
        "cultural_meaning",
        ""
    )

    tone = record.get(
        "intended_emotion_tone",
        ""
    )

    return (
        f"বাংলা সাংস্কৃতিক অভিব্যক্তি: {phrase}. "
        f"আক্ষরিক অর্থ: {literal}. "
        f"সাংস্কৃতিক অর্থ: {cultural}. "
        f"ভাব বা প্রয়োগের ধরন: {tone}."
    )


# ============================================================
# WEIGHTED HYBRID SCORE
# ============================================================

def hybrid_score(
    semantic_score,
    bm25_score,
    max_bm25_score
):
    """
    Weighted hybrid score.

    Semantic retrieval gets higher priority because
    Bengali natural-language queries often have little
    lexical overlap with idioms.
    """

    semantic_weight = 0.75
    bm25_weight = 0.25

    if max_bm25_score > 0:

        bm25_norm = (
            bm25_score
            / max_bm25_score
        )

    else:

        bm25_norm = 0.0

    return (
        semantic_weight
        * semantic_score
        +
        bm25_weight
        * bm25_norm
    )


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(self):

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        print(
            "Loading final dataset..."
        )

        with open(
            DATASET_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            self.records = json.load(
                f
            )

        print(
            f"Dataset records: "
            f"{len(self.records)}"
        )


        # ----------------------------------------------------
        # Load E5 embeddings
        # ----------------------------------------------------

        print(
            "\nLoading E5 embeddings..."
        )

        self.embeddings = np.load(
            EMBEDDING_PATH
        )

        print(
            f"Embedding matrix: "
            f"{self.embeddings.shape}"
        )


        # ----------------------------------------------------
        # Normalize embeddings
        # ----------------------------------------------------

        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True
        )

        self.embeddings = (
            self.embeddings
            / np.maximum(
                norms,
                1e-12
            )
        )


        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        print(
            "\nLoading embedding model..."
        )

        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

        print(
            "Embedding model loaded."
        )


        # ----------------------------------------------------
        # Build BM25 corpus
        # ----------------------------------------------------

        print(
            "\nBuilding BM25 index..."
        )

        self.passages = [
            build_rich_passage(
                record
            )
            for record in self.records
        ]

        self.tokenized_corpus = [
            tokenize(
                passage
            )
            for passage in self.passages
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print(
            "BM25 index ready."
        )


        # ----------------------------------------------------
        # ID mapping
        # ----------------------------------------------------

        self.id_to_index = {
            str(record["id"]): index
            for index, record in enumerate(
                self.records
            )
        }

        print(
            "\nHybrid retriever initialized."
        )


    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def semantic_search(
        self,
        query,
        top_k=SEMANTIC_TOP_K
    ):

        query_embedding = self.model.encode(
            f"query: {query}",
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            record = self.records[
                index
            ]

            results.append(
                {
                    "id":
                        str(record["id"]),

                    "index":
                        int(index),

                    "score":
                        float(scores[index])
                }
            )

        return results


    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def bm25_search(
        self,
        query,
        top_k=BM25_TOP_K
    ):

        query_tokens = tokenize(
            query
        )

        if not query_tokens:
            return []


        scores = self.bm25.get_scores(
            query_tokens
        )


        # ----------------------------------------------------
        # Only keep documents with actual BM25 match
        # ----------------------------------------------------

        valid_indices = [
            i
            for i, score in enumerate(
                scores
            )
            if score > 0
        ]


        if not valid_indices:
            return []


        valid_indices = sorted(
            valid_indices,
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]


        results = []

        for index in valid_indices:

            record = self.records[
                index
            ]

            results.append(
                {
                    "id":
                        str(record["id"]),

                    "index":
                        int(index),

                    "score":
                        float(scores[index])
                }
            )

        return results


    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def search(
        self,
        query,
        semantic_top_k=SEMANTIC_TOP_K,
        bm25_top_k=BM25_TOP_K,
        final_top_k=FINAL_TOP_K
    ):

        # ----------------------------------------------------
        # 1. Semantic retrieval
        # ----------------------------------------------------

        semantic_results = self.semantic_search(
            query,
            semantic_top_k
        )


        # ----------------------------------------------------
        # 2. BM25 retrieval
        # ----------------------------------------------------

        bm25_results = self.bm25_search(
            query,
            bm25_top_k
        )


        # ----------------------------------------------------
        # 3. Create candidate pool
        # ----------------------------------------------------

        candidates = {}


        # ----------------------------------------------------
        # Add semantic candidates
        # ----------------------------------------------------

        for item in semantic_results:

            doc_id = item["id"]

            candidates[doc_id] = {
                "id":
                    doc_id,

                "index":
                    item["index"],

                "semantic_score":
                    item["score"],

                "bm25_score":
                    0.0
            }


        # ----------------------------------------------------
        # Add BM25 candidates
        # ----------------------------------------------------

        for item in bm25_results:

            doc_id = item["id"]


            if doc_id not in candidates:

                candidates[doc_id] = {
                    "id":
                        doc_id,

                    "index":
                        item["index"],

                    "semantic_score":
                        0.0,

                    "bm25_score":
                        item["score"]
                }

            else:

                candidates[doc_id][
                    "bm25_score"
                ] = item["score"]


        # ----------------------------------------------------
        # Convert dictionary to list
        # ----------------------------------------------------

        candidates = list(
            candidates.values()
        )


        # ----------------------------------------------------
        # 4. Find maximum BM25 score
        # ----------------------------------------------------

        max_bm25 = max(
            [
                candidate["bm25_score"]
                for candidate in candidates
            ],
            default=0.0
        )


        # ----------------------------------------------------
        # 5. Calculate weighted hybrid score
        # ----------------------------------------------------

        for candidate in candidates:

            candidate["hybrid_score"] = (
                hybrid_score(
                    candidate["semantic_score"],
                    candidate["bm25_score"],
                    max_bm25
                )
            )


            # ------------------------------------------------
            # BM25-only penalty
            # ------------------------------------------------
            #
            # If a result comes only from BM25 and has
            # no semantic score, reduce its final score.
            #

            if (
                candidate["semantic_score"] == 0
                and candidate["bm25_score"] > 0
            ):

                candidate["hybrid_score"] *= 0.70


        # ----------------------------------------------------
        # 6. Sort candidates by hybrid score
        # ----------------------------------------------------

        candidates.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )


        # ----------------------------------------------------
        # 7. Build final results
        # ----------------------------------------------------

        results = []

        for candidate in candidates[
            :final_top_k
        ]:

            index = candidate["index"]

            record = self.records[
                index
            ]


            results.append(
                {
                    "id":
                        candidate["id"],

                    "phrase":
                        record[
                            "bengali_phrase"
                        ],

                    "cultural_meaning":
                        record[
                            "cultural_meaning"
                        ],

                    "intended_emotion_tone":
                        record[
                            "intended_emotion_tone"
                        ],

                    "semantic_score":
                        candidate[
                            "semantic_score"
                        ],

                    "bm25_score":
                        candidate[
                            "bm25_score"
                        ],

                    "hybrid_score":
                        candidate[
                            "hybrid_score"
                        ]
                }
            )

        return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "BENGALI CULTURAL RAG - "
        "WEIGHTED HYBRID RETRIEVAL"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Initialize retriever
    # --------------------------------------------------------

    retriever = HybridRetriever()


    # --------------------------------------------------------
    # Query loop
    # --------------------------------------------------------

    while True:

        print(
            "\nআপনার প্রশ্ন লিখুন:"
        )

        query = input(
            "> "
        ).strip()


        if not query:

            continue


        if query.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            break


        # ----------------------------------------------------
        # Run hybrid search
        # ----------------------------------------------------

        results = retriever.search(
            query
        )


        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "WEIGHTED HYBRID RETRIEVAL RESULTS"
        )

        print(
            "=" * 70
        )


        if not results:

            print(
                "\nকোনো relevant result পাওয়া যায়নি।"
            )

            continue


        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\n[{i}] "
                f"{result['phrase']}"
            )

            print(
                f"Semantic : "
                f"{result['semantic_score']:.4f}"
            )

            print(
                f"BM25     : "
                f"{result['bm25_score']:.4f}"
            )

            print(
                f"Hybrid   : "
                f"{result['hybrid_score']:.4f}"
            )

            print(
                f"Meaning  : "
                f"{result['cultural_meaning']}"
            )

            print(
                f"Tone     : "
                f"{result['intended_emotion_tone']}"
            )