import sys
from pathlib import Path


# Allow imports from scripts directory
sys.path.append(
    str(Path(__file__).parent)
)

from llm_adapter import GeminiLLM

from relevance_selector import (
    select_relevant_context
)


# ============================================================
# CONFIG
# ============================================================

TOP_K = 10
SELECTED_K = 4


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    query,
    selected_results
):

    context_parts = []

    for i, item in enumerate(
        selected_results,
        start=1
    ):

        context_parts.append(
            f"""
[{i}]

বাংলা অভিব্যক্তি:
{item["bengali_phrase"]}

অর্থ:
{item["cultural_meaning"]}

ভাব/প্রয়োগের ধরন:
{item["intended_emotion_tone"]}

Similarity:
{item["similarity"]:.4f}
""".strip()
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
তুমি একজন Bengali Cultural Assistant।

তোমার কাজ হলো ব্যবহারকারীর প্রশ্নের উত্তর
শুধুমাত্র প্রদত্ত সাংস্কৃতিক context-এর ভিত্তিতে
দেওয়া।

==============================
ব্যবহারকারীর প্রশ্ন
==============================

{query}


==============================
RETRIEVED CULTURAL CONTEXT
==============================

{context}


==============================
নির্দেশনা
==============================

1. শুধুমাত্র প্রদত্ত context ব্যবহার করবে।

2. Context-এর বাইরে কোনো বাংলা প্রবাদ,
   বাগধারা, সাংস্কৃতিক অর্থ বা তথ্য বানাবে না।

3. ব্যবহারকারীর প্রশ্নের সঙ্গে সবচেয়ে
   প্রাসঙ্গিক বাংলা অভিব্যক্তিটি প্রথমে
   উল্লেখ করবে।

4. কেন এটি সবচেয়ে উপযুক্ত তা সংক্ষেপে
   ব্যাখ্যা করবে।

5. প্রয়োজনে context থেকে সর্বোচ্চ
   2টি related expression উল্লেখ করবে।

6. অপ্রাসঙ্গিক expression উল্লেখ করবে না।

7. Expression-এর নাম হুবহু লিখবে।

8. উত্তর সম্পূর্ণ বাংলায় দেবে।

9. উত্তর সংক্ষিপ্ত, স্বাভাবিক এবং
   সহজবোধ্য রাখবে।

10. Context-এ যথেষ্ট তথ্য না থাকলে
    স্পষ্টভাবে বলবে যে প্রদত্ত সাংস্কৃতিক
    context থেকে নিশ্চিত উত্তর দেওয়া যাচ্ছে না।


==============================
EXPECTED ANSWER FORMAT
==============================

সবচেয়ে উপযুক্ত অভিব্যক্তি:
<expression>

অর্থ:
<meaning>

কেন উপযুক্ত:
<short explanation>

সম্পর্কিত অভিব্যক্তি:
<optional related expressions>
""".strip()

    return prompt


# ============================================================
# MAIN RAG ENGINE
# ============================================================

def run_rag(
    query,
    collection,
    embedding_model
):

    # --------------------------------------------------------
    # 1. Encode query
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()


    # --------------------------------------------------------
    # 2. Chroma retrieval
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


    # --------------------------------------------------------
    # 3. Parse results
    # --------------------------------------------------------

    retrieved = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        similarity = 1 - distance

        retrieved.append(
            {
                "document": document,

                "bengali_phrase":
                    metadata["bengali_phrase"],

                "cultural_meaning":
                    metadata["cultural_meaning"],

                "intended_emotion_tone":
                    metadata["intended_emotion_tone"],

                "similarity":
                    similarity
            }
        )


    # --------------------------------------------------------
    # 4. Select relevant context
    # --------------------------------------------------------

    scored_retrieved, selected = (
        select_relevant_context(
            query=query,
            retrieved_items=retrieved,
            max_results=SELECTED_K,
            min_score=0.45
        )
    )


    # --------------------------------------------------------
    # 5. Build grounded prompt
    # --------------------------------------------------------

    prompt = build_gemini_prompt(
        query,
        selected
    )


    # --------------------------------------------------------
    # 6. Gemini generation
    # --------------------------------------------------------

    llm = GeminiLLM()

    answer = llm.generate(
        prompt
    )


    return {
        "query": query,

        "retrieved": retrieved,

        "scored": scored_retrieved,

        "selected": selected,

        "prompt": prompt,

        "answer": answer
    }


# ============================================================
# RELEVANCE RERANKING DISPLAY
# ============================================================

def display_reranking(
    result
):

    print("\n")
    print("=" * 70)
    print("RELEVANCE RERANKING")
    print("=" * 70)

    for i, item in enumerate(
        result["scored"],
        start=1
    ):

        print(
            f"\n[{i}] "
            f"{item['bengali_phrase']}"
        )

        print(
            f"Semantic      : "
            f"{item['semantic_score']:.4f}"
        )

        print(
            f"Keyword       : "
            f"{item['keyword_score']:.4f}"
        )

        print(
            f"Meaning       : "
            f"{item['meaning_score']:.4f}"
        )

        print(
            f"Phrase        : "
            f"{item['phrase_score']:.4f}"
        )

        print(
            f"FINAL SCORE   : "
            f"{item['relevance_score']:.4f}"
        )


# ============================================================
# DISPLAY
# ============================================================

def display_result(
    result
):

    print("\n")
    print("=" * 70)
    print("RETRIEVED CULTURAL EXPRESSIONS")
    print("=" * 70)

    for i, item in enumerate(
        result["retrieved"],
        start=1
    ):

        print(
            f"\n[{i}] "
            f"{item['bengali_phrase']}"
        )

        print(
            f"Similarity: "
            f"{item['similarity']:.4f}"
        )

        print(
            f"Meaning: "
            f"{item['cultural_meaning']}"
        )

        print(
            f"Tone: "
            f"{item['intended_emotion_tone']}"
        )


    print("\n")
    print("=" * 70)
    print("SELECTED CONTEXT")
    print("=" * 70)

    for i, item in enumerate(
        result["selected"],
        start=1
    ):

        print(
            f"\n[{i}] "
            f"{item['bengali_phrase']}"
        )

        print(
            f"Similarity: "
            f"{item['similarity']:.4f}"
        )

        print(
            f"Meaning: "
            f"{item['cultural_meaning']}"
        )


    print("\n")
    print("=" * 70)
    print("GEMINI ANSWER")
    print("=" * 70)

    print(
        result["answer"]
    )

    print("\n")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("BENGALI CULTURAL RAG + GEMINI")
    print("=" * 70)


    # --------------------------------------------------------
    # Import existing RAG components
    # --------------------------------------------------------

    from sentence_transformers import (
        SentenceTransformer
    )

    import chromadb


    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embedding_model = SentenceTransformer(
        "intfloat/multilingual-e5-base"
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
        f"Collection loaded: "
        f"{collection.name}"
    )

    print(
        f"Records: "
        f"{collection.count()}"
    )


    # --------------------------------------------------------
    # User query
    # --------------------------------------------------------

    print("\n")

    query = input(
        "আপনার প্রশ্ন লিখুন: "
    ).strip()


    if not query:

        print(
            "কোনো প্রশ্ন দেওয়া হয়নি।"
        )

        sys.exit(0)


    # --------------------------------------------------------
    # Run RAG
    # --------------------------------------------------------

    try:

        result = run_rag(
            query,
            collection,
            embedding_model
        )


        # ----------------------------------------------------
        # Display reranking scores
        # ----------------------------------------------------

        display_reranking(
            result
        )


        # ----------------------------------------------------
        # Display final result
        # ----------------------------------------------------

        display_result(
            result
        )


    except Exception as e:

        print("\n")
        print("=" * 70)
        print("RAG ERROR")
        print("=" * 70)

        print(
            f"\nError type: "
            f"{type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )

        print(
            "\nPlease check the error above."
        )