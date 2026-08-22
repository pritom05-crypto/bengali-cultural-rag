from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "intfloat/multilingual-e5-base"


print("=" * 70)
print("BENGALI CULTURAL RAG - EMBEDDING MODEL TEST")
print("=" * 70)

print(f"\nLoading model: {MODEL_NAME}")
print("Device: CPU")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("\nModel loaded successfully.")

# ------------------------------------------------------------
# Test sentences
# ------------------------------------------------------------

sentences = [
    "query: আমি খুব বিপদে পড়েছি",
    "passage: অথৈ জলে পড়া",
    "passage: অকূল পাথার",
    "passage: আনন্দে আত্মহারা",
    "passage: এক ঢিলে দু’পাখি"
]

# ------------------------------------------------------------
# Generate embeddings
# ------------------------------------------------------------

embeddings = model.encode(
    sentences,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("\nEmbedding shape:")
print(embeddings.shape)

# ------------------------------------------------------------
# Check numerical validity
# ------------------------------------------------------------

print("\nEmbedding dtype:")
print(embeddings.dtype)

print("\nFirst vector - first 10 values:")
print(embeddings[0][:10])

print("\nContains NaN:")
print(np.isnan(embeddings).any())

print("Contains Inf:")
print(np.isinf(embeddings).any())

# ------------------------------------------------------------
# Similarity
# ------------------------------------------------------------

query_embedding = embeddings[0]

scores = np.dot(
    embeddings[1:],
    query_embedding
)

print("\n" + "=" * 70)
print("SEMANTIC SIMILARITY TEST")
print("=" * 70)

results = [
    ("অথৈ জলে পড়া", scores[0]),
    ("অকূল পাথার", scores[1]),
    ("আনন্দে আত্মহারা", scores[2]),
    ("এক ঢিলে দু’পাখি", scores[3]),
]

results.sort(
    key=lambda x: x[1],
    reverse=True
)

for phrase, score in results:
    print(
        f"{phrase:<25} -> {score:.4f}"
    )

print("\n" + "=" * 70)
print("EMBEDDING TEST COMPLETE")
print("=" * 70)