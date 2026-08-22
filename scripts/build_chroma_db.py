import json
import shutil
from pathlib import Path

import numpy as np
import chromadb


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

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "e5_rich_embeddings.npy"
)

CHROMA_DIR = (
    PROJECT_ROOT
    / "data"
    / "chroma_db"
)


# ============================================================
# SETTINGS
# ============================================================

COLLECTION_NAME = "bengali_cultural_expressions"


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("BENGALI CULTURAL RAG - CHROMADB BUILD")
print("=" * 75)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading final dataset...")

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print(
    f"Dataset records: {len(data)}"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(
    EMBEDDINGS_FILE
)

print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# VALIDATION
# ============================================================

if len(data) != 404:
    raise ValueError(
        f"Expected 404 dataset records, "
        f"found {len(data)}"
    )


if embeddings.shape != (404, 768):
    raise ValueError(
        f"Expected (404, 768) embeddings, "
        f"found {embeddings.shape}"
    )


# ============================================================
# CREATE CHROMA DIRECTORY
# ============================================================

CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CREATE CHROMADB CLIENT
# ============================================================

print("\nInitializing ChromaDB...")

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# ============================================================
# REMOVE OLD COLLECTION IF EXISTS
# ============================================================

existing = [
    c.name
    for c in client.list_collections()
]

if COLLECTION_NAME in existing:

    print(
        f"\nExisting collection found: "
        f"{COLLECTION_NAME}"
    )

    print("Deleting old collection...")

    client.delete_collection(
        COLLECTION_NAME
    )


# ============================================================
# CREATE COLLECTION
# ============================================================

print("\nCreating collection...")

collection = client.create_collection(
    name=COLLECTION_NAME,

    metadata={
        "description":
            "Bengali cultural expressions "
            "with rich semantic embeddings",

        "embedding_model":
            "intfloat/multilingual-e5-base",

        "embedding_dimension":
            768,

        "dataset_records":
            404,

        "representation":
            "rich_cultural_passage",
    },

    # IMPORTANT:
    # Disable automatic embedding generation.
    embedding_function=None
)


# ============================================================
# PREPARE RECORDS
# ============================================================

ids = []
documents = []
metadatas = []
embedding_vectors = []


for i, item in enumerate(data):

    phrase = (
        item["bengali_phrase"]
        .strip()
    )

    cultural_meaning = (
        item.get("cultural_meaning")
        or ""
    ).strip()

    tone = (
        item.get("intended_emotion_tone")
        or ""
    ).strip()


    # --------------------------------------------------------
    # Same rich passage used during embedding
    # --------------------------------------------------------

    document = (
        f"বাংলা সাংস্কৃতিক "
        f"অভিব্যক্তি: {phrase}. "
        f"অর্থ: {cultural_meaning}. "
        f"ভাব বা প্রয়োগের ধরন: {tone}."
    )


    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    record_id = str(
        item["id"]
    )


    ids.append(record_id)

    documents.append(document)

    embedding_vectors.append(
        embeddings[i].tolist()
    )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadatas.append({

        "bengali_phrase":
            phrase,

        "cultural_meaning":
            cultural_meaning,

        "intended_emotion_tone":
            tone,

        "review_status":
            item.get(
                "review_status",
                ""
            ),

        "source_record_ids":
            ",".join(
                map(
                    str,
                    item.get(
                        "source_record_ids",
                        []
                    )
                )
            ),
    })


# ============================================================
# ADD TO CHROMADB
# ============================================================

print(
    "\nAdding 404 records to ChromaDB..."
)

collection.add(

    ids=ids,

    documents=documents,

    metadatas=metadatas,

    embeddings=embedding_vectors,
)


# ============================================================
# VERIFY
# ============================================================

count = collection.count()

print(
    "\nCollection record count:",
    count
)


if count != 404:

    raise RuntimeError(
        f"Expected 404 records, "
        f"but ChromaDB contains {count}"
    )


# ============================================================
# SAMPLE RECORD
# ============================================================

sample = collection.get(
    ids=["2"]
)

print("\nSample record:")

print(
    "ID:",
    sample["ids"]
)

print(
    "Document:",
    sample["documents"]
)

print(
    "Metadata:",
    sample["metadatas"]
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("CHROMADB BUILD COMPLETE")
print("=" * 75)

print(
    "\nDatabase location:"
)

print(
    CHROMA_DIR
)

print(
    "\nCollection:"
)

print(
    COLLECTION_NAME
)

print(
    "\nRecords:",
    count
)

print("\n" + "=" * 75)