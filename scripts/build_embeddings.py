# -*- coding: utf-8 -*-

"""
BENGALI CULTURAL RAG
FINAL E5 EMBEDDING BUILDER

Dataset-first
Dynamic dataset size
NO hard-coded record count
NO GEMINI
NO GROQ
NO API KEY

Important:
The embedding row order MUST exactly match
final_cultural_dataset.json row order.
"""

import json
import shutil
from pathlib import Path

import numpy as np
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

EMBEDDING_DIR = (
    BASE_DIR
    / "data"
    / "embeddings"
)

EMBEDDING_PATH = (
    EMBEDDING_DIR
    / "e5_rich_embeddings.npy"
)

MODEL_NAME = "intfloat/multilingual-e5-base"

BATCH_SIZE = 32


# ============================================================
# HELPERS
# ============================================================

def load_dataset(path):

    print("\nLoading final dataset...")

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # Support both:
    # [...]
    #
    # and:
    # {"records": [...]}

    if isinstance(data, list):

        records = data

    elif isinstance(data, dict):

        possible_keys = [
            "records",
            "data",
            "dataset",
            "items",
            "entries"
        ]

        records = None

        for key in possible_keys:

            if isinstance(
                data.get(key),
                list
            ):

                records = data[key]
                break

        if records is None:

            raise ValueError(
                "Could not find dataset records "
                "inside JSON object."
            )

    else:

        raise ValueError(
            "Unsupported dataset JSON format."
        )

    print(
        f"Dataset records: {len(records)}"
    )

    if len(records) == 0:

        raise ValueError(
            "Dataset is empty."
        )

    return records


# ============================================================
# FIELD EXTRACTION
# ============================================================

def get_value(record, keys):

    if not isinstance(record, dict):
        return ""

    for key in keys:

        value = record.get(key)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


def get_phrase(record):

    return get_value(
        record,
        [
            "bengali_phrase",
            "phrase",
            "expression",
            "idiom",
            "idiom_phrase",
            "cultural_expression",
            "cultural_phrase",
            "phrase_bn",
            "expression_bn",
            "headword",
            "term",
            "title",
            "name"
        ]
    )


def get_meaning(record):

    return get_value(
        record,
        [
            "cultural_meaning",
            "meaning",
            "meaning_bn",
            "meaning_en",
            "definition",
            "definition_bn",
            "semantic_meaning"
        ]
    )


def get_literal_meaning(record):

    return get_value(
        record,
        [
            "literal_meaning",
            "literal",
            "literal_meaning_en",
            "literal_translation"
        ]
    )


def get_tone(record):

    return get_value(
        record,
        [
            "tone",
            "tone_category",
            "emotion",
            "sentiment",
            "usage_tone"
        ]
    )


def get_example(record):

    return get_value(
        record,
        [
            "example",
            "example_sentence",
            "example_bn",
            "usage_example",
            "sentence"
        ]
    )


# ============================================================
# BUILD RICH PASSAGES
# ============================================================

def build_passage(record):

    phrase = get_phrase(record)
    meaning = get_meaning(record)
    literal = get_literal_meaning(record)
    tone = get_tone(record)
    example = get_example(record)

    parts = []

    if phrase:
        parts.append(
            f"Expression: {phrase}"
        )

    if meaning:
        parts.append(
            f"Meaning: {meaning}"
        )

    if literal:
        parts.append(
            f"Literal meaning: {literal}"
        )

    if tone:
        parts.append(
            f"Tone: {tone}"
        )

    if example:
        parts.append(
            f"Example: {example}"
        )

    # Fallback:
    # If none of the expected fields exists,
    # use all textual fields from the record.

    if not parts:

        for key, value in record.items():

            if isinstance(value, str):

                value = value.strip()

                if value:

                    parts.append(
                        f"{key}: {value}"
                    )

    return " | ".join(parts)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "BENGALI CULTURAL RAG - FINAL E5 EMBEDDING BUILD"
    )
    print(
        "DYNAMIC DATASET SIZE / NO HARDCODED COUNT"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    records = load_dataset(
        DATASET_PATH
    )

    dataset_count = len(records)

    # --------------------------------------------------------
    # Build passages
    # --------------------------------------------------------

    print(
        "\nBuilding rich passages..."
    )

    passages = []

    empty_count = 0

    for i, record in enumerate(records):

        passage = build_passage(
            record
        )

        if not passage.strip():

            empty_count += 1

            passage = (
                "Expression: "
                + str(record)
            )

        passages.append(
            passage
        )

    print(
        f"Passages created: {len(passages)}"
    )

    if empty_count:

        print(
            f"Warning: {empty_count} records "
            "had no recognized fields."
        )

    # --------------------------------------------------------
    # Show sample
    # --------------------------------------------------------

    print(
        "\nSample embedding passage:"
    )

    print("-" * 75)

    print(
        passages[0][:1000]
    )

    print("-" * 75)

    # --------------------------------------------------------
    # Backup existing embeddings
    # --------------------------------------------------------

    EMBEDDING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if EMBEDDING_PATH.exists():

        backup_path = (
            EMBEDDING_PATH.with_suffix(
                ".old.npy"
            )
        )

        try:

            shutil.copy2(
                EMBEDDING_PATH,
                backup_path
            )

            print(
                f"\nOld embedding backed up:"
            )

            print(
                backup_path
            )

        except Exception as e:

            print(
                f"\nWarning: Could not backup "
                f"old embedding: {e}"
            )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading E5 model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "E5 model loaded."
    )

    # --------------------------------------------------------
    # E5 passage prefix
    # --------------------------------------------------------

    e5_passages = [
        "passage: " + text
        for text in passages
    ]

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    print(
        "\nEncoding dataset..."
    )

    embeddings = model.encode(
        e5_passages,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # HARD SAFETY CHECK
    # --------------------------------------------------------

    if embeddings.shape[0] != dataset_count:

        raise RuntimeError(
            "\nCRITICAL ERROR:\n"
            f"Dataset records = {dataset_count}\n"
            f"Embedding rows   = {embeddings.shape[0]}\n"
            "The counts must be identical."
        )

    if embeddings.ndim != 2:

        raise RuntimeError(
            "Embedding matrix must be 2-dimensional."
        )

    # Expected multilingual-e5-base dimension
    if embeddings.shape[1] != 768:

        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"{embeddings.shape[1]}. "
            "Expected 768 for multilingual-e5-base."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        EMBEDDING_PATH,
        embeddings
    )

    print(
        "\nEmbedding saved:"
    )

    print(
        EMBEDDING_PATH
    )

    # --------------------------------------------------------
    # Verify saved file
    # --------------------------------------------------------

    print(
        "\nVerifying saved embedding..."
    )

    saved = np.load(
        EMBEDDING_PATH
    )

    print(
        f"Saved shape: {saved.shape}"
    )

    if saved.shape[0] != dataset_count:

        raise RuntimeError(
            "Saved embedding count does not "
            "match dataset count."
        )

    print("\n" + "=" * 75)

    print(
        "EMBEDDING BUILD COMPLETE"
    )

    print("=" * 75)

    print(
        f"Dataset records : {dataset_count}"
    )

    print(
        f"Embedding rows  : {saved.shape[0]}"
    )

    print(
        f"Embedding dim   : {saved.shape[1]}"
    )

    print(
        "Status          : OK"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()