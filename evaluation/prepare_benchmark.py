import json
import random
from pathlib import Path


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

OUTPUT_PATH = (
    BASE_DIR
    / "evaluation"
    / "benchmark_queries.json"
)

N_SAMPLES = 120

RANDOM_SEED = 42


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("BENGALI CULTURAL RAG — BENCHMARK PREPARATION")
print("=" * 70)

print("\nLoading dataset...")

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
# CLEAN RECORDS
# ============================================================

records = []

for idx, item in enumerate(dataset):

    phrase = str(
        item.get(
            "bengali_phrase",
            ""
        )
    ).strip()

    meaning = str(
        item.get(
            "cultural_meaning",
            ""
        )
    ).strip()

    literal = str(
        item.get(
            "literal_meaning",
            ""
        )
    ).strip()

    tone = str(
        item.get(
            "intended_emotion_tone",
            ""
        )
    ).strip()

    if not phrase:
        continue

    records.append({

        "dataset_index": idx,

        "gold_phrase": phrase,

        "cultural_meaning": meaning,

        "literal_meaning": literal,

        "tone": tone,

        # Human annotator will fill this.
        "query": "",

        # Optional:
        "difficulty": "",

        # Keep true only after human verification.
        "verified": False,

        "notes": ""
    })


# ============================================================
# SAMPLE
# ============================================================

random.seed(
    RANDOM_SEED
)

if len(records) > N_SAMPLES:

    selected = random.sample(
        records,
        N_SAMPLES
    )

else:

    selected = records


# ============================================================
# SORT
# ============================================================

selected.sort(
    key=lambda x: x["dataset_index"]
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        selected,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("BENCHMARK TEMPLATE CREATED")
print("=" * 70)

print(
    f"Selected records: {len(selected)}"
)

print()
print(
    f"Saved:"
)

print(
    OUTPUT_PATH
)

print()
print("IMPORTANT:")
print(
    "Fill the 'query' field manually using natural Bengali "
    "sentences."
)

print()
print("Example:")
print(
    '\"query\": \"সে খুব অলস\"'
)

print(
    '\"gold_phrase\": \"কুঁড়ের বাঘ\"'
)

print()
print(
    "After annotation, set:"
)

print(
    '\"verified\": true'
)