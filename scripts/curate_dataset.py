import json
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "idioms_bangla_to_english_dataset.json"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "cultural_expressions.json"
CONFLICT_FILE = PROCESSED_DIR / "unresolved_conflicts.json"


# ============================================================
# DUPLICATE GROUP DECISIONS
# ============================================================

# These groups are safe to merge.
SAFE_MERGE = {
    "আড়ং ঘাটা",
    "আদাড়ের হাঁড়ি",
    "ইতর বিশেষ",
    "ঈদের চাঁদ",
    "ইঁদুর কপালে",
    "ইলশে গুঁড়ি",
    "ইতর",
    "ইন্দ্রপতন",
    "ইষ্টনাম জপা",
    "ইতুনিদকুঁড়ে",
    "ইশপিশ করা",
    "ইকড়ি-মিকড়ি",
    "ইল্লতে কাণ্ড",
    "ইতস্তত করা",
    "ইন্দ্রের শচী",
    "এককে একুশ করা",
    "কলির সন্ধ্যা",
    "কিপটের জাসু",
}

# These need review before merging.
REVIEW_MERGE = {
    "ইঁচড়ে পাকা",
    "ইয়ারবকসি",
}

# This phrase has conflicting meanings.
CONFLICTING = {
    "ইনিয়ে বিনিয়ে",
}


# ============================================================
# HELPERS
# ============================================================

def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_phrase(value):
    value = clean(value)
    value = " ".join(value.split())
    value = value.replace(" / ", "/")
    return value


def split_tone(tone):
    tone = clean(tone)

    if not tone:
        return "", ""

    parts = [
        clean(x)
        for x in tone.split("/")
        if clean(x)
    ]

    if len(parts) == 1:
        return parts[0], ""

    return parts[0], " / ".join(parts[1:])


def expression_type(record):
    """
    Conservative classification.

    We intentionally do NOT try to infer a highly specific
    cultural category from the English meaning alone.
    """

    phrase = clean(record.get("bengali_phrase"))

    # Known proverb
    if phrase == "এক মাঘে শীত যায় না":
        return "Proverb"

    if phrase == "উনো বর্ষায় দুনো শীত":
        return "Proverb"

    # Known religious/cultural expression
    if phrase in {
        "ইষ্টনাম জপা",
        "ঊর্ধ্বলোক",
        "ঊর্ধ্বদেহ",
    }:
        return "Religious/Cultural Expression"

    # Known poetic expression
    if phrase in {
        "ইলশে গুঁড়ি",
        "ঊর্মিমালী",
        "ঊর্মিভঙ্গ",
    }:
        return "Poetic Expression"

    # Default
    return "Cultural Expression"


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_data():

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_FILE}"
        )

    with open(
        RAW_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Raw dataset must be a JSON list."
        )

    return data


# ============================================================
# GROUP RECORDS BY NORMALIZED PHRASE
# ============================================================

def group_records(data):

    groups = defaultdict(list)

    for record in data:

        phrase = normalize_phrase(
            record.get("bengali_phrase")
        )

        if phrase:
            groups[phrase].append(record)

    return groups


# ============================================================
# CREATE CANONICAL RECORD
# ============================================================

def create_canonical_record(
    records,
    status="verified"
):

    # Use first record as canonical source.
    first = records[0]

    raw_tones = []

    for record in records:

        tone = clean(
            record.get(
                "intended_emotion_tone"
            )
        )

        if tone and tone not in raw_tones:
            raw_tones.append(tone)

    # Primary/secondary tone from first record.
    primary, secondary = split_tone(
        first.get(
            "intended_emotion_tone"
        )
    )

    source_ids = [
        clean(record.get("id"))
        for record in records
    ]

    canonical = {
        "id": clean(first.get("id")),

        "bengali_phrase":
            clean(first.get("bengali_phrase")),

        "variants": [],

        "literal_meaning":
            clean(first.get("literal_meaning")),

        "cultural_meaning":
            clean(first.get("cultural_meaning")),

        "intended_emotion_tone":
            clean(first.get(
                "intended_emotion_tone"
            )),

        "primary_tone": primary,

        "secondary_tone": secondary,

        "expression_type":
            expression_type(first),

        "review_status": status,

        "source_record_ids": source_ids,

        "raw_tones": raw_tones,
    }

    return canonical


# ============================================================
# MAIN CURATION
# ============================================================

def main():

    print("=" * 70)
    print("BENGALI CULTURAL RAG - DATASET CURATION")
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data = load_data()

    print(f"\nRaw records: {len(data)}")

    groups = group_records(data)

    curated = []
    conflicts = []
    review_candidates = []

    # --------------------------------------------------------
    # Process every phrase group
    # --------------------------------------------------------

    for phrase, records in groups.items():

        # ----------------------------------------------------
        # Single record
        # ----------------------------------------------------

        if len(records) == 1:

            record = records[0]

            primary, secondary = split_tone(
                record.get(
                    "intended_emotion_tone"
                )
            )

            item = {
                "id": clean(record.get("id")),

                "bengali_phrase":
                    clean(record.get(
                        "bengali_phrase"
                    )),

                "variants": [],

                "literal_meaning":
                    clean(record.get(
                        "literal_meaning"
                    )),

                "cultural_meaning":
                    clean(record.get(
                        "cultural_meaning"
                    )),

                "intended_emotion_tone":
                    clean(record.get(
                        "intended_emotion_tone"
                    )),

                "primary_tone": primary,

                "secondary_tone": secondary,

                "expression_type":
                    expression_type(record),

                "review_status": "verified",

                "source_record_ids": [
                    clean(record.get("id"))
                ],

                "raw_tones": [
                    clean(record.get(
                        "intended_emotion_tone"
                    ))
                ],
            }

            curated.append(item)

            continue

        # ----------------------------------------------------
        # Conflicting meaning
        # ----------------------------------------------------

        if phrase in CONFLICTING:

            for record in records:

                primary, secondary = split_tone(
                    record.get(
                        "intended_emotion_tone"
                    )
                )

                item = {
                    "id": clean(record.get("id")),

                    "bengali_phrase":
                        clean(record.get(
                            "bengali_phrase"
                        )),

                    "variants": [],

                    "literal_meaning":
                        clean(record.get(
                            "literal_meaning"
                        )),

                    "cultural_meaning":
                        clean(record.get(
                            "cultural_meaning"
                        )),

                    "intended_emotion_tone":
                        clean(record.get(
                            "intended_emotion_tone"
                        )),

                    "primary_tone": primary,

                    "secondary_tone": secondary,

                    "expression_type":
                        expression_type(record),

                    "review_status":
                        "needs_review_conflicting_meaning",

                    "source_record_ids": [
                        clean(record.get("id"))
                    ],

                    "raw_tones": [
                        clean(record.get(
                            "intended_emotion_tone"
                        ))
                    ],
                }

                curated.append(item)

            conflicts.append({
                "bengali_phrase": phrase,
                "records": [
                    {
                        "id": clean(r.get("id")),
                        "cultural_meaning":
                            clean(r.get(
                                "cultural_meaning"
                            )),
                        "intended_emotion_tone":
                            clean(r.get(
                                "intended_emotion_tone"
                            )),
                    }
                    for r in records
                ],
            })

            continue

        # ----------------------------------------------------
        # Review merge candidates
        # ----------------------------------------------------

        if phrase in REVIEW_MERGE:

            item = create_canonical_record(
                records,
                status="needs_review_merge"
            )

            review_candidates.append({
                "bengali_phrase": phrase,
                "source_record_ids": [
                    clean(r.get("id"))
                    for r in records
                ],
            })

            curated.append(item)

            continue

        # ----------------------------------------------------
        # Safe duplicate merge
        # ----------------------------------------------------

        if phrase in SAFE_MERGE:

            item = create_canonical_record(
                records,
                status="merged_duplicate"
            )

            curated.append(item)

            continue

        # ----------------------------------------------------
        # Unexpected duplicate
        # ----------------------------------------------------

        for record in records:

            primary, secondary = split_tone(
                record.get(
                    "intended_emotion_tone"
                )
            )

            item = {
                "id": clean(record.get("id")),

                "bengali_phrase":
                    clean(record.get(
                        "bengali_phrase"
                    )),

                "variants": [],

                "literal_meaning":
                    clean(record.get(
                        "literal_meaning"
                    )),

                "cultural_meaning":
                    clean(record.get(
                        "cultural_meaning"
                    )),

                "intended_emotion_tone":
                    clean(record.get(
                        "intended_emotion_tone"
                    )),

                "primary_tone": primary,

                "secondary_tone": secondary,

                "expression_type":
                    expression_type(record),

                "review_status":
                    "needs_review_duplicate",

                "source_record_ids": [
                    clean(record.get("id"))
                ],

                "raw_tones": [
                    clean(record.get(
                        "intended_emotion_tone"
                    ))
                ],
            }

            curated.append(item)

    # ========================================================
    # SAVE CURATED DATASET
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            curated,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # SAVE CONFLICT REPORT
    # ========================================================

    conflict_report = {
        "total_conflicting_groups":
            len(conflicts),

        "conflicts": conflicts,

        "review_merge_candidates":
            review_candidates,
    }

    with open(
        CONFLICT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            conflict_report,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    status_counts = defaultdict(int)

    for item in curated:
        status_counts[
            item["review_status"]
        ] += 1

    print("\n" + "=" * 70)
    print("CURATION SUMMARY")
    print("=" * 70)

    print(
        f"Raw records             : {len(data)}"
    )

    print(
        f"Curated records         : {len(curated)}"
    )

    print(
        f"Conflicting groups      : {len(conflicts)}"
    )

    print(
        f"Review merge candidates : "
        f"{len(review_candidates)}"
    )

    print("\nStatus distribution:")

    for status, count in status_counts.items():

        print(
            f"  {status:<35}: {count}"
        )

    print("\nGenerated files:")

    print(
        f"1. {OUTPUT_FILE}"
    )

    print(
        f"2. {CONFLICT_FILE}"
    )

    print("\nRAW DATASET WAS NOT MODIFIED.")

    print("=" * 70)


if __name__ == "__main__":
    main()