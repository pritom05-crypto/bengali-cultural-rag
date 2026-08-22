import json
import re
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "idioms_bangla_to_english_dataset.json"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

AUDIT_REPORT = PROCESSED_DIR / "audit_report.json"
REVIEW_QUEUE = PROCESSED_DIR / "review_queue.json"
DUPLICATE_REPORT = PROCESSED_DIR / "duplicate_groups.json"


# ============================================================
# REQUIRED FIELDS
# ============================================================

REQUIRED_FIELDS = [
    "id",
    "bengali_phrase",
    "literal_meaning",
    "cultural_meaning",
    "intended_emotion_tone",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """Convert value to a clean string."""
    if value is None:
        return ""

    return str(value).strip()


def normalize_phrase(value):
    """
    Normalize phrase only for comparison.
    IMPORTANT:
    This does NOT modify the original dataset.
    """
    value = clean_text(value)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value)

    # Normalize spaces around slash
    value = re.sub(r"\s*/\s*", "/", value)

    return value


def is_short_phrase(value):
    return len(clean_text(value)) < 4


def is_short_meaning(value):
    return len(clean_text(value)) < 12


def contains_variant_separator(value):
    """
    Detect phrases containing multiple variants.
    """
    value = clean_text(value)

    return "/" in value


def split_tone(tone):
    """
    Split raw tone such as:

        Sarcastic / Mocking

    into primary and secondary tone.

    We preserve the original raw tone separately.
    """

    tone = clean_text(tone)

    if not tone:
        return "", ""

    parts = [
        clean_text(part)
        for part in tone.split("/")
        if clean_text(part)
    ]

    if len(parts) == 1:
        return parts[0], ""

    return parts[0], " / ".join(parts[1:])


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_FILE}"
        )

    with open(RAW_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")

    return data


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_records(data):

    missing_fields = []
    duplicate_ids = []

    seen_ids = set()

    for index, record in enumerate(data):

        # Check object type
        if not isinstance(record, dict):
            missing_fields.append({
                "index": index,
                "reason": "Record is not a JSON object",
            })
            continue

        # Required fields
        missing = [
            field
            for field in REQUIRED_FIELDS
            if not clean_text(record.get(field))
        ]

        if missing:
            missing_fields.append({
                "index": index,
                "id": record.get("id"),
                "missing_fields": missing,
            })

        # Duplicate ID
        record_id = clean_text(record.get("id"))

        if record_id in seen_ids:
            duplicate_ids.append(record_id)

        seen_ids.add(record_id)

    return {
        "missing_fields": missing_fields,
        "duplicate_ids": duplicate_ids,
    }


# ============================================================
# DUPLICATE PHRASE DETECTION
# ============================================================

def find_duplicate_phrases(data):

    groups = defaultdict(list)

    for record in data:

        phrase = normalize_phrase(
            record.get("bengali_phrase", "")
        )

        if phrase:
            groups[phrase].append(record)

    duplicates = {}

    for phrase, records in groups.items():

        if len(records) > 1:

            duplicates[phrase] = [
                {
                    "id": clean_text(record.get("id")),
                    "bengali_phrase": clean_text(
                        record.get("bengali_phrase")
                    ),
                    "literal_meaning": clean_text(
                        record.get("literal_meaning")
                    ),
                    "cultural_meaning": clean_text(
                        record.get("cultural_meaning")
                    ),
                    "intended_emotion_tone": clean_text(
                        record.get("intended_emotion_tone")
                    ),
                }
                for record in records
            ]

    return duplicates


# ============================================================
# VARIANT DETECTION
# ============================================================

def find_variant_records(data):

    variants = []

    for record in data:

        phrase = clean_text(
            record.get("bengali_phrase")
        )

        if contains_variant_separator(phrase):

            variants.append({
                "id": clean_text(record.get("id")),
                "bengali_phrase": phrase,
                "cultural_meaning": clean_text(
                    record.get("cultural_meaning")
                ),
                "intended_emotion_tone": clean_text(
                    record.get("intended_emotion_tone")
                ),
            })

    return variants


# ============================================================
# TONE ANALYSIS
# ============================================================

def analyze_tones(data):

    raw_tones = Counter()
    primary_tones = Counter()

    tone_mapping = []

    for record in data:

        raw = clean_text(
            record.get("intended_emotion_tone")
        )

        primary, secondary = split_tone(raw)

        if raw:
            raw_tones[raw] += 1

        if primary:
            primary_tones[primary] += 1

        tone_mapping.append({
            "id": clean_text(record.get("id")),
            "bengali_phrase": clean_text(
                record.get("bengali_phrase")
            ),
            "raw_tone": raw,
            "primary_tone": primary,
            "secondary_tone": secondary,
        })

    return {
        "raw_tone_distribution": dict(raw_tones),
        "primary_tone_distribution": dict(primary_tones),
        "tone_mapping": tone_mapping,
    }


# ============================================================
# REVIEW QUEUE
# ============================================================

def build_review_queue(data, duplicate_phrases):

    review_queue = []

    duplicate_ids = set()

    for records in duplicate_phrases.values():

        for record in records:
            duplicate_ids.add(
                clean_text(record.get("id"))
            )

    for record in data:

        record_id = clean_text(record.get("id"))
        phrase = clean_text(record.get("bengali_phrase"))
        literal = clean_text(record.get("literal_meaning"))
        cultural = clean_text(record.get("cultural_meaning"))
        tone = clean_text(record.get("intended_emotion_tone"))

        reasons = []

        # ----------------------------------------------------
        # Missing fields
        # ----------------------------------------------------

        missing = [
            field
            for field in REQUIRED_FIELDS
            if not clean_text(record.get(field))
        ]

        if missing:
            reasons.append("missing_required_field")

        # ----------------------------------------------------
        # Duplicate phrase
        # ----------------------------------------------------

        if record_id in duplicate_ids:
            reasons.append("duplicate_phrase")

        # ----------------------------------------------------
        # Variant phrase
        # ----------------------------------------------------

        if contains_variant_separator(phrase):
            reasons.append("multi_variant_phrase")

        # ----------------------------------------------------
        # Short phrase
        # ----------------------------------------------------

        if is_short_phrase(phrase):
            reasons.append("very_short_phrase")

        # ----------------------------------------------------
        # Short meaning
        # ----------------------------------------------------

        if is_short_meaning(cultural):
            reasons.append("short_cultural_meaning")

        # ----------------------------------------------------
        # Repeated / weak wording
        # ----------------------------------------------------

        if cultural.lower() == literal.lower() and cultural:
            reasons.append("literal_and_cultural_meaning_same")

        # ----------------------------------------------------
        # Tone anomaly
        # ----------------------------------------------------

        primary, secondary = split_tone(tone)

        if not tone:
            reasons.append("missing_tone")

        # ----------------------------------------------------
        # Add to review queue
        # ----------------------------------------------------

        if reasons:

            review_queue.append({
                "id": record_id,
                "bengali_phrase": phrase,
                "literal_meaning": literal,
                "cultural_meaning": cultural,
                "intended_emotion_tone": tone,
                "primary_tone_candidate": primary,
                "secondary_tone_candidate": secondary,
                "review_reasons": reasons,
            })

    return review_queue


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("BENGALI CULTURAL RAG - DATASET AUDIT")
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_dataset()

    print(f"\nRaw dataset: {RAW_FILE}")
    print(f"Total records: {len(data)}")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    validation = validate_records(data)

    # --------------------------------------------------------
    # Duplicate analysis
    # --------------------------------------------------------

    duplicate_phrases = find_duplicate_phrases(data)

    # --------------------------------------------------------
    # Variant analysis
    # --------------------------------------------------------

    variants = find_variant_records(data)

    # --------------------------------------------------------
    # Tone analysis
    # --------------------------------------------------------

    tone_analysis = analyze_tones(data)

    # --------------------------------------------------------
    # Review queue
    # --------------------------------------------------------

    review_queue = build_review_queue(
        data,
        duplicate_phrases
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    short_phrases = [
        {
            "id": clean_text(x.get("id")),
            "bengali_phrase": clean_text(
                x.get("bengali_phrase")
            ),
        }
        for x in data
        if is_short_phrase(
            x.get("bengali_phrase")
        )
    ]

    short_meanings = [
        {
            "id": clean_text(x.get("id")),
            "bengali_phrase": clean_text(
                x.get("bengali_phrase")
            ),
            "cultural_meaning": clean_text(
                x.get("cultural_meaning")
            ),
        }
        for x in data
        if is_short_meaning(
            x.get("cultural_meaning")
        )
    ]

    audit_report = {

        "dataset": {
            "source_file": str(RAW_FILE),
            "total_records": len(data),
        },

        "validation": {
            "required_fields": REQUIRED_FIELDS,
            "missing_field_records": len(
                validation["missing_fields"]
            ),
            "duplicate_id_count": len(
                validation["duplicate_ids"]
            ),
        },

        "duplicates": {
            "duplicate_phrase_group_count": len(
                duplicate_phrases
            ),
        },

        "variants": {
            "multi_variant_record_count": len(
                variants
            ),
        },

        "quality_flags": {
            "very_short_phrase_count": len(
                short_phrases
            ),
            "short_cultural_meaning_count": len(
                short_meanings
            ),
            "review_queue_count": len(
                review_queue
            ),
        },

        "tone": {
            "unique_raw_tone_count": len(
                tone_analysis["raw_tone_distribution"]
            ),
            "unique_primary_tone_count": len(
                tone_analysis["primary_tone_distribution"]
            ),
            "raw_tone_distribution":
                tone_analysis["raw_tone_distribution"],
            "primary_tone_distribution":
                tone_analysis["primary_tone_distribution"],
        },

        "details": {
            "missing_fields":
                validation["missing_fields"],

            "duplicate_ids":
                validation["duplicate_ids"],

            "short_phrases":
                short_phrases,

            "short_meanings":
                short_meanings,

            "variants":
                variants,
        },
    }

    # --------------------------------------------------------
    # Save audit report
    # --------------------------------------------------------

    with open(
        AUDIT_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            audit_report,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Save duplicate report
    # --------------------------------------------------------

    with open(
        DUPLICATE_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            duplicate_phrases,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Save review queue
    # --------------------------------------------------------

    with open(
        REVIEW_QUEUE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            review_queue,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(
        f"Total records              : {len(data)}"
    )

    print(
        f"Duplicate IDs              : "
        f"{len(validation['duplicate_ids'])}"
    )

    print(
        f"Duplicate phrase groups   : "
        f"{len(duplicate_phrases)}"
    )

    print(
        f"Multi-variant records     : "
        f"{len(variants)}"
    )

    print(
        f"Very short phrases        : "
        f"{len(short_phrases)}"
    )

    print(
        f"Short cultural meanings   : "
        f"{len(short_meanings)}"
    )

    print(
        f"Unique raw tones          : "
        f"{len(tone_analysis['raw_tone_distribution'])}"
    )

    print(
        f"Unique primary tones      : "
        f"{len(tone_analysis['primary_tone_distribution'])}"
    )

    print(
        f"Review queue records      : "
        f"{len(review_queue)}"
    )

    print("\nGenerated files:")

    print(
        f"1. {AUDIT_REPORT}"
    )

    print(
        f"2. {DUPLICATE_REPORT}"
    )

    print(
        f"3. {REVIEW_QUEUE}"
    )

    print("\nRaw dataset was NOT modified.")

    print("=" * 70)


if __name__ == "__main__":
    main()