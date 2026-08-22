import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CURATED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cultural_expressions.json"
)

FINAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_cultural_dataset.json"
)

FINAL_REVIEW = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_review_log.json"
)


def load_data():

    with open(
        CURATED_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def split_tone(tone):

    if not tone:
        return "", ""

    parts = [
        x.strip()
        for x in tone.split("/")
        if x.strip()
    ]

    if len(parts) == 1:
        return parts[0], ""

    return parts[0], " / ".join(parts[1:])


def find_by_ids(data, ids):

    return [
        item
        for item in data
        if item["id"] in ids
    ]


def main():

    data = load_data()

    # ========================================================
    # 1. Resolve "ইনিয়ে বিনিয়ে"
    # ========================================================

    inie_items = find_by_ids(
        data,
        {"169", "188"}
    )

    if inie_items:
        canonical = inie_items[0]
        canonical["cultural_meaning"] = (
            "ঘুরিয়ে ফিরিয়ে / নানা প্রকারে / "
            "অনুনয়-বিনয় করে"
        )

        canonical["expression_type"] = (
            "Cultural Expression"
        )

        canonical["review_status"] = (
            "verified_enriched"
        )
        
        canonical["source_record_ids"] = [
            "169",
            "188"
    ]

        canonical["verified_meaning_source"] = (
            "Bangladesh Accessible Dictionary"
        )


        canonical["source_note"] = (
            "Raw records contained conflicting meanings; "
            "the canonical meaning was verified and the "
            "conflicting duplicate records were merged."
        )
        
        data = [
            item
            for item in data
            if item["id"] != "188"
        ]

    # ========================================================
    # 2. Merge "ইঁচড়ে পাকা"
    # ========================================================

    ichre_items = find_by_ids(
        data,
        {"165", "184"}
    )

    if ichre_items:

        canonical = ichre_items[0]

        canonical["cultural_meaning"] = (
            "অকালপক্ব; বয়সের তুলনায় অতিরিক্ত "
            "পরিণত আচরণকারী"
        )

        canonical["expression_type"] = (
            "Idiom"
        )

        canonical["review_status"] = (
            "verified_merged"
        )

        canonical["source_record_ids"] = [
            "165",
            "184"
        ]

        canonical["verified_meaning_source"] = (
            "Bangladesh Accessible Dictionary"
        )

        canonical["source_note"] = (
            "Duplicate semantic entries merged."
        )

        # Remove second duplicate
        data = [
            item
            for item in data
            if item["id"] != "184"
        ]

    # ========================================================
    # 3. Merge "ইয়ারবকসি"
    # ========================================================

    iyar_items = find_by_ids(
        data,
        {"170", "191"}
    )

    if iyar_items:

        canonical = iyar_items[0]

        canonical["cultural_meaning"] = (
            "বন্ধুবান্ধব / সঙ্গী-সাথীদের দল"
        )

        canonical["expression_type"] = (
            "Cultural Expression"
        )

        canonical["review_status"] = (
            "verified_merged"
        )

        canonical["source_record_ids"] = [
            "170",
            "191"
        ]

        canonical["verified_meaning_source"] = (
            "Bangladesh Accessible Dictionary"
        )

        canonical["source_note"] = (
            "Duplicate semantic entries merged."
        )

        # Remove second duplicate
        data = [
            item
            for item in data
            if item["id"] != "191"
        ]

    # ========================================================
    # 4. Save final dataset
    # ========================================================

    with open(
        FINAL_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # 5. Save final review log
    # ========================================================

    review_log = {

        "resolved_conflict": {
            "phrase": "ইনিয়ে বিনিয়ে",
            "source_ids": ["169", "188"],
            "decision": "verified_enriched",
            "canonical_meaning":
                "ঘুরিয়ে ফিরিয়ে / নানা প্রকারে / অনুনয়-বিনয় করে"
        },

        "merged_records": [
            {
                "phrase": "ইঁচড়ে পাকা",
                "source_ids": ["165", "184"],
                "decision": "verified_merged"
            },
            {
                "phrase": "ইয়ারবকসি",
                "source_ids": ["170", "191"],
                "decision": "verified_merged"
            }
        ],

        "sources": [
            "Bangladesh Accessible Dictionary",
            "Bangladesh Accessible Dictionary Bengali-to-English"
        ]
    }

    with open(
        FINAL_REVIEW,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            review_log,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # Summary
    # ========================================================

    print("=" * 70)
    print("FINAL DATASET FINALIZATION")
    print("=" * 70)

    print(
        f"Previous curated records : 405"
    )

    print(
        f"Final records             : {len(data)}"
    )

    print(
        "\nResolved:"
    )

    print(
        "  ✓ ইনিয়ে বিনিয়ে"
    )

    print(
        "  ✓ ইঁচড়ে পাকা"
    )

    print(
        "  ✓ ইয়ারবকসি"
    )

    print(
        "\nGenerated:"
    )

    print(
        FINAL_FILE
    )

    print(
        FINAL_REVIEW
    )

    print(
        "\nRaw dataset remains untouched."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()