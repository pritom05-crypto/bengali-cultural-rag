import json
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BENCHMARK_PATH = (
    BASE_DIR
    / "evaluation"
    / "benchmark_queries.json"
)


# ============================================================
# LOAD
# ============================================================

with open(
    BENCHMARK_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print("=" * 70)
print("BENGALI CULTURAL RAG — BENCHMARK ANNOTATION")
print("=" * 70)

print()
print(
    f"Total records: {len(data)}"
)

print()
print(
    "Commands:"
)

print(
    "  ENTER      = skip"
)

print(
    "  /quit      = save and exit"
)

print(
    "  /skip      = skip current"
)

print()


# ============================================================
# ANNOTATION
# ============================================================

for i, item in enumerate(
    data,
    start=1
):

    # --------------------------------------------------------
    # Skip already verified
    # --------------------------------------------------------

    if item.get("verified") is True:

        continue


    phrase = str(
        item.get(
            "gold_phrase",
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


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        f"[{i}/{len(data)}]"
    )

    print(
        f"Gold Expression: {phrase}"
    )

    print()

    print(
        f"Cultural Meaning: {meaning}"
    )

    if literal:

        print(
            f"Literal Meaning: {literal}"
        )

    print()


    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    query = input(
        "Natural Bengali query > "
    ).strip()


    # --------------------------------------------------------
    # COMMANDS
    # --------------------------------------------------------

    if query == "/quit":

        break


    if query == "/skip":

        continue


    if not query:

        continue


    # --------------------------------------------------------
    # DIFFICULTY
    # --------------------------------------------------------

    print()

    print(
        "Difficulty:"
    )

    print(
        "1 = easy"
    )

    print(
        "2 = medium"
    )

    print(
        "3 = hard"
    )

    difficulty_input = input(
        "Select [1/2/3] > "
    ).strip()


    difficulty_map = {

        "1": "easy",

        "2": "medium",

        "3": "hard"

    }

    difficulty = difficulty_map.get(
        difficulty_input,
        "medium"
    )


    # --------------------------------------------------------
    # SAVE RECORD
    # --------------------------------------------------------

    item["query"] = query

    item["difficulty"] = difficulty

    item["verified"] = True

    item["notes"] = ""


    # --------------------------------------------------------
    # SAVE IMMEDIATELY
    # --------------------------------------------------------

    with open(
        BENCHMARK_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


    print()
    print(
        "✓ Saved"
    )


# ============================================================
# FINAL SAVE
# ============================================================

with open(
    BENCHMARK_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

verified = sum(
    1
    for x in data
    if x.get("verified") is True
)

print()
print("=" * 70)
print("ANNOTATION STATUS")
print("=" * 70)

print(
    f"Verified: {verified}/{len(data)}"
)

print(
    f"Remaining: {len(data) - verified}"
)

print()

if verified == len(data):

    print(
        "✓ Benchmark is ready for evaluation."
    )

else:

    print(
        "⚠ More annotation is required."
    )