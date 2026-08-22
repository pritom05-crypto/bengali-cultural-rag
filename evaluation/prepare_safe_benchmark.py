import json
import re
import random
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_cultural_dataset.json"
)

OUTPUT = (
    BASE_DIR
    / "evaluation"
    / "benchmark_queries.json"
)

REPORT = (
    BASE_DIR
    / "evaluation"
    / "safe_benchmark_report.json"
)

TARGET = 80
SEED = 623

random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def clean(value):
    if value is None:
        return ""

    return str(value).strip()


def normalize(text):
    text = clean(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def is_verified(record):

    status = normalize(
        record.get("review_status", "")
    )

    return status == "verified"


def has_meaning(record):

    cultural = clean(
        record.get("cultural_meaning")
    )

    literal = clean(
        record.get("literal_meaning")
    )

    return bool(cultural or literal)


# ============================================================
# MEANING -> NATURAL BENGALI QUERY RULES
#
# IMPORTANT:
# Each query is generated ONLY when the record's
# meaning contains a matching semantic concept.
#
# Gold phrase = SAME RECORD'S bengali_phrase
# ============================================================

RULES = [

    # --------------------------------------------------------
    # LAZY
    # --------------------------------------------------------

    (
        [
            "lazy",
            "terribly lazy",
            "very lazy",
            "idle person",
            "extremely lazy",
            "very idle",
            "inactive person"
        ],
        "সে খুব অলস এবং কোনো কাজ করতে চায় না।"
    ),

    # --------------------------------------------------------
    # ANGRY
    # --------------------------------------------------------

    (
        [
            "angry",
            "extremely angry",
            "extremely angry/furious",
            "furious",
            "rage",
            "very angry"
        ],
        "সে খুব রেগে গেছে।"
    ),

    # --------------------------------------------------------
    # FEAR
    # --------------------------------------------------------

    (
        [
            "fear",
            "afraid",
            "frightened",
            "scared",
            "terrified",
            "fearful",
            "panic"
        ],
        "ঘটনাটি দেখে সে খুব ভয় পেয়ে গেছে।"
    ),

    # --------------------------------------------------------
    # DANGER / TROUBLE
    # --------------------------------------------------------

    (
        [
            "danger",
            "great danger",
            "terrible danger",
            "trouble",
            "extreme trouble",
            "serious trouble",
            "crisis",
            "adversity",
            "distress"
        ],
        "সে খুব বিপদে পড়েছে এবং পরিস্থিতি থেকে বের হতে পারছে না।"
    ),

    # --------------------------------------------------------
    # CONFUSED
    # --------------------------------------------------------

    (
        [
            "confused",
            "confusion",
            "bewildered",
            "unable to decide",
            "cannot decide",
            "uncertain"
        ],
        "সে কী করবে বুঝতে পারছে না।"
    ),

    # --------------------------------------------------------
    # SURPRISED
    # --------------------------------------------------------

    (
        [
            "surprised",
            "unexpected",
            "astonished",
            "amazed",
            "shocked",
            "surprising",
            "unexpected event"
        ],
        "ঘটনাটি দেখে সে খুব অবাক হয়ে গেছে।"
    ),

    # --------------------------------------------------------
    # SADNESS / PAIN
    # --------------------------------------------------------

    (
        [
            "sad",
            "sadness",
            "deep sadness",
            "sorrow",
            "grief",
            "pain",
            "suffering",
            "distress"
        ],
        "ঘটনাটি তাকে খুব কষ্ট দিয়েছে।"
    ),

    # --------------------------------------------------------
    # DEPENDENT
    # --------------------------------------------------------

    (
        [
            "dependent",
            "dependence",
            "reliance",
            "relying on others",
            "dependent on others"
        ],
        "সে নিজের কাজ না করে সবসময় অন্যের ওপর নির্ভর করে।"
    ),

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    (
        [
            "successful",
            "success",
            "achievement",
            "achieve success",
            "victory",
            "triumph"
        ],
        "শেষ পর্যন্ত সে সফল হয়েছে।"
    ),

    # --------------------------------------------------------
    # FAILURE
    # --------------------------------------------------------

    (
        [
            "failure",
            "failed",
            "unsuccessful",
        ],
        "তার চেষ্টা শেষ পর্যন্ত ব্যর্থ হয়েছে।"
    ),

    # --------------------------------------------------------
    # PROCRASTINATION
    # --------------------------------------------------------

    (
        [
            "procrastination",
            "procrastinate",
            "delay",
            "delaying",
            "putting off",
            "postponing"
        ],
        "সে কাজটি বারবার পিছিয়ে দিচ্ছে।"
    ),

    # --------------------------------------------------------
    # CHEATING / DECEPTION
    # --------------------------------------------------------

    (
        [
            "cheat",
            "cheating",
            "deceive",
            "deception",
            "fraud",
            "trick others",
            "trick"
        ],
        "লোকটি কৌশলে অন্যদের ঠকিয়েছে।"
    ),

    # --------------------------------------------------------
    # SELF PRAISE
    # --------------------------------------------------------

    (
        [
            "praise oneself",
            "boast",
            "boasting",
            "self praise",
            "bragging"
        ],
        "সে সবসময় নিজের প্রশংসা করে।"
    ),

    # --------------------------------------------------------
    # CRITICISM
    # --------------------------------------------------------

    (
        [
            "criticism",
            "criticize",
            "critical",
            "disapproval",
            "condemnation"
        ],
        "তার কাজের জন্য মানুষ তাকে সমালোচনা করছে।"
    ),

    # --------------------------------------------------------
    # STUBBORN
    # --------------------------------------------------------

    (
        [
            "stubborn",
            "obstinate",
            "unyielding",
            "inflexible",
            "refusing to change"
        ],
        "অনেক বোঝালেও সে নিজের অবস্থান থেকে সরছে না।"
    ),

    # --------------------------------------------------------
    # HARD WORK
    # --------------------------------------------------------

    (
        [
            "hard work",
            "working hard",
            "diligent",
            "diligence",
            "effort",
            "persistent effort"
        ],
        "সে সফল হওয়ার জন্য কঠোর পরিশ্রম করছে।"
    ),

    # --------------------------------------------------------
    # CARE / CONCERN
    # --------------------------------------------------------

    (
        [
            "care",
            "concern",
            "take care",
            "caring"
        ],
        "সে বিষয়টি নিয়ে খুব চিন্তিত ও যত্নশীল।"
    ),

    # --------------------------------------------------------
    # MONEY / POVERTY
    # --------------------------------------------------------

    (
        [
            "poor",
            "poverty",
            "moneyless",
            "destitute",
            "beggar",
            "financial hardship"
        ],
        "লোকটি খুব দরিদ্র এবং অর্থকষ্টে আছে।"
    ),

    # --------------------------------------------------------
    # DEATH
    # --------------------------------------------------------

    (
        [
            "to die",
            "death",
            "dead",
            "die"
        ],
        "লোকটি মারা গেছে।"
    ),

    # --------------------------------------------------------
    # TALK
    # --------------------------------------------------------

    (
        [
            "talk",
            "conversation",
            "idle talk",
            "useless talk",
            "gossip"
        ],
        "সে সারাদিন অপ্রয়োজনীয় কথাবার্তা বলে।"
    ),

    # --------------------------------------------------------
    # HELP
    # --------------------------------------------------------

    (
        [
            "help",
            "assistance",
            "support"
        ],
        "প্রয়োজনে সে অন্যের সাহায্য নেয়।"
    ),

    # --------------------------------------------------------
    # DIFFICULTY
    # --------------------------------------------------------

    (
        [
            "difficult",
            "difficulty",
            "hard situation",
            "hardship",
            "problem"
        ],
        "সে একটি কঠিন পরিস্থিতির মধ্যে পড়েছে।"
    ),

    # --------------------------------------------------------
    # BOASTFUL
    # --------------------------------------------------------

    (
        [
            "boastful",
            "arrogant",
            "proud",
            "pride"
        ],
        "সে নিজের ক্ষমতা নিয়ে খুব অহংকার করে।"
    ),

    # --------------------------------------------------------
    # UNAUTHORIZED
    # --------------------------------------------------------

    (
        [
            "unauthorized",
            "unauthorized practice",
            "interference",
            "interfering"
        ],
        "সে অন্যের কাজে অনধিকার হস্তক্ষেপ করেছে।"
    ),

    # --------------------------------------------------------
    # ANXIOUS
    # --------------------------------------------------------

    (
        [
            "anxious",
            "anxiety",
            "distressed",
            "restless",
            "worried"
        ],
        "পরিস্থিতির কারণে সে খুব উদ্বিগ্ন ও অস্থির হয়ে পড়েছে।"
    ),
]


# ============================================================
# MATCHING
# ============================================================

def find_matching_query(record):

    cultural = normalize(
        record.get("cultural_meaning", "")
    )

    literal = normalize(
        record.get("literal_meaning", "")
    )

    combined = cultural + " " + literal

    matches = []

    for keywords, query in RULES:

        score = 0

        for keyword in keywords:

            keyword = normalize(keyword)

            if keyword in combined:
                score += 1

        if score > 0:

            matches.append(
                (
                    score,
                    query
                )
            )

    if not matches:
        return None

    matches.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return matches[0][1]


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 75)
print("SAFE BENCHMARK PREPARATION V2")
print("DATASET-GROUNDED / NO API")
print("=" * 75)

with open(
    DATASET,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)

print(
    f"Dataset records: {len(dataset)}"
)


# ============================================================
# BUILD CANDIDATES
# ============================================================

candidates = []

reason_counter = Counter()

for record in dataset:

    phrase = clean(
        record.get("bengali_phrase")
    )

    if not phrase:

        reason_counter[
            "missing_phrase"
        ] += 1

        continue

    if not is_verified(record):

        reason_counter[
            "not_verified"
        ] += 1

        continue

    if not has_meaning(record):

        reason_counter[
            "missing_meaning"
        ] += 1

        continue

    query = find_matching_query(record)

    if not query:

        reason_counter[
            "no_safe_semantic_rule"
        ] += 1

        continue

    candidates.append({

        "id":
            record.get("id"),

        "query":
            query,

        "gold_phrase":
            phrase,

        "cultural_meaning":
            clean(
                record.get(
                    "cultural_meaning"
                )
            ),

        "literal_meaning":
            clean(
                record.get(
                    "literal_meaning"
                )
            ),

        "primary_tone":
            clean(
                record.get(
                    "primary_tone"
                )
            ),

        "secondary_tone":
            clean(
                record.get(
                    "secondary_tone"
                )
            ),

        "verified":
            True,

        "annotation_source":
            "dataset_meaning_rule_based_v2"

    })


# ============================================================
# REMOVE DUPLICATE QUERY + GOLD
# ============================================================

unique = []

seen = set()

for item in candidates:

    key = (
        normalize(item["query"]),
        normalize(item["gold_phrase"])
    )

    if key in seen:
        continue

    seen.add(key)

    unique.append(item)


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(unique)

final = unique[:TARGET]


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final,
        f,
        ensure_ascii=False,
        indent=2
    )


report = {

    "dataset_records":
        len(dataset),

    "candidate_records":
        len(candidates),

    "unique_records":
        len(unique),

    "final_benchmark_records":
        len(final),

    "verified_records":
        len([
            x for x in final
            if x["verified"]
        ]),

    "api_used":
        False,

    "gold_source":
        "same_dataset_record",

    "method":
        "conservative meaning-keyword mapping",

    "excluded_reasons":
        dict(reason_counter)

}


with open(
    REPORT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("BENCHMARK CREATED")
print("=" * 75)

print(
    f"Dataset records       : {len(dataset)}"
)

print(
    f"Candidate records     : {len(candidates)}"
)

print(
    f"Unique records        : {len(unique)}"
)

print(
    f"Final benchmark       : {len(final)}"
)

print(
    f"Verified              : "
    f"{sum(x['verified'] for x in final)}"
)


print("\nEXCLUSION SUMMARY")

for reason, count in reason_counter.most_common():

    print(
        f"{reason:25s}: {count}"
    )


print("\nSAMPLES")

for i, item in enumerate(
    final[:20],
    1
):

    print(
        f"\n[{i}]"
    )

    print(
        "Query :",
        item["query"]
    )

    print(
        "Gold  :",
        item["gold_phrase"]
    )

    print(
        "Meaning:",
        item["cultural_meaning"]
    )


print("\nSaved:")
print(OUTPUT)
print(REPORT)

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)