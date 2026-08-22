import json
import re
import random
from pathlib import Path

# ============================================================
# BENGALI CULTURAL RAG
# GOLD-SAFE BENCHMARK GENERATOR V3
#
# IMPORTANT:
# - NO API
# - NO GEMINI
# - NO GROQ
# - NO E5 AUTO GOLD SELECTION
# - Gold phrase MUST come from the actual dataset
# - Ambiguous records are excluded
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

REPORT_PATH = (
    BASE_DIR
    / "evaluation"
    / "gold_safe_benchmark_report.json"
)

random.seed(623)


# ============================================================
# HELPERS
# ============================================================

def clean(x):
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def norm(x):
    return clean(x).lower()


def contains_any(text, words):
    text = norm(text)
    return any(norm(w) in text for w in words)


# ============================================================
# SAFE GOLD GROUPS
#
# We only accept meanings that are VERY explicit.
# No semantic similarity is used here.
# ============================================================

SAFE_GROUPS = {

    # --------------------------------------------------------
    # LAZY
    # --------------------------------------------------------

    "lazy": {
        "query": "সে খুব অলস এবং কোনো কাজ করতে চায় না।",

        "meaning": [
            "lazy",
            "terribly lazy",
            "very lazy",
            "extremely lazy",
            "very idle",
            "idle person",
            "very inactive",
            "অলস",
            "অকর্মণ্য"
        ],

        "exclude": [
            "inactive / minor",
            "minor",
            "impersonal",
            "useless or idle talk",
            "wasting time"
        ]
    },

    # --------------------------------------------------------
    # ANGRY
    # --------------------------------------------------------

    "angry": {
        "query": "সে খুব রেগে গেছে এবং প্রচণ্ড রাগান্বিত।",

        "meaning": [
            "extremely angry",
            "extremely angry/furious",
            "furious",
            "very angry",
            "angry",
            "rage",
            "রাগান্বিত",
            "প্রচণ্ড রাগ"
        ],

        "exclude": [
            "anxious",
            "distressed",
            "hostile",
            "resentful"
        ]
    },

    # --------------------------------------------------------
    # DANGER
    # --------------------------------------------------------

    "danger": {
        "query": "সে এমন বিপদে পড়েছে যেখান থেকে বের হওয়ার পথ পাচ্ছে না।",

        "meaning": [
            "great danger",
            "terrible danger",
            "falling into sudden danger or trouble",
            "extreme trouble",
            "serious danger",
            "serious trouble",
            "crisis",
            "danger",
            "বিপদ"
        ],

        "exclude": [
            "angry",
            "confused",
            "surprised",
            "poor"
        ]
    },

    # --------------------------------------------------------
    # FEAR
    # --------------------------------------------------------

    "fear": {
        "query": "ঘটনাটি দেখে সে খুব ভয় পেয়ে গেছে।",

        "meaning": [
            "fear",
            "afraid",
            "frightened",
            "scared",
            "terrified",
            "fearful",
            "panic",
            "ভয়",
            "আতঙ্ক"
        ],

        "exclude": [
            "danger",
            "trouble",
            "crisis",
            "distress",
            "anxious"
        ]
    },

    # --------------------------------------------------------
    # CONFUSED
    # --------------------------------------------------------

    "confused": {
        "query": "সে কী করবে বুঝতে পারছে না এবং খুব বিভ্রান্ত হয়ে গেছে।",

        "meaning": [
            "confused",
            "confusion",
            "bewildered",
            "unable to decide",
            "cannot decide",
            "uncertain",
            "বিভ্রান্ত"
        ],

        "exclude": [
            "bad advice",
            "inactive",
            "minor",
            "angry"
        ]
    },

    # --------------------------------------------------------
    # SURPRISED
    # --------------------------------------------------------

    "surprised": {
        "query": "ঘটনাটি দেখে সে খুব অবাক ও বিস্মিত হয়ে গেছে।",

        "meaning": [
            "surprised",
            "astonished",
            "amazed",
            "shocked",
            "unexpected",
            "surprising",
            "বিস্মিত",
            "অবাক"
        ],

        "exclude": [
            "big event",
            "danger",
            "crisis"
        ]
    },

    # --------------------------------------------------------
    # SAD / PAIN
    # --------------------------------------------------------

    "pain": {
        "query": "ঘটনাটি তাকে খুব কষ্ট দিয়েছে।",

        "meaning": [
            "to be in pain",
            "painful",
            "deep pain",
            "suffering",
            "hurt someone's feelings deeply",
            "emotional pain",
            "কষ্ট",
            "যন্ত্রণা"
        ],

        "exclude": [
            "pride",
            "success",
            "danger"
        ]
    },

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    "success": {
        "query": "শেষ পর্যন্ত সে সফল হয়েছে এবং লক্ষ্য অর্জন করেছে।",

        "meaning": [
            "to win or achieve success",
            "achieve success",
            "success",
            "successful",
            "victory",
            "triumph",
            "সাফল্য"
        ],

        "exclude": [
            "failure",
            "reluctantly",
            "poor"
        ]
    },

    # --------------------------------------------------------
    # FAILURE
    # --------------------------------------------------------

    "failure": {
        "query": "অনেক চেষ্টা করেও শেষ পর্যন্ত তার কাজটি ব্যর্থ হয়েছে।",

        "meaning": [
            "failure",
            "failed",
            "unsuccessful",
            "failure of effort",
            "ব্যর্থ"
        ],

        "exclude": [
            "reluctantly",
            "success",
            "victory"
        ]
    },

    # --------------------------------------------------------
    # PROCRASTINATION
    # --------------------------------------------------------

    "procrastination": {
        "query": "সে কাজটি বারবার পিছিয়ে দিচ্ছে এবং সময় নষ্ট করছে।",

        "meaning": [
            "procrastination",
            "procrastinate",
            "delaying",
            "delay",
            "postponing",
            "putting off",
            "wasting time without work",
            "গড়িমসি",
            "সময় নষ্ট"
        ],

        "exclude": [
            "idle talk",
            "lazy",
            "failure"
        ]
    },

    # --------------------------------------------------------
    # DECEPTION
    # --------------------------------------------------------

    "deception": {
        "query": "লোকটি কৌশলে অন্যদের ঠকিয়েছে।",

        "meaning": [
            "trickery or deception",
            "deception",
            "deceive",
            "trickery",
            "cheating",
            "fraud",
            "প্রতারণা",
            "ঠকানো"
        ],

        "exclude": [
            "protest",
            "criticism",
            "donate"
        ]
    },

    # --------------------------------------------------------
    # CRITICISM
    # --------------------------------------------------------

    "criticism": {
        "query": "তার কাজ ও আচরণ নিয়ে সবাই সমালোচনা করছে।",

        "meaning": [
            "criticism",
            "criticize",
            "critical",
            "disapproval",
            "condemnation",
            "সমালোচনা"
        ],

        "exclude": [
            "wrong or whimsical action",
            "protest"
        ]
    },

    # --------------------------------------------------------
    # STUBBORN
    # --------------------------------------------------------

    "stubborn": {
        "query": "অনেক বোঝালেও সে নিজের অবস্থান থেকে সরছে না।",

        "meaning": [
            "stubborn",
            "obstinate",
            "unyielding",
            "inflexible",
            "refusing to change",
            "একগুঁয়ে",
            "গোঁয়ার"
        ],

        "exclude": [
            "impersonal",
            "dependent"
        ]
    },

    # --------------------------------------------------------
    # HARD WORK
    # --------------------------------------------------------

    "hardwork": {
        "query": "সে সফল হওয়ার জন্য খুব কঠোর পরিশ্রম করছে।",

        "meaning": [
            "working hard",
            "hard work",
            "diligent",
            "diligence",
            "putting in utmost effort",
            "utmost effort",
            "কঠোর পরিশ্রম"
        ],

        "exclude": [
            "strict promise",
            "pride",
            "failure"
        ]
    },

    # --------------------------------------------------------
    # DEPENDENCY
    # --------------------------------------------------------

    "dependency": {
        "query": "সে নিজের কাজ নিজে না করে সবসময় অন্যের ওপর নির্ভর করে।",

        "meaning": [
            "dependent",
            "dependence",
            "reliance",
            "relying on others",
            "dependent on others",
            "নির্ভরশীল"
        ],

        "exclude": [
            "no other way",
            "impersonal"
        ]
    },

    # --------------------------------------------------------
    # POVERTY
    # --------------------------------------------------------

    "poverty": {
        "query": "লোকটি খুব দরিদ্র এবং অর্থকষ্টে জীবনযাপন করছে।",

        "meaning": [
            "poor / beggar",
            "poor person",
            "moneyless",
            "destitute",
            "poverty",
            "poor",
            "দরিদ্র",
            "অর্থকষ্ট"
        ],

        "exclude": [
            "poor but luxurious"
        ]
    },

    # --------------------------------------------------------
    # BOASTING
    # --------------------------------------------------------

    "boasting": {
        "query": "সে নিজের ক্ষমতা ও কৃতিত্ব নিয়ে অহংকার করে।",

        "meaning": [
            "boastful",
            "boasting",
            "arrogant",
            "proud",
            "pride",
            "fall of pride",
            "অহংকার"
        ],

        "exclude": [
            "donate",
            "generous",
            "benevolent"
        ]
    },

    # --------------------------------------------------------
    # IDLE TALK
    # --------------------------------------------------------

    "idle_talk": {
        "query": "সে সারাদিন অপ্রয়োজনীয় ও অর্থহীন কথাবার্তা বলে।",

        "meaning": [
            "useless or idle talk",
            "idle talk",
            "useless talk",
            "meaningless talk",
            "gossip",
            "অর্থহীন কথাবার্তা"
        ],

        "exclude": [
            "useless",
            "poor",
            "lazy"
        ]
    },

    # --------------------------------------------------------
    # ANXIETY
    # --------------------------------------------------------

    "anxiety": {
        "query": "পরিস্থিতির কারণে সে খুব উদ্বিগ্ন ও অস্থির হয়ে পড়েছে।",

        "meaning": [
            "anxiety",
            "anxious",
            "restlessness",
            "restless",
            "worried",
            "worry",
            "উদ্বেগ",
            "অস্থিরতা"
        ],

        "exclude": [
            "confused",
            "danger",
            "angry"
        ]
    },
}


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 75)
print("BENGALI CULTURAL RAG — GOLD SAFE BENCHMARK V4")
print("STRICT MEANING VALIDATION / NO API")
print("=" * 75)

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
# VERIFIED DATA ONLY
# ============================================================

verified = []

for record in dataset:

    phrase = clean(
        record.get("bengali_phrase")
    )

    meaning = clean(
        record.get("cultural_meaning")
    )

    literal = clean(
        record.get("literal_meaning")
    )

    status = norm(
        record.get("review_status")
    )

    if not phrase:
        continue

    if status != "verified":
        continue

    if not meaning:
        continue

    verified.append(record)


print(
    f"Verified records: {len(verified)}"
)


# ============================================================
# STRICT MATCH
# ============================================================

benchmark = []

used_phrases = set()

diagnostics = []


for group_name, config in SAFE_GROUPS.items():

    matches = []

    for record in verified:

        phrase = clean(
            record.get("bengali_phrase")
        )

        meaning = norm(
            record.get("cultural_meaning")
        )

        literal = norm(
            record.get("literal_meaning")
        )

        combined = (
            meaning + " " + literal
        )

        # -----------------------------------------------
        # Positive match
        # -----------------------------------------------

        positive_hits = 0

        for keyword in config["meaning"]:

            if norm(keyword) in combined:

                positive_hits += 1

        # -----------------------------------------------
        # Negative match
        # -----------------------------------------------

        negative_hits = 0

        for keyword in config["exclude"]:

            if norm(keyword) in combined:

                negative_hits += 1

        # -----------------------------------------------
        # STRICT DECISION
        # -----------------------------------------------

        if positive_hits == 0:
            continue

        if negative_hits > 0:
            continue

        # Require strong evidence
        if positive_hits < 1:
            continue

        matches.append(
            (
                positive_hits,
                record
            )
        )

    # -------------------------------------------------------
    # Sort strongest matches first
    # -------------------------------------------------------

    matches.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # -------------------------------------------------------
    # Keep ONLY strongest unique expression
    # -------------------------------------------------------

    selected = None

    for hit_count, record in matches:

        phrase = clean(
            record.get("bengali_phrase")
        )

        if phrase in used_phrases:
            continue

        selected = (
            hit_count,
            record
        )

        break

    # -------------------------------------------------------
    # No safe match
    # -------------------------------------------------------

    if selected is None:

        diagnostics.append({

            "group": group_name,

            "query": config["query"],

            "status": "NO_SAFE_GOLD",

            "candidate_count":
                len(matches)

        })

        continue

    hit_count, record = selected

    phrase = clean(
        record.get("bengali_phrase")
    )

    used_phrases.add(
        phrase
    )

    benchmark.append({

        "id":
            record.get("id"),

        "query":
            config["query"],

        "gold_phrase":
            phrase,

        "category":
            group_name,

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

        "verified":
            True,

        "annotation_source":
            "strict_dataset_meaning_match_v4",

        "positive_evidence_count":
            hit_count

    })

    diagnostics.append({

        "group":
            group_name,

        "query":
            config["query"],

        "status":
            "SELECTED",

        "gold_phrase":
            phrase,

        "meaning":
            clean(
                record.get(
                    "cultural_meaning"
                )
            ),

        "positive_evidence_count":
            hit_count,

        "candidate_count":
            len(matches)

    })


# ============================================================
# SAVE BENCHMARK
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        benchmark,
        f,
        ensure_ascii=False,
        indent=2
    )


report = {

    "dataset_records":
        len(dataset),

    "verified_records":
        len(verified),

    "intent_groups":
        len(SAFE_GROUPS),

    "benchmark_records":
        len(benchmark),

    "api_used":
        False,

    "gold_source":
        "actual_verified_dataset_record",

    "gold_selection":
        "strict_positive_and_negative_meaning_match",

    "diagnostics":
        diagnostics

}


with open(
    REPORT_PATH,
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
# PRINT
# ============================================================

print("\n" + "=" * 75)
print("BENCHMARK CREATED")
print("=" * 75)

print(
    f"Verified dataset : {len(verified)}"
)

print(
    f"Intent groups    : {len(SAFE_GROUPS)}"
)

print(
    f"Safe benchmark   : {len(benchmark)}"
)


print("\nSELECTED GOLD RECORDS")

for i, item in enumerate(
    benchmark,
    1
):

    print(
        f"\n[{i}] {item['category']}"
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


print("\nUNRESOLVED GROUPS")

for item in diagnostics:

    if item["status"] == "NO_SAFE_GOLD":

        print(
            f"- {item['group']}: "
            f"NO SAFE GOLD"
        )


print("\nSaved:")
print(OUTPUT_PATH)
print(REPORT_PATH)

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)