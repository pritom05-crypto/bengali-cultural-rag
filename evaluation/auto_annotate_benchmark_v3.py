import json
import os
import random
import re
from pathlib import Path

# ============================================================
# PATHS
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
    / "benchmark_annotation_report.json"
)

TARGET_RECORDS = 120
SEED = 623

random.seed(SEED)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean(value):
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(str(x) for x in value)

    return str(value).strip()


def norm(value):
    value = clean(value)
    value = value.lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


# ============================================================
# DATASET-SPECIFIC EXTRACTION
# ============================================================

def get_phrase(record):
    return clean(
        record.get("bengali_phrase", "")
    )


def get_meaning(record):
    return clean(
        record.get("cultural_meaning", "")
    )


def get_literal(record):
    return clean(
        record.get("literal_meaning", "")
    )


def get_tone(record):
    primary = clean(
        record.get("primary_tone", "")
    )

    secondary = clean(
        record.get("secondary_tone", "")
    )

    intended = clean(
        record.get("intended_emotion_tone", "")
    )

    parts = []

    if primary:
        parts.append(primary)

    if secondary:
        parts.append(secondary)

    if intended and intended not in parts:
        parts.append(intended)

    return " / ".join(parts)


# ============================================================
# BAD / SUSPICIOUS RECORD FILTER
# ============================================================

# These are records that were already identified during
# previous validation as suspicious or semantically inconsistent.

KNOWN_BAD = {
    "অর্ধচন্দ্র",
    "অষ্টরম্ভা",
    "অতি দর্পে হত লঙ্কা",
    "অঞ্চল প্রভাব",
    "অক্কা পাওয়া",
}


def valid_record(record):

    phrase = get_phrase(record)
    meaning = get_meaning(record)

    if not phrase:
        return False, "missing bengali_phrase"

    if not meaning:
        return False, "missing cultural_meaning"

    if phrase in KNOWN_BAD:
        return False, "known suspicious record"

    # Don't use very short garbage
    if len(norm(phrase)) < 2:
        return False, "phrase too short"

    if len(norm(meaning)) < 3:
        return False, "meaning too short"

    return True, ""


# ============================================================
# MEANING CATEGORY
# ============================================================

def detect_category(meaning, tone):

    text = norm(
        f"{meaning} {tone}"
    )

    groups = {

        "anger": [
            "angry",
            "furious",
            "anger",
            "rage",
            "রাগ",
            "রেগে",
            "ক্রোধ",
            "ক্রুদ্ধ",
            "রাগান্বিত",
        ],

        "fear": [
            "fear",
            "afraid",
            "frightened",
            "scared",
            "terrified",
            "panic",
            "ভয়",
            "ভয়",
            "ভীত",
            "আতঙ্ক",
            "শঙ্কা",
        ],

        "danger": [
            "danger",
            "trouble",
            "crisis",
            "adversity",
            "distress",
            "বিপদ",
            "সংকট",
            "ঝুঁকি",
            "দুর্দশা",
            "সমস্যা",
        ],

        "lazy": [
            "lazy",
            "idle",
            "inactive",
            "অলস",
            "কুঁড়ে",
            "কুঁড়ে",
            "কর্মবিমুখ",
            "কাজ করতে চায় না",
            "কাজ করতে চায় না",
        ],

        "sadness": [
            "sad",
            "sadness",
            "sorrow",
            "pain",
            "grief",
            "দুঃখ",
            "বেদনা",
            "কষ্ট",
            "যন্ত্রণা",
        ],

        "success": [
            "success",
            "successful",
            "lucky",
            "fortune",
            "সাফল্য",
            "সফল",
            "ভাগ্য",
            "সৌভাগ্য",
        ],

        "failure": [
            "failure",
            "failed",
            "defeat",
            "ব্যর্থ",
            "ব্যর্থতা",
            "পরাজয়",
            "পরাজয়",
        ],

        "deception": [
            "deceive",
            "deception",
            "cheat",
            "fake",
            "fraud",
            "প্রতারণা",
            "ঠকানো",
            "ফাঁকি",
            "ধোঁকা",
        ],

        "stubborn": [
            "stubborn",
            "unyielding",
            "জেদি",
            "জেদ",
            "অনড়",
            "অনড়",
            "একগুঁয়ে",
            "একগুঁয়ে",
        ],

        "boast": [
            "boast",
            "boasting",
            "self praise",
            "নিজের প্রশংসা",
            "বড়াই",
            "বড়াই",
            "আত্মপ্রশংসা",
        ],

        "confusion": [
            "confused",
            "confusion",
            "বিভ্রান্ত",
            "বিভ্রান্তি",
            "হতবুদ্ধি",
        ],

        "surprise": [
            "surprise",
            "surprised",
            "shocked",
            "unexpected",
            "বিস্মিত",
            "আশ্চর্য",
            "হতবাক",
            "অপ্রত্যাশিত",
        ],

        "poverty": [
            "poor",
            "poverty",
            "দারিদ্র্য",
            "দারিদ্র",
            "অভাব",
            "গরিব",
            "দুঃস্থ",
        ],

        "effort": [
            "effort",
            "hard work",
            "trying",
            "চেষ্টা",
            "পরিশ্রম",
            "প্রচেষ্টা",
        ],
    }

    for category, words in groups.items():

        for word in words:

            if word in text:
                return category

    return "general"


# ============================================================
# QUERY GENERATORS
# ============================================================

QUERY_TEMPLATES = {

    "anger": [
        "সে প্রচণ্ড রেগে গেছে এবং খুব উত্তেজিত হয়ে আছে।",
        "লোকটি ভীষণ রাগান্বিত হয়ে পড়েছে।",
        "তার রাগ এত বেশি যে তার সঙ্গে কথা বলা কঠিন।",
        "সে অত্যন্ত ক্রুদ্ধ হয়ে আছে।",
    ],

    "fear": [
        "ঘটনাটি দেখে সে খুব ভয় পেয়ে গেছে।",
        "হঠাৎ ঘটনায় সে একেবারে আতঙ্কিত হয়ে পড়েছে।",
        "ভয়ে সে দিশেহারা হয়ে গেছে।",
        "সে প্রচণ্ড ভয় পেয়ে অস্থির হয়ে পড়েছে।",
    ],

    "danger": [
        "আমি এমন এক বিপদের মধ্যে পড়েছি যেখান থেকে বের হওয়ার পথ পাচ্ছি না।",
        "সে হঠাৎ খুব বড় বিপদে পড়ে গেছে।",
        "পরিস্থিতি এত খারাপ যে সে সংকট থেকে বের হতে পারছে না।",
        "লোকটি ভয়াবহ বিপদের মধ্যে পড়েছে।",
    ],

    "lazy": [
        "সে সারাদিন অলস হয়ে বসে থাকে এবং কোনো কাজ করতে চায় না।",
        "লোকটি অত্যন্ত অলস ও কর্মবিমুখ।",
        "সে কোনো কাজ করতে আগ্রহী নয়, সবসময় অলসভাবে বসে থাকে।",
        "ছেলেটি কাজকর্মে একেবারেই অনীহা দেখায়।",
    ],

    "sadness": [
        "ঘটনাটি তাকে ভেতর থেকে খুব কষ্ট দিয়েছে।",
        "ঘটনার কারণে সে গভীরভাবে দুঃখ পেয়েছে।",
        "বিষয়টি তাকে মানসিকভাবে খুব কষ্ট দিয়েছে।",
    ],

    "success": [
        "হঠাৎ তার ভাগ্য ভালো হয়ে গেল এবং সে সাফল্য পেল।",
        "সঠিক সময়ে সুযোগ পাওয়ায় তার কাজটি সফল হলো।",
        "ভালো ভাগ্যের কারণে তার পরিকল্পনা সফল হয়েছে।",
    ],

    "failure": [
        "শেষ পর্যন্ত তার পরিকল্পনা সম্পূর্ণ ব্যর্থ হয়ে গেল।",
        "ব্যর্থতার কারণে সে খুব হতাশ হয়ে পড়েছে।",
        "তার প্রচেষ্টা শেষ পর্যন্ত ব্যর্থ হয়েছে।",
    ],

    "deception": [
        "লোকটি কৌশলে অন্যদের ঠকিয়ে নিজের দায়িত্ব এড়িয়ে গেল।",
        "সে সত্যি কথা না বলে সবাইকে ফাঁকি দেওয়ার চেষ্টা করছে।",
        "লোকটি অন্যদের সঙ্গে প্রতারণা করছে।",
    ],

    "stubborn": [
        "অনেক বোঝালেও লোকটি নিজের সিদ্ধান্ত থেকে সরছে না।",
        "সে অত্যন্ত জেদি এবং কারও কথা শুনতে চায় না।",
        "লোকটি নিজের অবস্থান থেকে একেবারেই নড়ছে না।",
    ],

    "boast": [
        "লোকটি সবসময় নিজের প্রশংসা করতে ভালোবাসে।",
        "সে নিজের যোগ্যতার কথা বলে বেড়ায়।",
        "লোকটি নিজের সাফল্য নিয়ে সবসময় বড়াই করে।",
    ],

    "confusion": [
        "ঘটনাটি বুঝতে না পেরে সে সম্পূর্ণ বিভ্রান্ত হয়ে গেছে।",
        "সে কী করবে বুঝতে পারছে না।",
        "পরিস্থিতিতে পড়ে সে একেবারে হতবুদ্ধি হয়ে গেছে।",
    ],

    "surprise": [
        "ঘটনাটি এতটাই অপ্রত্যাশিত ছিল যে সবাই হতবাক হয়ে গেল।",
        "হঠাৎ ঘটনাটি দেখে সবাই খুব অবাক হয়ে গেল।",
        "ঘটনাটি দেখে সে বিস্মিত হয়ে গেল।",
    ],

    "poverty": [
        "অভাবের কারণে পরিবারটি খুব কষ্টে জীবনযাপন করছে।",
        "লোকটি প্রচণ্ড অর্থকষ্টের মধ্যে দিন কাটাচ্ছে।",
        "পরিবারটি দীর্ঘদিন ধরে দারিদ্র্যের মধ্যে রয়েছে।",
    ],

    "effort": [
        "কাজটি শেষ করতে সে অত্যন্ত দৃঢ়ভাবে চেষ্টা করছে।",
        "সফল হওয়ার জন্য সে সর্বশক্তি দিয়ে চেষ্টা করছে।",
        "সে লক্ষ্য অর্জনের জন্য কঠোর পরিশ্রম করছে।",
    ],
}


# ============================================================
# SAFE FALLBACK
# ============================================================

def make_query(record, index):

    meaning = get_meaning(record)
    tone = get_tone(record)

    category = detect_category(
        meaning,
        tone
    )

    if category in QUERY_TEMPLATES:

        templates = QUERY_TEMPLATES[category]

        return (
            templates[index % len(templates)],
            category,
            True
        )

    # For unknown meanings we DO NOT pretend to know
    # the correct situation.

    return (
        f"এই অর্থের পরিস্থিতি বোঝাতে কোন বাংলা "
        f"অভিব্যক্তি ব্যবহার করা যায়?",
        "general",
        False
    )


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("BENGALI CULTURAL RAG — BENCHMARK V3")
print("DATASET-AWARE / NO API")
print("=" * 75)

print("\nDataset:")
print(DATASET_PATH)

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


if not isinstance(dataset, list):

    raise RuntimeError(
        "Expected final_cultural_dataset.json to contain a JSON list."
    )


print(
    f"Dataset records: {len(dataset)}"
)


# ============================================================
# VALIDATION
# ============================================================

valid = []
excluded = []

for record in dataset:

    ok, reason = valid_record(record)

    if ok:

        valid.append(record)

    else:

        excluded.append({
            "phrase": get_phrase(record),
            "reason": reason
        })


print(
    f"Valid records: {len(valid)}"
)

print(
    f"Excluded records: {len(excluded)}"
)


if len(valid) == 0:

    print("\nERROR:")
    print(
        "No valid records were found."
    )

    print(
        "\nFirst dataset record:"
    )

    print(
        json.dumps(
            dataset[0],
            ensure_ascii=False,
            indent=2
        )
    )

    raise RuntimeError(
        "Dataset field structure still does not match."
    )


# ============================================================
# UNIQUE
# ============================================================

unique = {}

for record in valid:

    phrase = get_phrase(record)

    key = norm(phrase)

    if key not in unique:

        unique[key] = record


valid = list(unique.values())

print(
    f"Unique expressions: {len(valid)}"
)


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(valid)

selected = valid[
    :min(TARGET_RECORDS, len(valid))
]

print(
    f"Selected records: {len(selected)}"
)


# ============================================================
# BUILD
# ============================================================

benchmark = []

used_queries = set()

auto_generated = 0
manual_review_needed = 0

for i, record in enumerate(
    selected,
    start=1
):

    phrase = get_phrase(record)

    meaning = get_meaning(record)

    literal = get_literal(record)

    tone = get_tone(record)

    query, category, reliable = make_query(
        record,
        i
    )

    if not reliable:

        manual_review_needed += 1

        # Do NOT mark unreliable automatically
        verified = False

    else:

        auto_generated += 1
        verified = True


    # --------------------------------------------------------
    # Avoid query duplication
    # --------------------------------------------------------

    qkey = norm(query)

    if qkey in used_queries:

        # Create meaning-specific fallback
        query = (
            f"{phrase} না বলে, এর অর্থ "
            f"'{meaning}' বোঝাতে কোন বাংলা "
            f"অভিব্যক্তি ব্যবহার করা যায়?"
        )

        # IMPORTANT:
        # This contains the phrase and therefore should NOT
        # be used as a real benchmark query.

        verified = False

    used_queries.add(
        norm(query)
    )


    benchmark.append({

        "id": i,

        "query": query,

        "gold_phrase": phrase,

        "cultural_meaning": meaning,

        "literal_meaning": literal,

        "tone": tone,

        "semantic_category": category,

        "verified": verified,

        "annotation_source":
            "dataset_meaning_grounded_v3"

    })


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
        benchmark,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

report = {

    "dataset_records":
        len(dataset),

    "valid_records":
        len(valid),

    "selected_records":
        len(selected),

    "verified_records":
        sum(
            1
            for x in benchmark
            if x["verified"]
        ),

    "manual_review_needed":
        manual_review_needed,

    "excluded_records":
        len(excluded),

    "excluded":
        excluded,

    "uses_gemini":
        False,

    "uses_groq":
        False,

    "uses_api":
        False

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
# SHOW SAMPLE
# ============================================================

print("\n" + "=" * 75)
print("BENCHMARK GENERATED")
print("=" * 75)

print(
    f"\nTotal selected : {len(benchmark)}"
)

print(
    f"Verified       : {report['verified_records']}"
)

print(
    f"Needs review   : {manual_review_needed}"
)

print(
    f"Excluded       : {len(excluded)}"
)


print("\nFIRST 10 RECORDS")

for item in benchmark[:10]:

    print("\n--------------------------------------")

    print(
        "Query:",
        item["query"]
    )

    print(
        "Gold:",
        item["gold_phrase"]
    )

    print(
        "Meaning:",
        item["cultural_meaning"]
    )

    print(
        "Category:",
        item["semantic_category"]
    )

    print(
        "Verified:",
        item["verified"]
    )


print("\nSaved:")
print(OUTPUT_PATH)
print(REPORT_PATH)

print("\nDONE")
print("=" * 75)