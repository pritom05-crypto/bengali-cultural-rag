import json
import random
import re
from pathlib import Path

# ============================================================
# BENGALI CULTURAL RAG
# BENCHMARK AUTO ANNOTATION V4
#
# IMPORTANT:
# - NO GEMINI
# - NO GROQ
# - NO API
# - GOLD PHRASE ALWAYS COMES FROM THE SAME DATASET RECORD
# - NO RANDOM CROSS-RECORD GOLD ASSIGNMENT
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

TARGET_RECORDS = 100
SEED = 623

random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def clean(value):
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(
            str(x) for x in value
        )

    return str(value).strip()


def norm(value):
    text = clean(value)

    text = text.lower()

    # Normalize Bengali whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_phrase(record):
    return clean(
        record.get("bengali_phrase")
    )


def get_literal(record):
    return clean(
        record.get("literal_meaning")
    )


def get_cultural(record):
    return clean(
        record.get("cultural_meaning")
    )


def get_tone(record):

    values = []

    for key in [
        "primary_tone",
        "secondary_tone",
        "intended_emotion_tone"
    ]:

        value = clean(
            record.get(key)
        )

        if value and value not in values:
            values.append(value)

    return " / ".join(values)


# ============================================================
# KNOWN SUSPICIOUS RECORDS
# ============================================================

# These were repeatedly flagged during the previous validation
# process. We do not use them for automatic benchmark creation.

KNOWN_SUSPICIOUS = {

    "অর্ধচন্দ্র",
    "অষ্টরম্ভা",
    "অতি দর্পে হত লঙ্কা",
    "অঞ্চল প্রভাব",
    "অক্কা পাওয়া",

}


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

def validate_record(record):

    phrase = get_phrase(record)

    cultural = get_cultural(record)

    literal = get_literal(record)

    if not phrase:
        return False, "missing_phrase"

    if not cultural:
        return False, "missing_cultural_meaning"

    if len(norm(phrase)) < 2:
        return False, "phrase_too_short"

    if len(norm(cultural)) < 3:
        return False, "meaning_too_short"

    if phrase in KNOWN_SUSPICIOUS:
        return False, "known_suspicious"

    # Avoid obvious placeholder-like values
    bad_values = {
        "unknown",
        "n/a",
        "none",
        "null",
        "undefined",
        "-"
    }

    if norm(cultural) in bad_values:
        return False, "invalid_meaning"

    return True, ""


# ============================================================
# SEMANTIC CATEGORY
# ============================================================

def detect_category(record):

    text = norm(
        " ".join([
            get_cultural(record),
            get_literal(record),
            get_tone(record),
            get_phrase(record)
        ])
    )

    categories = {

        "anger": [
            "angry",
            "furious",
            "anger",
            "rage",
            "রাগ",
            "রেগে",
            "ক্রোধ",
            "ক্রুদ্ধ",
            "রাগান্বিত"
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
            "শঙ্কা"
        ],

        "danger": [
            "danger",
            "trouble",
            "crisis",
            "distress",
            "adversity",
            "বিপদ",
            "সংকট",
            "ঝুঁকি",
            "দুর্দশা",
            "আতান্তর"
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
            "কাজ করতে চায় না"
        ],

        "sadness": [
            "sad",
            "sadness",
            "sorrow",
            "grief",
            "pain",
            "দুঃখ",
            "বেদনা",
            "কষ্ট",
            "যন্ত্রণা"
        ],

        "success": [
            "success",
            "successful",
            "lucky",
            "fortune",
            "সাফল্য",
            "সফল",
            "ভাগ্য",
            "সৌভাগ্য"
        ],

        "failure": [
            "failure",
            "failed",
            "defeat",
            "ব্যর্থ",
            "ব্যর্থতা",
            "পরাজয়",
            "পরাজয়"
        ],

        "deception": [
            "deception",
            "deceive",
            "cheat",
            "fake",
            "fraud",
            "প্রতারণা",
            "ঠকানো",
            "ফাঁকি",
            "ধোঁকা"
        ],

        "stubborn": [
            "stubborn",
            "unyielding",
            "জেদি",
            "জেদ",
            "অনড়",
            "অনড়",
            "একগুঁয়ে",
            "একগুঁয়ে"
        ],

        "boast": [
            "boast",
            "boasting",
            "self praise",
            "নিজের প্রশংসা",
            "বড়াই",
            "বড়াই",
            "আত্মপ্রশংসা"
        ],

        "confusion": [
            "confused",
            "confusion",
            "বিভ্রান্ত",
            "বিভ্রান্তি",
            "হতবুদ্ধি"
        ],

        "surprise": [
            "surprise",
            "surprised",
            "shocked",
            "unexpected",
            "বিস্মিত",
            "আশ্চর্য",
            "হতবাক",
            "অপ্রত্যাশিত"
        ],

        "poverty": [
            "poor",
            "poverty",
            "দারিদ্র্য",
            "দারিদ্র",
            "অভাব",
            "গরিব",
            "দুঃস্থ"
        ],

        "effort": [
            "effort",
            "hard work",
            "trying",
            "চেষ্টা",
            "পরিশ্রম",
            "প্রচেষ্টা"
        ],

        "procrastination": [
            "procrastination",
            "delay",
            "delayed",
            "বিলম্ব",
            "দেরি",
            "গড়িমসি",
            "গড়িমসি"
        ],

        "criticism": [
            "critical",
            "criticism",
            "সমালোচনা",
            "সমালোচনামূলক"
        ],

        "dependency": [
            "dependent",
            "dependence",
            "নির্ভরশীল",
            "নির্ভরশীলতা"
        ]

    }

    for category, keywords in categories.items():

        for keyword in keywords:

            if keyword in text:
                return category

    return "general"


# ============================================================
# QUERY TEMPLATES
#
# IMPORTANT:
# The query describes the meaning.
# GOLD IS ALWAYS record["bengali_phrase"].
# ============================================================

QUERY_TEMPLATES = {

    "anger": [
        "সে খুব রেগে গেছে।",
        "সে প্রচণ্ড রাগান্বিত হয়ে আছে।",
        "লোকটি ভীষণ রেগে গেছে।",
        "তার প্রচণ্ড রাগ হয়েছে।"
    ],

    "fear": [
        "সে খুব ভয় পেয়ে গেছে।",
        "ঘটনাটি দেখে সে আতঙ্কিত হয়ে গেছে।",
        "ভয়ে সে অস্থির হয়ে পড়েছে।",
        "সে প্রচণ্ড ভয় পেয়েছে।"
    ],

    "danger": [
        "আমি খুব বিপদে পড়ে গেছি।",
        "সে হঠাৎ বড় বিপদের মধ্যে পড়েছে।",
        "আমি ভয়াবহ সমস্যার মধ্যে পড়েছি।",
        "সে কঠিন সংকটে পড়ে গেছে।"
    ],

    "lazy": [
        "সে খুব অলস।",
        "সে কোনো কাজ করতে চায় না।",
        "সে সারাদিন অলসভাবে বসে থাকে।",
        "ছেলেটি কাজকর্মে খুবই অনীহা দেখায়।"
    ],

    "sadness": [
        "ঘটনাটি তাকে খুব কষ্ট দিয়েছে।",
        "সে গভীরভাবে দুঃখ পেয়েছে।",
        "বিষয়টি তাকে খুব দুঃখ দিয়েছে।",
        "ঘটনার কারণে সে খুব কষ্টে আছে।"
    ],

    "success": [
        "তার ভাগ্য ভালো হওয়ায় সে সফল হয়েছে।",
        "শেষ পর্যন্ত তার কাজটি সফল হয়েছে।",
        "সে ভালো সুযোগ পেয়ে সফল হয়েছে।",
        "তার ভাগ্য হঠাৎ ভালো হয়ে গেছে।"
    ],

    "failure": [
        "তার চেষ্টা শেষ পর্যন্ত ব্যর্থ হয়েছে।",
        "তার পরিকল্পনা ব্যর্থ হয়ে গেছে।",
        "সে কাজে ব্যর্থ হয়েছে।",
        "চেষ্টার পরও সে সফল হতে পারেনি।"
    ],

    "deception": [
        "লোকটি অন্যদের ঠকিয়েছে।",
        "সে কৌশলে সবাইকে ফাঁকি দিয়েছে।",
        "লোকটি প্রতারণা করে নিজের কাজ হাসিল করেছে।",
        "সে অন্যদের সঙ্গে প্রতারণা করেছে।"
    ],

    "stubborn": [
        "অনেক বোঝালেও সে নিজের অবস্থান থেকে সরছে না।",
        "সে খুব জেদি এবং কারও কথা শুনছে না।",
        "লোকটি নিজের সিদ্ধান্ত থেকে নড়ছে না।",
        "সে কোনোভাবেই নিজের মত পরিবর্তন করছে না।"
    ],

    "boast": [
        "সে সবসময় নিজের প্রশংসা করে।",
        "লোকটি নিজের যোগ্যতা নিয়ে বড়াই করে।",
        "সে নিজের সাফল্যের কথা বলে বেড়ায়।",
        "লোকটি সবসময় নিজের কথা বড় করে বলে।"
    ],

    "confusion": [
        "পরিস্থিতিতে পড়ে সে একেবারে হতবুদ্ধি হয়ে গেছে।",
        "সে কী করবে বুঝতে পারছে না।",
        "ঘটনাটি বুঝতে না পেরে সে বিভ্রান্ত হয়ে গেছে।",
        "সে পরিস্থিতি নিয়ে খুব বিভ্রান্ত।"
    ],

    "surprise": [
        "ঘটনাটি দেখে সে খুব অবাক হয়ে গেছে।",
        "হঠাৎ ঘটনাটি দেখে সবাই হতবাক।",
        "সে অপ্রত্যাশিত ঘটনায় বিস্মিত হয়ে গেছে।",
        "ঘটনাটি তার কাছে খুব আশ্চর্যজনক ছিল।"
    ],

    "poverty": [
        "পরিবারটি খুব অভাবের মধ্যে আছে।",
        "লোকটি অর্থকষ্টে দিন কাটাচ্ছে।",
        "তারা দীর্ঘদিন ধরে দারিদ্র্যের মধ্যে আছে।",
        "অভাবের কারণে তাদের জীবন খুব কঠিন।"
    ],

    "effort": [
        "সে সফল হওয়ার জন্য কঠোর পরিশ্রম করছে।",
        "লক্ষ্য অর্জনের জন্য সে অনেক চেষ্টা করছে।",
        "সে কাজটি শেষ করার জন্য প্রাণপণে চেষ্টা করছে।",
        "সে সফল হওয়ার জন্য সর্বশক্তি দিয়ে চেষ্টা করছে।"
    ],

    "procrastination": [
        "সে কাজটি বারবার পিছিয়ে দিচ্ছে।",
        "সে সময়মতো কাজ শেষ করছে না।",
        "সে কাজ করতে দেরি করছে।",
        "সে সবসময় কাজ ফেলে রাখে।"
    ],

    "criticism": [
        "লোকটির কাজ নিয়ে সবাই সমালোচনা করছে।",
        "তার আচরণ নিয়ে মানুষ অসন্তুষ্ট।",
        "লোকটির কাজের জন্য তাকে সমালোচনা করা হচ্ছে।"
    ],

    "dependency": [
        "সে সবসময় অন্যের সাহায্যের ওপর নির্ভর করে।",
        "নিজে কিছু না করে সে অন্যের ওপর নির্ভরশীল।",
        "লোকটি নিজের কাজের জন্য সবসময় অন্যের সাহায্য নেয়।"
    ]
}


# ============================================================
# FALLBACK QUERY
# ============================================================

def make_query(record, occurrence):

    category = detect_category(record)

    templates = QUERY_TEMPLATES.get(category)

    if templates:

        query = templates[
            occurrence % len(templates)
        ]

        return query, category, True

    # IMPORTANT:
    # Unknown category is NOT automatically verified.

    return (
        "",
        category,
        False
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 75)
print("BENGALI CULTURAL RAG — BENCHMARK V4")
print("MEANING-GROUNDED / SAME-RECORD GOLD")
print("NO GEMINI | NO GROQ | NO API")
print("=" * 75)

print("\nLoading dataset...")

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

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
# VALID RECORDS
# ============================================================

valid_records = []

excluded = []

for record in dataset:

    ok, reason = validate_record(record)

    if ok:

        valid_records.append(record)

    else:

        excluded.append({

            "phrase":
                get_phrase(record),

            "reason":
                reason

        })


# ============================================================
# REMOVE DUPLICATE PHRASES
# ============================================================

unique_records = {}

for record in valid_records:

    phrase = get_phrase(record)

    key = norm(phrase)

    if key not in unique_records:

        unique_records[key] = record


valid_records = list(
    unique_records.values()
)


print(
    f"Valid unique records: {len(valid_records)}"
)

print(
    f"Excluded records: {len(excluded)}"
)


if not valid_records:

    raise RuntimeError(
        "No valid dataset records found."
    )


# ============================================================
# RANDOM SHUFFLE
# ============================================================

random.shuffle(
    valid_records
)


selected = valid_records[
    :min(
        TARGET_RECORDS,
        len(valid_records)
    )
]


print(
    f"Selected records: {len(selected)}"
)


# ============================================================
# BUILD BENCHMARK
# ============================================================

benchmark = []

used_queries = set()

category_counter = {}

skipped_unknown = 0

for record in selected:

    phrase = get_phrase(record)

    cultural = get_cultural(record)

    literal = get_literal(record)

    tone = get_tone(record)

    category = detect_category(record)

    count = category_counter.get(
        category,
        0
    )

    query, category, reliable = make_query(
        record,
        count
    )

    category_counter[category] = count + 1

    if not reliable:

        skipped_unknown += 1

        continue


    # --------------------------------------------------------
    # Query uniqueness
    # --------------------------------------------------------

    qkey = norm(query)

    if qkey in used_queries:

        # Try another template
        templates = QUERY_TEMPLATES.get(
            category,
            []
        )

        found = False

        for alternative in templates:

            alt_key = norm(
                alternative
            )

            if alt_key not in used_queries:

                query = alternative

                found = True

                break

        if not found:

            skipped_unknown += 1

            continue


    used_queries.add(
        norm(query)
    )


    # ========================================================
    # CRITICAL:
    #
    # gold_phrase MUST be this SAME record's phrase.
    #
    # NEVER choose another phrase based on category.
    # ========================================================

    gold_phrase = phrase


    benchmark.append({

        "id":
            record.get("id"),

        "query":
            query,

        "gold_phrase":
            gold_phrase,

        "cultural_meaning":
            cultural,

        "literal_meaning":
            literal,

        "tone":
            tone,

        "semantic_category":
            category,

        "verified":
            True,

        "annotation_source":
            "same_record_meaning_grounded_v4"

    })


# ============================================================
# SAFETY CHECK
# ============================================================

print("\nRunning benchmark integrity checks...")


integrity_errors = []


for item in benchmark:

    gold = norm(
        item["gold_phrase"]
    )

    # Gold must exist
    if not gold:

        integrity_errors.append({
            "id": item["id"],
            "error": "empty_gold"
        })

    # Query must exist
    if not norm(
        item["query"]
    ):

        integrity_errors.append({
            "id": item["id"],
            "error": "empty_query"
        })


if integrity_errors:

    raise RuntimeError(
        "Benchmark integrity failed:\n"
        +
        json.dumps(
            integrity_errors,
            ensure_ascii=False,
            indent=2
        )
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
        benchmark,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

report = {

    "version":
        "V4",

    "dataset_records":
        len(dataset),

    "valid_unique_records":
        len(valid_records),

    "selected_records":
        len(selected),

    "benchmark_records":
        len(benchmark),

    "verified_records":
        sum(
            1
            for x in benchmark
            if x["verified"]
        ),

    "skipped_unknown_category":
        skipped_unknown,

    "excluded_records":
        len(excluded),

    "uses_gemini":
        False,

    "uses_groq":
        False,

    "uses_api":
        False,

    "gold_assignment":
        "same_dataset_record",

    "integrity_errors":
        len(integrity_errors)

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
# DISPLAY
# ============================================================

print("\n" + "=" * 75)
print("BENCHMARK CREATED")
print("=" * 75)

print(
    f"\nDataset records       : "
    f"{len(dataset)}"
)

print(
    f"Valid unique records  : "
    f"{len(valid_records)}"
)

print(
    f"Selected records      : "
    f"{len(selected)}"
)

print(
    f"Benchmark queries     : "
    f"{len(benchmark)}"
)

print(
    f"Verified queries      : "
    f"{sum(x['verified'] for x in benchmark)}"
)

print(
    f"Skipped unknown       : "
    f"{skipped_unknown}"
)


# ============================================================
# SAMPLE
# ============================================================

print("\n" + "=" * 75)
print("FIRST 15 BENCHMARK RECORDS")
print("=" * 75)

for i, item in enumerate(
    benchmark[:15],
    1
):

    print(
        f"\n[{i}]"
    )

    print(
        "Query      :",
        item["query"]
    )

    print(
        "Gold       :",
        item["gold_phrase"]
    )

    print(
        "Meaning    :",
        item["cultural_meaning"]
    )

    print(
        "Category   :",
        item["semantic_category"]
    )

    print(
        "Verified   :",
        item["verified"]
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("SAVED")
print("=" * 75)

print(
    OUTPUT_PATH
)

print(
    REPORT_PATH
)

print("\nDONE.")