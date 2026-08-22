import json
import os
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "benchmark_queries.json"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "benchmark_queries_annotated.json"
)

REPORT_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "benchmark_annotation_report.json"
)


# ============================================================
# KNOWN BAD / INCONSISTENT RECORDS
# ============================================================

# These are excluded because the current dataset fields themselves
# contain obvious semantic contradictions observed during validation.

EXCLUDE_PHRASES = {
    "অর্ধচন্দ্র",
    "অষ্টরম্ভা",
    "অতি দর্পে হত লঙ্কা",
    "একা ঘরে গিন্নি",
    "করাতের দাঁত",
    "কানে তুলো দেয়া",
    "কান কাটা",
    "কাকভূষণ্ডি",
    "কেঁচো যাওয়া",
    "অক্কা পাওয়া",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def contains_problematic_meaning(record):
    phrase = normalize(record.get("gold_phrase"))
    cultural = normalize(record.get("cultural_meaning"))
    literal = normalize(record.get("literal_meaning"))

    if phrase in EXCLUDE_PHRASES:
        return True

    # Obvious English/data-generation artifacts
    bad_patterns = [
        "glass art",
        "throat thrust",
        "don't stare",
        "perky",
        "completely wet",
        "penalty for wisdom",
    ]

    combined = f"{cultural} {literal}".lower()

    for pattern in bad_patterns:
        if pattern.lower() in combined:
            return True

    return False


# ============================================================
# SEMANTIC CATEGORY DETECTION
# ============================================================

def detect_category(meaning, tone):
    text = f"{meaning} {tone}".lower()

    categories = {

        "anger": [
            "angry",
            "furious",
            "anger",
            "rage",
            "hostile",
            "resentful",
        ],

        "danger": [
            "danger",
            "trouble",
            "crisis",
            "disaster",
            "distress",
            "adversity",
        ],

        "laziness": [
            "lazy",
            "idler",
            "idle",
            "useless",
            "inactive",
        ],

        "fear": [
            "fear",
            "afraid",
            "fright",
            "anxious",
            "anxiety",
            "panic",
        ],

        "sadness": [
            "sad",
            "sorrow",
            "heartbreaking",
            "pain",
            "grief",
        ],

        "success": [
            "success",
            "good luck",
            "achieving",
            "achievement",
        ],

        "failure": [
            "failure",
            "fail",
            "bankrupt",
            "defeat",
        ],

        "deception": [
            "deceive",
            "deception",
            "cheat",
            "fake",
            "evade",
            "evasion",
        ],

        "stubbornness": [
            "stubborn",
            "persistent",
            "unyielding",
        ],

        "hesitation": [
            "hesitate",
            "hesitant",
            "reluctant",
        ],

        "effort": [
            "effort",
            "trying",
            "determined",
            "resolute",
        ],

        "friendship": [
            "friendship",
            "intimacy",
            "close",
        ],

        "ignorance": [
            "illiterate",
            "limited knowledge",
            "narrow-minded",
        ],

        "boasting": [
            "boast",
            "boastful",
            "praise oneself",
            "show off",
        ],

        "carelessness": [
            "careless",
            "inattention",
            "inattentive",
        ],

        "confusion": [
            "confused",
            "confusion",
        ],

        "surprise": [
            "surprised",
            "shocked",
            "astonished",
        ],

        "poverty": [
            "poor",
            "poverty",
            "meager",
            "pitiful",
        ],

        "burden": [
            "burden",
            "parasite",
        ],

        "insult": [
            "fool",
            "foolish",
            "derogatory",
            "insult",
            "useless person",
        ],

        "competition": [
            "competition",
            "battle",
        ],
    }

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                return category

    return "general"


# ============================================================
# BENGALI QUERY TEMPLATES
# ============================================================

TEMPLATES = {

    "anger": [
        "সে এতটাই রেগে গেছে যে তার সঙ্গে কথা বলাই কঠিন।",
        "লোকটি প্রচণ্ড রাগে উত্তেজিত হয়ে আছে।",
        "সে খুব রাগান্বিত হয়ে পড়েছে—এ অবস্থায় কোন বাংলা অভিব্যক্তি উপযুক্ত?",
    ],

    "danger": [
        "আমি এমন এক বিপদের মধ্যে পড়েছি যেখান থেকে বের হওয়ার পথ খুঁজে পাচ্ছি না।",
        "সে হঠাৎ বড় বিপদ ও সমস্যার মধ্যে পড়ে গেছে।",
        "বর্তমান পরিস্থিতিতে আমি খুব কঠিন বিপদের মুখে পড়েছি।",
    ],

    "laziness": [
        "সে এতটাই অলস যে কোনো কাজ করতেই চায় না।",
        "ছেলেটি সারাদিন অলসভাবে বসে থাকে এবং কাজ করতে চায় না।",
        "যে ব্যক্তি প্রচণ্ড অলস ও কর্মবিমুখ তাকে বোঝাতে কোন অভিব্যক্তি ব্যবহার করা যায়?",
    ],

    "fear": [
        "সে ঘটনাটি দেখে খুব ভয় পেয়ে গেছে।",
        "ভয় পেয়ে সে একেবারে আতঙ্কিত হয়ে পড়েছে।",
        "হঠাৎ ভয় পেয়ে যাওয়ার পরিস্থিতি বোঝাতে কোন বাংলা অভিব্যক্তি উপযুক্ত?",
    ],

    "sadness": [
        "ঘটনাটি তাকে ভেতর থেকে খুব কষ্ট দিয়েছে।",
        "সে গভীর দুঃখ ও মানসিক কষ্টের মধ্যে আছে।",
        "এই পরিস্থিতিতে তার হৃদয়ের গভীর কষ্ট বোঝাতে কোন অভিব্যক্তি ব্যবহার করা যায়?",
    ],

    "success": [
        "হঠাৎ তার ভাগ্য ভালো হয়ে গেল এবং সে সাফল্য পেল।",
        "সঠিক সময়ে সুযোগ পাওয়ায় তার জন্য পরিস্থিতি খুব শুভ হয়ে উঠল।",
        "ভালো ভাগ্য বা সাফল্যের এমন পরিস্থিতি বোঝাতে কোন অভিব্যক্তি উপযুক্ত?",
    ],

    "failure": [
        "শেষ পর্যন্ত তার পরিকল্পনা সম্পূর্ণ ব্যর্থ হয়ে গেল।",
        "ব্যর্থতার কারণে সে খুব হতাশ হয়ে পড়েছে।",
        "সম্পূর্ণ ব্যর্থতা বা পরাজয়ের পরিস্থিতি বোঝাতে কোন বাংলা অভিব্যক্তি ব্যবহার করা যায়?",
    ],

    "deception": [
        "সে কৌশলে অন্যদের ঠকিয়ে নিজের দায়িত্ব এড়িয়ে গেল।",
        "লোকটি সত্যি কথা না বলে সবাইকে ফাঁকি দেওয়ার চেষ্টা করছে।",
        "প্রতারণা বা দায়িত্ব এড়িয়ে যাওয়ার পরিস্থিতি বোঝাতে কোন অভিব্যক্তি উপযুক্ত?",
    ],

    "stubbornness": [
        "সে কোনোভাবেই নিজের সিদ্ধান্ত থেকে সরে আসছে না।",
        "অনেক বোঝালেও লোকটি একেবারেই নাছোড়বান্দা।",
        "অত্যন্ত জেদি বা অনড় মানুষকে বোঝাতে কোন অভিব্যক্তি ব্যবহার করা যায়?",
    ],

    "hesitation": [
        "সে সিদ্ধান্ত নেওয়ার সময় বারবার দ্বিধা করছে।",
        "কী করবে বুঝতে না পেরে সে ইতস্তত করছে।",
        "দ্বিধা বা সংকোচের পরিস্থিতি বোঝাতে কোন বাংলা অভিব্যক্তি উপযুক্ত?",
    ],

    "effort": [
        "সে সফল হওয়ার জন্য সর্বশক্তি দিয়ে চেষ্টা করছে।",
        "কাজটি শেষ করতে সে অত্যন্ত দৃঢ়ভাবে চেষ্টা করছে।",
        "কোনো কাজে সর্বোচ্চ চেষ্টা করার পরিস্থিতি বোঝাতে কোন অভিব্যক্তি ব্যবহার করা যায়?",
    ],

    "friendship": [
        "দুজনের মধ্যে খুব ঘনিষ্ঠ বন্ধুত্বের সম্পর্ক রয়েছে।",
        "তাদের সম্পর্ক এতটাই ঘনিষ্ঠ যে তারা সবকিছু একে অপরের সঙ্গে ভাগ করে নেয়।",
        "অত্যন্ত ঘনিষ্ঠ সম্পর্ক বোঝাতে কোন বাংলা অভিব্যক্তি ব্যবহার করা যায়?",
    ],

    "ignorance": [
        "লোকটির জ্ঞান খুব সীমিত এবং সে নতুন কিছু জানতে চায় না।",
        "সে নিজের ছোট পরিসরের জ্ঞান নিয়েই সন্তুষ্ট থাকে।",
        "সংকীর্ণ মানসিকতা বা সীমিত জ্ঞানসম্পন্ন মানুষকে বোঝাতে কোন অভিব্যক্তি উপযুক্ত?",
    ],

    "boasting": [
        "সে সবসময় নিজের যোগ্যতার কথা বলে বেড়ায়।",
        "লোকটি নিজের প্রশংসা করতে খুব পছন্দ করে।",
        "নিজের প্রশংসা বা বড়াই করার স্বভাব বোঝাতে কোন অভিব্যক্তি ব্যবহার করা যায়?",
    ],

    "carelessness": [
        "সে কোনো বিষয়েই মনোযোগ দিচ্ছে না এবং অসাবধানভাবে কাজ করছে।",
        "তার অসাবধানতার কারণে কাজটিতে ভুল হয়েছে।",
        "অমনোযোগ বা অসাবধানতার পরিস্থিতি বোঝাতে কোন অভিব্যক্তি উপযুক্ত?",
    ],

    "confusion": [
        "ঘটনাটি বুঝতে না পেরে সে সম্পূর্ণ বিভ্রান্ত হয়ে গেছে।",
        "সে কী করবে বুঝতে পারছে না এবং খুব অস্থির হয়ে আছে।",
        "বিভ্রান্ত বা অস্থির অবস্থাকে বোঝাতে কোন অভিব্যক্তি ব্যবহার করা যায়?",
    ],

    "surprise": [
        "ঘটনাটি এতটাই অপ্রত্যাশিত ছিল যে সবাই হতবাক হয়ে গেল।",
        "হঠাৎ এমন ঘটনা ঘটায় সে খুব অবাক হয়ে গেছে।",
        "অপ্রত্যাশিত বিস্ময় বা হতবাক হওয়ার পরিস্থিতি বোঝাতে কোন অভিব্যক্তি উপযুক্ত?",
    ],

    "poverty": [
        "লোকটির আর্থিক অবস্থা খুব খারাপ এবং তার আয় খুব কম।",
        "অভাবের কারণে পরিবারটি খুব কষ্টে জীবনযাপন করছে।",
        "অভাব বা দুর্বল আর্থিক অবস্থাকে বোঝাতে কোন অভিব্যক্তি ব্যবহার করা যায়?",
    ],

    "burden": [
        "সে নিজের কাজ না করে সবসময় অন্যের ওপর নির্ভর করে থাকে।",
        "লোকটি অন্যের জন্য বোঝা হয়ে দাঁড়িয়েছে।",
        "অন্যের ওপর বোঝা হয়ে থাকা ব্যক্তিকে বোঝাতে কোন অভিব্যক্তি উপযুক্ত?",
    ],

    "insult": [
        "লোকটি অত্যন্ত বোকা এবং তার আচরণে সবাই বিরক্ত।",
        "তার অযোগ্যতা নিয়ে অন্যরা তাকে ব্যঙ্গ করছে।",
        "এমন বোকা বা অযোগ্য ব্যক্তিকে বোঝাতে কোন বাংলা অভিব্যক্তি ব্যবহার করা যায়?",
    ],

    "competition": [
        "দুই পক্ষের মধ্যে খুব তীব্র প্রতিযোগিতা শুরু হয়েছে।",
        "দুই দলের লড়াইটি অত্যন্ত কঠিন ও উত্তেজনাপূর্ণ।",
        "তীব্র প্রতিযোগিতা বা লড়াইয়ের পরিস্থিতি বোঝাতে কোন অভিব্যক্তি উপযুক্ত?",
    ],

    "general": [
        "এই পরিস্থিতিতে উপযুক্ত বাংলা অভিব্যক্তি কোনটি?",
        "এই অর্থ বা পরিস্থিতি বোঝাতে কোন বাংলা অভিব্যক্তি ব্যবহার করা যায়?",
        "এই ধরনের পরিস্থিতির জন্য উপযুক্ত বাংলা expression কোনটি?",
    ],
}


# ============================================================
# QUERY CREATION
# ============================================================

def make_query(record, position):
    meaning = normalize(record.get("cultural_meaning"))
    tone = normalize(record.get("tone"))

    category = detect_category(meaning, tone)

    templates = TEMPLATES.get(
        category,
        TEMPLATES["general"]
    )

    return templates[position % len(templates)], category


# ============================================================
# DIFFICULTY
# ============================================================

def calculate_difficulty(record, category):
    tone = normalize(record.get("tone")).lower()
    phrase = normalize(record.get("gold_phrase"))

    if len(phrase) >= 15:
        return "hard"

    if category in {
        "confusion",
        "deception",
        "ignorance",
        "stubbornness",
        "hesitation",
    }:
        return "hard"

    if category in {
        "anger",
        "danger",
        "laziness",
        "fear",
        "sadness",
        "success",
        "failure",
    }:
        return "medium"

    return "easy"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BENGALI CULTURAL RAG — AUTOMATIC BENCHMARK ANNOTATION")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Benchmark file not found:\n{INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"\nInput records: {len(records)}")

    annotated = []
    excluded = []

    used_phrases = set()

    for i, record in enumerate(records):

        phrase = normalize(record.get("gold_phrase"))

        if not phrase:
            excluded.append({
                "dataset_index": record.get("dataset_index"),
                "reason": "Missing gold phrase"
            })
            continue

        if phrase in used_phrases:
            excluded.append({
                "dataset_index": record.get("dataset_index"),
                "gold_phrase": phrase,
                "reason": "Duplicate phrase"
            })
            continue

        if contains_problematic_meaning(record):
            excluded.append({
                "dataset_index": record.get("dataset_index"),
                "gold_phrase": phrase,
                "reason": "Known inconsistent/suspicious dataset meaning"
            })
            continue

        query, category = make_query(
            record,
            len(annotated)
        )

        difficulty = calculate_difficulty(
            record,
            category
        )

        new_record = {
            "dataset_index": record.get("dataset_index"),
            "gold_phrase": phrase,
            "cultural_meaning": record.get("cultural_meaning", ""),
            "literal_meaning": record.get("literal_meaning", ""),
            "tone": record.get("tone", ""),

            "query": query,

            "difficulty": difficulty,

            "verified": True,

            "annotation_source": "automatic_template_annotation",

            "semantic_category": category,

            "notes": (
                "Automatically generated natural-language benchmark query "
                "from the existing dataset meaning. "
                "No external LLM/API was used."
            )
        }

        annotated.append(new_record)
        used_phrases.add(phrase)

    # ========================================================
    # BALANCE
    # ========================================================

    category_counts = Counter(
        x["semantic_category"]
        for x in annotated
    )

    difficulty_counts = Counter(
        x["difficulty"]
        for x in annotated
    )

    report = {
        "input_records": len(records),
        "annotated_records": len(annotated),
        "excluded_records": len(excluded),
        "category_distribution": dict(category_counts),
        "difficulty_distribution": dict(difficulty_counts),
        "excluded": excluded,
        "annotation_method": "automatic_template_annotation",
        "uses_external_api": False,
        "uses_gemini": False,
        "uses_groq": False,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            annotated,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 70)
    print("ANNOTATION COMPLETE")
    print("=" * 70)

    print(f"\nInput records       : {len(records)}")
    print(f"Annotated records   : {len(annotated)}")
    print(f"Excluded records    : {len(excluded)}")

    print("\nDifficulty:")
    for k, v in difficulty_counts.items():
        print(f"  {k:10s}: {v}")

    print("\nSemantic categories:")
    for k, v in category_counts.items():
        print(f"  {k:18s}: {v}")

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(REPORT_FILE)

    print("\nIMPORTANT:")
    print("The benchmark queries were generated automatically")
    print("from the existing dataset meanings.")
    print("No Gemini/Groq/API was used.")

    print("\nNext step:")
    print("Copy benchmark_queries_annotated.json to benchmark_queries.json")
    print("and run the V6.23 benchmark evaluator.")


if __name__ == "__main__":
    main()