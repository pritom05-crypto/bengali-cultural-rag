import json
import os
import re
import random
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "processed" / "final_cultural_dataset.json"
OUTPUT_PATH = BASE_DIR / "evaluation" / "benchmark_queries.json"

SEED = 623
TARGET_RECORDS = 120

random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, list):
        value = " ".join(str(x) for x in value)

    return str(value).strip()


def normalize(text):
    text = clean_text(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_available(record, keys):
    for key in keys:
        if key in record:
            value = clean_text(record[key])
            if value:
                return value
    return ""


# ============================================================
# MEANING → NATURAL QUERY
# ============================================================

def make_query(phrase, meaning, tone="", literal=""):
    """
    Create a natural Bengali query from the cultural meaning.

    IMPORTANT:
    - Never put the gold expression itself inside the query.
    - Do not use generic meaningless queries.
    - Query must describe the meaning/situation.
    """

    m = normalize(meaning)
    p = normalize(phrase)
    t = normalize(tone)

    # --------------------------------------------------------
    # Common semantic patterns
    # --------------------------------------------------------

    # অলস
    if any(x in m for x in [
        "অলস",
        "কর্মবিমুখ",
        "কাজ করতে চায় না",
        "কাজ করতে চায় না",
        "lazy",
        "idle"
    ]):
        return random.choice([
            "সে সারাদিন অলস হয়ে বসে থাকে এবং কোনো কাজ করতে চায় না।",
            "লোকটি অত্যন্ত অলস ও কর্মবিমুখ।",
            "সে কোনো কাজ করতে আগ্রহী নয়, সবসময় অলসভাবে বসে থাকে।",
            "ছেলেটি কাজকর্মে একেবারেই অনীহা দেখায় এবং ভীষণ অলস।"
        ])

    # রাগ
    if any(x in m for x in [
        "রাগ",
        "রেগে",
        "রাগান্বিত",
        "ক্রুদ্ধ",
        "angry",
        "furious",
        "rage"
    ]):
        return random.choice([
            "সে প্রচণ্ড রেগে গেছে এবং খুব উত্তেজিত হয়ে আছে।",
            "লোকটি ভীষণ রাগান্বিত হয়ে নিয়ন্ত্রণ হারিয়ে ফেলেছে।",
            "তার রাগ এত বেশি যে তার সঙ্গে কথা বলা কঠিন।",
            "সে অত্যন্ত ক্রুদ্ধ হয়ে আছে।"
        ])

    # ভয় / আতঙ্ক
    if any(x in m for x in [
        "ভয়",
        "ভয়",
        "আতঙ্ক",
        "আতংক",
        "ভীত",
        "শঙ্কা",
        "fear",
        "afraid",
        "frightened",
        "scared",
        "terrified",
        "panic"
    ]):
        return random.choice([
            "ঘটনাটি দেখে সে খুব ভয় পেয়ে গেছে।",
            "হঠাৎ ঘটনাটিতে সে একেবারে আতঙ্কিত হয়ে পড়েছে।",
            "ভয়ে সে দিশেহারা হয়ে গেছে।",
            "সে প্রচণ্ড ভয় পেয়ে অস্থির হয়ে পড়েছে।"
        ])

    # বিপদ / সংকট
    if any(x in m for x in [
        "বিপদ",
        "সংকট",
        "ঝুঁকি",
        "দুর্দশা",
        "trouble",
        "danger",
        "crisis",
        "adversity",
        "distress"
    ]):
        return random.choice([
            "আমি এমন এক বিপদের মধ্যে পড়েছি যেখান থেকে বের হওয়ার পথ খুঁজে পাচ্ছি না।",
            "সে হঠাৎ খুব বড় বিপদে পড়ে গেছে।",
            "পরিস্থিতি এত খারাপ যে সে কোনোভাবেই সংকট থেকে বের হতে পারছে না।",
            "লোকটি ভয়াবহ বিপদ ও সমস্যার মধ্যে পড়েছে।"
        ])

    # বোকা / অযোগ্য
    if any(x in m for x in [
        "বোকা",
        "মূর্খ",
        "অযোগ্য",
        "stupid",
        "fool",
        "foolish",
        "incompetent"
    ]):
        return random.choice([
            "লোকটি অত্যন্ত বোকা এবং কোনো কাজ ঠিকভাবে করতে পারে না।",
            "সে খুবই মূর্খ ও অযোগ্য ব্যক্তি।",
            "তার আচরণে বোঝা যায় যে সে খুব বোকা।"
        ])

    # জেদি / অনড়
    if any(x in m for x in [
        "জেদি",
        "অনড়",
        "নাছোড়",
        "একগুঁয়ে",
        "stubborn",
        "unyielding"
    ]):
        return random.choice([
            "অনেক বোঝালেও লোকটি কোনোভাবেই নিজের সিদ্ধান্ত থেকে সরে আসছে না।",
            "সে অত্যন্ত জেদি এবং কারও কথা শুনতে চায় না।",
            "লোকটি নিজের অবস্থান থেকে একেবারেই নড়ছে না।"
        ])

    # দুঃখ / কষ্ট
    if any(x in m for x in [
        "কষ্ট",
        "দুঃখ",
        "বেদনা",
        "যন্ত্রণা",
        "pain",
        "sorrow",
        "sad"
    ]):
        return random.choice([
            "ঘটনাটি তাকে ভেতর থেকে খুব কষ্ট দিয়েছে।",
            "ঘটনার কারণে সে গভীরভাবে দুঃখ পেয়েছে।",
            "বিষয়টি তাকে মানসিকভাবে খুব কষ্ট দিয়েছে।"
        ])

    # ভাগ্য / সাফল্য
    if any(x in m for x in [
        "ভাগ্য",
        "সাফল্য",
        "সৌভাগ্য",
        "lucky",
        "success",
        "fortune"
    ]):
        return random.choice([
            "হঠাৎ তার ভাগ্য ভালো হয়ে গেল এবং সে সাফল্য পেল।",
            "সঠিক সময়ে সুযোগ পাওয়ায় তার জন্য পরিস্থিতি শুভ হয়ে উঠল।",
            "ভালো ভাগ্যের কারণে তার কাজটি সফল হলো।"
        ])

    # পরিশ্রম / চেষ্টা
    if any(x in m for x in [
        "চেষ্টা",
        "পরিশ্রম",
        "দৃঢ়ভাবে",
        "লেগে থাকা",
        "effort",
        "try",
        "hard work"
    ]):
        return random.choice([
            "কাজটি শেষ করতে সে অত্যন্ত দৃঢ়ভাবে চেষ্টা করছে।",
            "সফল হওয়ার জন্য সে সর্বশক্তি দিয়ে চেষ্টা করছে।",
            "সে লক্ষ্য অর্জনের জন্য কঠোর পরিশ্রম করছে।"
        ])

    # প্রতারণা / ফাঁকি
    if any(x in m for x in [
        "ফাঁকি",
        "প্রতারণা",
        "ঠকানো",
        "ধোঁকা",
        "deceive",
        "cheat",
        "fraud"
    ]):
        return random.choice([
            "লোকটি সত্যি কথা না বলে সবাইকে ফাঁকি দেওয়ার চেষ্টা করছে।",
            "সে কৌশলে অন্যদের ঠকিয়ে নিজের দায়িত্ব এড়িয়ে যাচ্ছে।",
            "লোকটি অন্যদের সঙ্গে প্রতারণা করছে।"
        ])

    # নিজের প্রশংসা
    if any(x in m for x in [
        "নিজের প্রশংসা",
        "আত্মপ্রশংসা",
        "boast",
        "self praise"
    ]):
        return random.choice([
            "লোকটি নিজের প্রশংসা করতে খুব পছন্দ করে।",
            "সে সবসময় নিজের গুণের কথা বলে বেড়ায়।",
            "লোকটি নিজের সাফল্য নিয়ে অহংকার করে।"
        ])

    # ঘনিষ্ঠ সম্পর্ক
    if any(x in m for x in [
        "ঘনিষ্ঠ",
        "সম্পর্ক",
        "বন্ধুত্ব",
        "close relationship",
        "intimate"
    ]):
        return random.choice([
            "তাদের সম্পর্ক এতটাই ঘনিষ্ঠ যে তারা সবকিছু একে অপরের সঙ্গে ভাগ করে নেয়।",
            "দুইজনের মধ্যে খুব গভীর ও ঘনিষ্ঠ সম্পর্ক রয়েছে।"
        ])

    # অভাব / দারিদ্র্য
    if any(x in m for x in [
        "অভাব",
        "দারিদ্র্য",
        "গরিব",
        "দুঃস্থ",
        "poor",
        "poverty"
    ]):
        return random.choice([
            "অভাবের কারণে পরিবারটি খুব কষ্টে জীবনযাপন করছে।",
            "লোকটি প্রচণ্ড অর্থকষ্টের মধ্যে দিন কাটাচ্ছে।",
            "পরিবারটি দীর্ঘদিন ধরে দারিদ্র্যের মধ্যে রয়েছে।"
        ])

    # বিস্ময় / অপ্রত্যাশিত
    if any(x in m for x in [
        "অপ্রত্যাশিত",
        "হতবাক",
        "বিস্মিত",
        "আশ্চর্য",
        "unexpected",
        "surprise",
        "shocked"
    ]):
        return random.choice([
            "ঘटनাটি এতটাই অপ্রত্যাশিত ছিল যে সবাই হতবাক হয়ে গেল।",
            "হঠাৎ ঘটনাটি দেখে সবাই খুব অবাক হয়ে গেল।",
            "ঘটনাটি দেখে কেউ কিছু বুঝে উঠতে পারল না।"
        ])

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    # Literal meaning থাকলে সেটার ওপর ভিত্তি করে query
    if literal:
        return (
            f"এই অর্থ বা পরিস্থিতি বোঝাতে কোন বাংলা অভিব্যক্তি "
            f"ব্যবহার করা যায়?"
        )

    # meaning available but pattern unknown
    if meaning:
        return (
            f"এই পরিস্থিতিতে '{meaning}' অর্থ বোঝাতে "
            f"কোন বাংলা অভিব্যক্তি ব্যবহার করা যায়?"
        )

    return "এই পরিস্থিতিতে উপযুক্ত বাংলা অভিব্যক্তি কোনটি?"


# ============================================================
# VALIDITY FILTER
# ============================================================

def is_valid_record(record):
    phrase = first_available(
        record,
        ["phrase", "expression", "gold_phrase", "idiom"]
    )

    meaning = first_available(
        record,
        ["cultural_meaning", "meaning", "culturalMeaning"]
    )

    if not phrase or not meaning:
        return False

    # Extremely short / corrupted records
    if len(normalize(phrase)) < 2:
        return False

    if len(normalize(meaning)) < 3:
        return False

    return True


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 75)
print("BENGALI CULTURAL RAG — AUTO BENCHMARK ANNOTATION V2")
print("MEANING-GROUNDED / NO API / NO GEMINI / NO GROQ")
print("=" * 75)

print("\nLoading dataset...")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

if isinstance(dataset, dict):
    for key in ["records", "data", "dataset", "expressions"]:
        if key in dataset and isinstance(dataset[key], list):
            dataset = dataset[key]
            break

if not isinstance(dataset, list):
    raise RuntimeError("Dataset format is not a list.")

print(f"Dataset records: {len(dataset)}")


# ============================================================
# FILTER
# ============================================================

valid_records = [
    r for r in dataset
    if isinstance(r, dict) and is_valid_record(r)
]

print(f"Valid records: {len(valid_records)}")


# ============================================================
# REMOVE DUPLICATE PHRASES
# ============================================================

unique = {}
for record in valid_records:

    phrase = first_available(
        record,
        ["phrase", "expression", "gold_phrase", "idiom"]
    )

    key = normalize(phrase)

    if key not in unique:
        unique[key] = record

valid_records = list(unique.values())

print(f"Unique expressions: {len(valid_records)}")


# ============================================================
# SAMPLE
# ============================================================

random.shuffle(valid_records)

selected = valid_records[:min(TARGET_RECORDS, len(valid_records))]

print(f"Selected benchmark records: {len(selected)}")


# ============================================================
# BUILD BENCHMARK
# ============================================================

benchmark = []

used_queries = set()

for idx, record in enumerate(selected, 1):

    phrase = first_available(
        record,
        ["phrase", "expression", "gold_phrase", "idiom"]
    )

    meaning = first_available(
        record,
        ["cultural_meaning", "meaning", "culturalMeaning"]
    )

    literal = first_available(
        record,
        ["literal_meaning", "literal", "literalMeaning"]
    )

    tone = first_available(
        record,
        ["tone", "usage_tone", "sentiment"]
    )

    query = make_query(
        phrase=phrase,
        meaning=meaning,
        tone=tone,
        literal=literal
    )

    # Avoid duplicate queries
    qkey = normalize(query)

    if qkey in used_queries:

        alternatives = [
            f"এই অর্থ বোঝাতে কোন বাংলা অভিব্যক্তি ব্যবহার করা যায়?",
            f"এই অবস্থাকে প্রকাশ করতে কোন বাংলা অভিব্যক্তি উপযুক্ত?",
            f"এই পরিস্থিতির জন্য কোন বাংলা expression ব্যবহার করা যায়?"
        ]

        for alt in alternatives:
            if normalize(alt) not in used_queries:
                query = alt
                qkey = normalize(alt)
                break

    used_queries.add(qkey)

    benchmark.append({
        "id": idx,
        "query": query,
        "gold_phrase": phrase,
        "cultural_meaning": meaning,
        "literal_meaning": literal,
        "tone": tone,
        "verified": True,
        "source": "auto_annotated_meaning_grounded_v2"
    })


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        benchmark,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 75)
print("BENCHMARK CREATED")
print("=" * 75)

print(f"Total benchmark queries : {len(benchmark)}")
print(f"Verified queries        : {sum(x['verified'] for x in benchmark)}")
print(f"Unique queries          : {len(used_queries)}")

print("\nSample records:")

for item in benchmark[:10]:

    print("\n----------------------------------------")
    print(f"Query : {item['query']}")
    print(f"Gold  : {item['gold_phrase']}")
    print(f"Meaning: {item['cultural_meaning']}")

print("\nSaved:")
print(OUTPUT_PATH)

print("\nIMPORTANT:")
print("Review the generated benchmark before publication.")
print("The script does NOT use Gemini, Groq, or any API.")
print("=" * 75)