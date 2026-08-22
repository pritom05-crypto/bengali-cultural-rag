# -*- coding: utf-8 -*-

"""
BENGALI CULTURAL RAG
PUBLICATION BENCHMARK

Purpose:
Create a semantically valid benchmark where
multiple culturally valid expressions can be accepted.

NO API
NO GEMINI
NO GROQ
"""

import json
from pathlib import Path


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
    / "publication_benchmark_report.json"
)


# ============================================================
# PUBLICATION BENCHMARK
# ============================================================

BENCHMARK = [

    {
        "intent": "lazy",
        "query": "সে খুব অলস এবং কোনো কাজ করতে চায় না।",

        "gold_phrase": "কুঁড়ের বাঘ",

        "acceptable_phrases": [
            "কুঁড়ের বাঘ",
            "কুঁড়ের বাঘ",
            "কুড়ের বাদশা",
            "অকর্মার ধাড়ি",
            "উদোগেঁড়ে",
            "ইতুনিদকুঁড়ে",
            "অপোগণ্ড"
        ]
    },

    {
        "intent": "angry",
        "query": "সে খুব রেগে গেছে এবং প্রচণ্ড রাগান্বিত।",

        "gold_phrase": "অগ্নিশর্মা",

        "acceptable_phrases": [
            "অগ্নিশর্মা"
        ]
    },

    {
        "intent": "danger",
        "query":
            "সে এমন বিপদে পড়েছে যেখান থেকে "
            "বের হওয়ার কোনো পথ পাচ্ছে না।",

        "gold_phrase": "অকূল পাথারে পড়া",

        "acceptable_phrases": [
            "অকূল পাথারে পড়া",
            "অকূল পাথার",
            "আতান্তরে পড়া",
            "অথৈ জল",
            "আকাশ ভেঙ্গে পড়া",
            "খইয়ের বন্ধনে পড়া"
        ]
    },

    {
        "intent": "confused",
        "query":
            "সে কী করবে বুঝতে পারছে না এবং "
            "খুব বিভ্রান্ত হয়ে গেছে।",

        "gold_phrase": "অন্ধকার দেখা",

        "acceptable_phrases": [
            "অন্ধকার দেখা"
        ]
    },

    {
        "intent": "success",
        "query":
            "শেষ পর্যন্ত সে সফল হয়েছে এবং "
            "তার লক্ষ্য অর্জন করেছে।",

        "gold_phrase": "কেল্লা ফতে",

        "acceptable_phrases": [
            "কেল্লা ফতে"
        ]
    },

    {
        "intent": "procrastination",
        "query":
            "সে কাজটি বারবার পিছিয়ে দিচ্ছে এবং "
            "সময় নষ্ট করছে।",

        "gold_phrase": "আঠার মাসে বছর",

        "acceptable_phrases": [
            "আঠার মাসে বছর"
        ]
    },

    {
        "intent": "deception",
        "query":
            "লোকটি কৌশল করে অন্যদের ঠকিয়েছে।",

        "gold_phrase": "কারিকুরি",

        "acceptable_phrases": [
            "কারিকুরি",
            "গণ্ডায় আণ্ডা দেয়া"
        ]
    },

    {
        "intent": "hardwork",
        "query":
            "সে সফল হওয়ার জন্য খুব কঠোর পরিশ্রম করছে।",

        "gold_phrase": "আদা জল খেয়ে লাগা",

        "acceptable_phrases": [
            "আদা জল খেয়ে লাগা",
            "উঠে পড়ে লাগা"
        ]
    },

    {
        "intent": "poverty",
        "query":
            "লোকটি খুব দরিদ্র এবং অর্থকষ্টে জীবনযাপন করছে।",

        "gold_phrase": "আকালকেঁড়ে",

        "acceptable_phrases": [
            "আকালকেঁড়ে",
            "অকড়িয়া",
            "উপোসি ছারপোকা",
            "কড়ার ভিখারি"
        ]
    },

    {
        "intent": "idle_talk",
        "query":
            "সে সারাদিন অপ্রয়োজনীয় ও অর্থহীন কথাবার্তা বলে।",

        "gold_phrase":
            "অগড়-বগড় / অগড়ম-বগড়ম",

        "acceptable_phrases": [
            "অগড়-বগড় / অগড়ম-বগড়ম",
            "অগড়-বগড়",
            "অগড়ম-বগড়ম",
            "খেজুরে আলাপ"
        ]
    },

    {
        "intent": "anxiety",
        "query":
            "পরিস্থিতির কারণে সে খুব উদ্বিগ্ন ও "
            "অস্থির হয়ে পড়েছে।",

        "gold_phrase": "আঁকুপাঁকু করা",

        "acceptable_phrases": [
            "আঁকুপাঁকু করা",
            "অস্থির পাজক",
            "অস্থির পঞ্চক"
        ]
    },

    {
        "intent": "pain",
        "query":
            "ঘটনাটি তাকে খুব কষ্ট দিয়েছে এবং "
            "তার মনে গভীর যন্ত্রণা হয়েছে।",

        "gold_phrase": "আঁতে ঘা",

        "acceptable_phrases": [
            "আঁতে ঘা",
            "কাঁটার জ্বালা",
            "অন্তর টিপুনি",
            "অগ্নিবান"
        ]
    }
]


# ============================================================
# LOAD DATASET
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


if isinstance(dataset, dict):

    for key in [
        "records",
        "data",
        "dataset",
        "items"
    ]:

        if isinstance(dataset.get(key), list):

            dataset = dataset[key]
            break


print("=" * 75)
print("BENGALI CULTURAL RAG — PUBLICATION BENCHMARK")
print("=" * 75)

print(
    f"Dataset records: {len(dataset)}"
)

# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if not text:
        return ""

    text = str(text).strip()

    text = (
        text
        .replace("কুঁড়ে", "কুঁড়ে")
        .replace("কুড়ে", "কুঁড়ে")
        .replace("পাথারে পড়া", "পাথারে পড়া")
        .replace("পড়া", "পড়া")
    )

    return text


dataset_phrases = set()

for record in dataset:

    if not isinstance(record, dict):
        continue

    for key in [
        "phrase",
        "expression",
        "bengali_phrase",
        "cultural_expression",
        "idiom"
    ]:

        value = record.get(key)

        if value:

            dataset_phrases.add(
                normalize(value)
            )


# ============================================================
# VALIDATE BENCHMARK
# ============================================================

validated = []

for item in BENCHMARK:

    acceptable = []

    for phrase in item[
        "acceptable_phrases"
    ]:

        if normalize(phrase) in dataset_phrases:

            acceptable.append(
                phrase
            )

    if not acceptable:

        print(
            "\nWARNING:",
            item["intent"],
            "has no dataset-supported phrase."
        )

        continue

    item["acceptable_phrases"] = acceptable

    item["verified"] = True

    validated.append(item)


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
        validated,
        f,
        ensure_ascii=False,
        indent=2
    )


report = {

    "dataset_records": len(dataset),

    "benchmark_records":
        len(validated),

    "verified_records":
        len(validated),

    "evaluation_policy":
        "Multiple semantically equivalent dataset expressions are acceptable.",

    "api_used": False,

    "gemini_used": False,

    "groq_used": False,

    "items": validated
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


print("\n" + "=" * 75)

print(
    "PUBLICATION BENCHMARK CREATED"
)

print("=" * 75)

print(
    f"Dataset records : {len(dataset)}"
)

print(
    f"Benchmark       : {len(validated)}"
)

print(
    f"Verified        : {len(validated)}"
)

print("\nSaved:")

print(OUTPUT_PATH)
print(REPORT_PATH)

print("\nNO API / NO GEMINI / NO GROQ")