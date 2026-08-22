import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT / "data" / "processed" / "final_cultural_dataset.json"
OUTPUT_PATH = ROOT / "evaluation" / "benchmark_queries.json"
REPORT_PATH = ROOT / "evaluation" / "clean_benchmark_report.json"


# ============================================================
# CLEAN, HIGH-CONFIDENCE BENCHMARK
# ============================================================

BENCHMARK = [
    {
        "id": "B01",
        "intent": "lazy",
        "query": "সে খুব অলস এবং কোনো কাজ করতে চায় না।",
        "gold_phrase": "কুঁড়ের বাঘ",
        "acceptable_phrases": [
            "কুঁড়ের বাঘ",
            "কুড়ের বাদশা",
            "অকর্মার ধাড়ি",
            "উদোগেঁড়ে",
            "ইতুনিদকুঁড়ে"
        ],
        "verified": True
    },

    {
        "id": "B02",
        "intent": "angry",
        "query": "সে খুব রেগে গেছে এবং প্রচণ্ড রাগান্বিত।",
        "gold_phrase": "অগ্নিশর্মা",
        "acceptable_phrases": [
            "অগ্নিশর্মা"
        ],
        "verified": True
    },

    {
        "id": "B03",
        "intent": "danger",
        "query": "সে এমন বিপদে পড়েছে যেখান থেকে বের হওয়ার কোনো পথ পাচ্ছে না।",
        "gold_phrase": "অকূল পাথারে পড়া",
        "acceptable_phrases": [
            "অকূল পাথারে পড়া",
            "আতান্তরে পড়া",
            "অকূল পাথার",
            "অথৈ জল",
            "আকাশ ভেঙ্গে পড়া"
        ],
        "verified": True
    },

    {
        "id": "B04",
        "intent": "confused",
        "query": "সে কী করবে বুঝতে পারছে না এবং খুব বিভ্রান্ত হয়ে গেছে।",
        "gold_phrase": "অন্ধকার দেখা",
        "acceptable_phrases": [
            "অন্ধকার দেখা"
        ],
        "verified": True
    },

    {
        "id": "B05",
        "intent": "success",
        "query": "শেষ পর্যন্ত সে সফল হয়েছে এবং তার লক্ষ্য অর্জন করেছে।",
        "gold_phrase": "কেল্লা ফতে",
        "acceptable_phrases": [
            "কেল্লা ফতে"
        ],
        "verified": True
    },

    {
        "id": "B06",
        "intent": "procrastination",
        "query": "সে কাজটি বারবার পিছিয়ে দিচ্ছে এবং সময় নষ্ট করছে।",
        "gold_phrase": "আঠার মাসে বছর",
        "acceptable_phrases": [
            "আঠার মাসে বছর"
        ],
        "verified": True
    },

    {
        "id": "B07",
        "intent": "deception",
        "query": "লোকটি কৌশল করে অন্যদের ঠকিয়েছে।",
        "gold_phrase": "কারিকুরি",
        "acceptable_phrases": [
            "কারিকুরি"
        ],
        "verified": True
    },

    {
        "id": "B08",
        "intent": "hardwork",
        "query": "সে সফল হওয়ার জন্য খুব কঠোর পরিশ্রম করছে।",
        "gold_phrase": "আদা জল খেয়ে লাগা",
        "acceptable_phrases": [
            "আদা জল খেয়ে লাগা"
        ],
        "verified": True
    },

    {
        "id": "B09",
        "intent": "poverty",
        "query": "লোকটি খুব দরিদ্র এবং অর্থকষ্টে জীবনযাপন করছে।",
        "gold_phrase": "আকালকেঁড়ে",
        "acceptable_phrases": [
            "আকালকেঁড়ে",
            "অকড়িয়া",
            "উপোসি ছারপোকা"
        ],
        "verified": True
    },

    {
        "id": "B10",
        "intent": "idle_talk",
        "query": "সে সারাদিন অপ্রয়োজনীয় ও অর্থহীন কথাবার্তা বলে।",
        "gold_phrase": "অগড়-বগড় / অগড়ম-বগড়ম",
        "acceptable_phrases": [
            "অগড়-বগড় / অগড়ম-বগড়ম",
            "অগড়-বগড়",
            "অগড়ম-বগড়ম",
            "খেজুরে আলাপ"
        ],
        "verified": True
    },

    {
        "id": "B11",
        "intent": "anxiety",
        "query": "পরিস্থিতির কারণে সে খুব উদ্বিগ্ন ও অস্থির হয়ে পড়েছে।",
        "gold_phrase": "আঁকুপাঁকু করা",
        "acceptable_phrases": [
            "আঁকুপাঁকু করা"
        ],
        "verified": True
    },

    {
        "id": "B12",
        "intent": "pain",
        "query": "ঘটনাটি তাকে খুব কষ্ট দিয়েছে এবং তার মনে গভীর যন্ত্রণা হয়েছে।",
        "gold_phrase": "আঁতে ঘা",
        "acceptable_phrases": [
            "আঁতে ঘা"
        ],
        "verified": True
    }
]


def normalize(text):
    if not text:
        return ""

    return (
        str(text)
        .strip()
        .lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )


def main():

    print("=" * 75)
    print("BENGALI CULTURAL RAG — CLEAN BENCHMARK")
    print("HIGH-CONFIDENCE / SEMANTICALLY RELEVANT")
    print("=" * 75)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\nDataset records: {len(dataset)}")

    dataset_phrases = {
        normalize(x.get("bengali_phrase", ""))
        for x in dataset
        if isinstance(x, dict)
    }

    cleaned = []

    for item in BENCHMARK:

        # Verify gold exists in dataset OR one acceptable phrase exists
        available = False

        for phrase in item["acceptable_phrases"]:

            if normalize(phrase) in dataset_phrases:
                available = True
                break

        if available:
            cleaned.append(item)

    # Remove duplicates
    seen = set()
    final = []

    for item in cleaned:

        key = normalize(item["query"])

        if key not in seen:
            seen.add(key)
            final.append(item)

    for item in final:
        item["verified"] = True

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            final,
            f,
            ensure_ascii=False,
            indent=2
        )

    report = {
        "dataset_records": len(dataset),
        "candidate_records": len(BENCHMARK),
        "final_records": len(final),
        "verified_records": len(final),
        "api_used": False,
        "model_used": "E5 + BM25 + Meaning Grounding",
        "benchmark_type": "high-confidence semantic retrieval",
        "records": final
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 75)
    print("CLEAN BENCHMARK CREATED")
    print("=" * 75)

    print(f"Dataset records : {len(dataset)}")
    print(f"Candidates      : {len(BENCHMARK)}")
    print(f"Final benchmark : {len(final)}")
    print(f"Verified        : {len(final)}")

    print("\nSELECTED QUERIES")

    for i, item in enumerate(final, 1):
        print(f"\n[{i}] {item['intent']}")
        print(f"Query : {item['query']}")
        print(f"Gold  : {item['gold_phrase']}")
        print(
            "Acceptable:",
            ", ".join(item["acceptable_phrases"])
        )

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(REPORT_PATH)

    print("\nNO API / NO GEMINI / NO GROQ")


if __name__ == "__main__":
    main()