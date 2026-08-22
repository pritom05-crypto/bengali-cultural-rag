# -*- coding: utf-8 -*-

"""
BENGALI CULTURAL RAG V6.26
MEANING-GROUNDED DATASET-FIRST RETRIEVAL

NO GEMINI
NO GROQ
NO API KEY

Architecture:
    E5 semantic retrieval
    + BM25 lexical retrieval
    + concept grounding
    + meaning grounding
    + primary-intent detection
    + safe Bengali normalization
    + confidence / margin filtering

Compatible with the existing benchmark scripts:
    retrieve(original_query, corrected_query, concepts, expanded_queries)
    choose_answer(results, concepts)
"""

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_cultural_dataset.json"
)

EMBEDDING_PATH = (
    BASE_DIR
    / "data"
    / "embeddings"
    / "e5_rich_embeddings.npy"
)

MODEL_NAME = "intfloat/multilingual-e5-base"

TOP_K_SEMANTIC = 50
TOP_K_BM25 = 40
TOP_K_FINAL = 10

MIN_CONFIDENCE = 0.30
STRONG_CONFIDENCE = 0.78

# ============================================================
# CONCEPT DICTIONARY
# ============================================================

CONCEPT_TERMS = {

    "অলস": [
        "অলস", "কুঁড়ে", "কুঁড়ে", "কর্মবিমুখ",
        "নিষ্ক্রিয়", "নিষ্ক্রিয়", "কাজে অনীহা",
        "কাজ করতে চায় না", "কাজ করতে চায় না",
        "পরিশ্রম বিমুখ", "ঢিলেমি", "উদ্যমহীন",
        "কাজকর্মে অনীহা", "lazy", "idle",
        "inactive", "terribly lazy",
    ],

    "রাগ": [
        "রাগ", "রেগে", "রাগান্বিত", "ক্রোধ",
        "ক্ষোভ", "উত্তেজিত", "অগ্নিশর্মা",
        "ভীষণ রাগ", "প্রচণ্ড রাগ", "angry",
        "anger", "furious", "fury", "rage",
        "extremely angry",
    ],

    "বিপদ": [
        "বিপদ", "বিপদে", "বিপাকে", "বিপাকে পড়া",
        "বিপাকে পড়া", "সংকট", "সমস্যা", "আতান্তর",
        "ঝুঁকি", "দুর্দশা", "danger", "trouble",
        "crisis", "risk", "distress", "adversity",
        "disaster", "great danger",
    ],

    "ভয়": [
        "ভয়", "ভয়", "ভীত", "ভয় পাওয়া", "ভয় পাওয়া",
        "ভয় পেয়েছে", "ভয় পেয়েছে", "আতঙ্ক",
        "আতঙ্কিত", "শঙ্কা", "ভয়ে", "ভয়ে",
        "fear", "afraid", "frightened", "scared",
        "terrified", "panic", "anxiety",
    ],

    "বিভ্রান্তি": [
        "বিভ্রান্ত", "বিভ্রান্তি", "হতবুদ্ধি",
        "কী করবে বুঝতে পারছে না",
        "কি করবে বুঝতে পারছে না",
        "বুঝতে পারছে না", "confused",
        "confusion", "bewildered", "puzzled",
    ],

    "বিস্ময়": [
        "বিস্ময়", "বিস্ময়", "বিস্মিত", "অবাক",
        "আশ্চর্য", "আশ্চর্যজনক", "অপ্রত্যাশিত",
        "unexpected", "surprised", "surprise",
        "astonished", "amazed",
    ],

    "দুঃখ": [
        "দুঃখ", "দুঃখিত", "দুঃখ পেয়েছে", "দুঃখ পেয়েছে",
        "কষ্ট", "মন খারাপ", "বেদনাহত", "sad",
        "sadness", "sorrow", "grief", "deep sadness",
    ],

    "কষ্ট": [
        "কষ্ট", "কষ্ট দিয়েছে", "কষ্ট দিয়েছে",
        "যন্ত্রণা", "ব্যথা", "মনে কষ্ট",
        "গভীর যন্ত্রণা", "pain", "hurt", "suffering",
        "sadness", "deep emotional pain",
    ],

    "সাফল্য": [
        "সাফল্য", "সফল", "সফল হয়েছে", "সফল হয়েছে",
        "লক্ষ্য অর্জন", "লক্ষ্য অর্জন করেছে",
        "জয়", "জয়", "জিতেছে", "জয়ী",
        "success", "successful", "win", "achievement",
        "achieve success",
    ],

    "ব্যর্থতা": [
        "ব্যর্থ", "ব্যর্থতা", "ব্যর্থ হয়েছে", "ব্যর্থ হয়েছে",
        "সফল হয়নি", "সফল হয়নি", "fail", "failure",
        "unsuccessful", "failed",
    ],

    "পরিশ্রম": [
        "পরিশ্রম", "কঠোর পরিশ্রম", "খুব কঠোর পরিশ্রম",
        "পরিশ্রম করছে", "পরিশ্রম করা", "খাটছে",
        "খাটুনি", "চেষ্টা করছে", "চেষ্টা করা",
        "সর্বোচ্চ চেষ্টা", "প্রচেষ্টা", "উদ্যম",
        "hard work", "hardwork", "effort",
        "utmost effort", "great effort",
        "trying desperately",
    ],

    "গড়িমসি": [
        "গড়িমসি", "গড়িমসি", "পিছিয়ে দিচ্ছে",
        "পিছিয়ে দিচ্ছে", "সময় নষ্ট করছে",
        "সময় নষ্ট করছে", "দেরি করছে", "procrastination",
        "delay", "delaying",
    ],

    "প্রতারণা": [
        "প্রতারণা", "প্রতারণা করেছে", "ঠকিয়েছে",
        "ঠকিয়েছে", "ঠকানো", "ধোঁকা", "ধোঁকা দিয়েছে",
        "ধোঁকা দিয়েছে", "ছলনা", "কারচুপি", "কৌশলে ঠক",
        "trickery", "deception", "deceive", "fraud",
        "cheat", "cheating",
    ],

    "সমালোচনা": [
        "সমালোচনা", "সমালোচনা করছে", "সমালোচনা করা",
        "দোষ ধরছে", "criticize", "criticism",
    ],

    "একগুঁয়েমি": [
        "একগুঁয়ে", "একগুঁয়েমি", "জেদি",
        "নিজের অবস্থান থেকে সরছে না", "সরছে না",
        "অনেক বোঝালেও", "stubborn", "stubbornness",
        "unyielding",
    ],

    "নির্ভরশীলতা": [
        "নির্ভরশীল", "নির্ভর করে", "অন্যের ওপর নির্ভর",
        "অন্যের সাহায্যের ওপর", "পরমুখাপেক্ষী",
        "dependent", "dependency", "reliant",
    ],

    "দারিদ্র্য": [
        "দরিদ্র", "দারিদ্র্য", "অর্থকষ্ট", "গরিব",
        "অভাব", "অর্থের অভাব", "অর্থহীন",
        "poverty", "poor", "moneyless", "destitute",
    ],

    "অর্থহীন_কথাবার্তা": [
        "অর্থহীন কথাবার্তা", "অপ্রয়োজনীয় কথাবার্তা",
        "অপ্রয়োজনীয় কথাবার্তা", "বাজে কথা", "ফালতু কথা",
        "বকবক", "আজেবাজে কথা", "meaningless talk",
        "idle talk", "useless talk", "nonsense",
        "rambling", "gibberish",
    ],

    "উদ্বেগ": [
        "উদ্বিগ্ন", "উদ্বেগ", "অস্থির", "অস্থির হয়ে",
        "অস্থির হয়ে", "দুশ্চিন্তা", "restlessness",
        "anxiety", "anxious", "worried",
    ],

    "অহংকার": [
        "অহংকার", "অহংকার করে", "দর্প", "গর্ব",
        "নিজের ক্ষমতা নিয়ে", "নিজের ক্ষমতা নিয়ে",
        "নিজের কৃতিত্ব নিয়ে", "নিজের কৃতিত্ব নিয়ে",
        "boasting", "pride", "arrogance",
    ],
}


# ============================================================
# DIRECT / RELATED MEANING PROFILES
# ============================================================

MEANING_PROFILES = {

    "অলস": {
        "direct": [
            "অলস", "lazy", "terribly lazy", "inactive",
            "কাজকর্ম করতে মোটেই ইচ্ছুক নয়",
            "কাজকর্মে অনীহা", "very lazy",
        ],
        "related": ["idle", "useless"],
    },

    "রাগ": {
        "direct": [
            "রাগান্বিত", "অত্যন্ত রাগ", "ভীষণ রাগ",
            "extremely angry", "angry", "furious",
            "fury", "rage",
        ],
        "related": [" উত্তেজিত", "anger"],
    },

    "বিপদ": {
        "direct": [
            "বিপদ", "বিপদে", "বিপাকে", "সংকট",
            "danger", "trouble", "crisis", "risk",
            "distress", "adversity", "disaster",
            "great danger", "falling into trouble",
            "extreme trouble", "severe adversity",
        ],
        "related": ["helpless", "desperate", "fearful"],
    },

    "ভয়": {
        "direct": [
            "ভয়", "ভয়", "ভীত", "আতঙ্ক", "আতঙ্কিত",
            "fear", "afraid", "frightened", "scared",
            "terrified", "panic",
        ],
        "related": ["anxiety", "danger", "distress"],
    },

    "বিভ্রান্তি": {
        "direct": [
            "confused", "confusion", "bewildered",
            "puzzled", "বিভ্রান্ত", "হতবুদ্ধি",
        ],
        "related": ["uncertain", "doubt"],
    },

    "বিস্ময়": {
        "direct": [
            "unexpected", "surprised", "surprise",
            "astonished", "amazed", "অপ্রত্যাশিত",
            "অবাক", "বিস্মিত", "আশ্চর্য",
        ],
        "related": ["shock", "shocked"],
    },

    "দুঃখ": {
        "direct": [
            "দুঃখ", "sad", "sadness", "sorrow",
            "grief", "deep sadness",
        ],
        "related": ["pain", "hurt"],
    },

    "কষ্ট": {
        "direct": [
            "কষ্ট", "ব্যথা", "যন্ত্রণা",
            "to hurt someone's feelings deeply",
            "hurt", "pain", "suffering",
            "deep emotional pain", "sharp and painful",
            "secret pain", "heartbreaking",
        ],
        "related": ["sadness", "sorrow"],
    },

    "সাফল্য": {
        "direct": [
            "সাফল্য", "সফল", "জয়", "জয়", "জিতেছে",
            "success", "successful", "win", "achievement",
            "achieve success", "to win or achieve success",
        ],
        "related": ["victory", "goal"],
    },

    "ব্যর্থতা": {
        "direct": [
            "ব্যর্থ", "ব্যর্থতা", "failure", "failed",
            "unsuccessful",
        ],
        "related": ["futile", "reluctant"],
    },

    "পরিশ্রম": {
        "direct": [
            "পরিশ্রম", "কঠোর পরিশ্রম", "খাটুনি",
            "চেষ্টা", "প্রচেষ্টা", "hard work", "hardwork",
            "effort", "utmost effort", "great effort",
            "trying desperately", "continuous hard work",
            "special effort",
        ],
        "related": ["achievement", "success"],
    },

    "গড়িমসি": {
        "direct": [
            "গড়িমসি", "গড়িমসি", "procrastination",
            "delay", "delaying", "wasting time",
        ],
        "related": ["late", "slow"],
    },

    "প্রতারণা": {
        "direct": [
            "প্রতারণা", "ঠকানো", "ধোঁকা", "ছলনা",
            "trickery", "deception", "deceive",
            "fraud", "cheat", "cheating",
            "to deceive, cheat, or evade duty",
        ],
        "related": ["cunning", "trick"],
    },

    "সমালোচনা": {
        "direct": [
            "সমালোচনা", "criticize", "criticism",
        ],
        "related": ["protest", "disapprove"],
    },

    "একগুঁয়েমি": {
        "direct": [
            "একগুঁয়ে", "একগুঁয়েমি", "জেদি",
            "stubborn", "stubbornness", "unyielding",
        ],
        "related": ["determined"],
    },

    "নির্ভরশীলতা": {
        "direct": [
            "নির্ভরশীল", "নির্ভর", "পরমুখাপেক্ষী",
            "dependent", "dependency", "reliant",
            "to have no other way",
        ],
        "related": ["help", "support"],
    },

    "দারিদ্র্য": {
        "direct": [
            "দরিদ্র", "দারিদ্র্য", "গরিব", "অর্থকষ্ট",
            "poverty", "poor", "moneyless", "destitute",
            "poor person", "poor / beggar", "extremely poor",
        ],
        "related": ["lowly person"],
    },

    "অর্থহীন_কথাবার্তা": {
        "direct": [
            "অর্থহীন", "অপ্রয়োজনীয়", "অপ্রয়োজনীয়",
            "বাজে কথা", "meaningless", "idle talk",
            "useless or idle talk", "nonsense",
            "rambling", "irrelevant talk", "gibberish",
            "nonsense / unnecessary talk",
        ],
        "related": ["useless"],
    },

    "উদ্বেগ": {
        "direct": [
            "উদ্বেগ", "উদ্বিগ্ন", "অস্থির",
            "restlessness", "anxiety", "anxious",
            "worried",
        ],
        "related": ["confused", "distress"],
    },

    "অহংকার": {
        "direct": [
            "অহংকার", "দর্প", "গর্ব",
            "boasting", "pride", "arrogance",
            "the fall of pride",
        ],
        "related": ["confidence"],
    },
}


# ============================================================
# SAFE PRIMARY INTENT PATTERNS
# ============================================================

PRIMARY_PATTERNS = {

    "পরিশ্রম": [
        "কঠোর পরিশ্রম",
        "খুব কঠোর পরিশ্রম",
        "পরিশ্রম করছে",
        "পরিশ্রম করা",
        "খাটছে",
        "চেষ্টা করছে",
        "সর্বোচ্চ চেষ্টা",
        "প্রচেষ্টা",
        "উদ্যম",
        "hard work",
        "hardwork",
    ],

    "সাফল্য": [
        "সফল হয়েছে",
        "সফল হয়েছে",
        "লক্ষ্য অর্জন করেছে",
        "জয় করেছে",
        "জয় করেছে",
        "জিতেছে",
        "সফল হয়েছে",
        "success",
    ],

    "প্রতারণা": [
        "কৌশলে অন্যদের ঠকিয়েছে",
        "কৌশলে অন্যদের ঠকিয়েছে",
        "অন্যদের ঠকিয়েছে",
        "অন্যদের ঠকিয়েছে",
        "ঠকিয়েছে",
        "ঠকিয়েছে",
        "প্রতারণা করেছে",
        "ধোঁকা দিয়েছে",
        "ধোঁকা দিয়েছে",
    ],

    "গড়িমসি": [
        "বারবার পিছিয়ে দিচ্ছে",
        "বারবার পিছিয়ে দিচ্ছে",
        "সময় নষ্ট করছে",
        "সময় নষ্ট করছে",
        "কাজটি পিছিয়ে",
        "কাজটি পিছিয়ে",
    ],

    "দারিদ্র্য": [
        "খুব দরিদ্র",
        "অর্থকষ্ট",
        "খুব গরিব",
        "গরিব",
        "অর্থের অভাব",
    ],

    "অর্থহীন_কথাবার্তা": [
        "অর্থহীন কথাবার্তা",
        "অপ্রয়োজনীয় কথাবার্তা",
        "অপ্রয়োজনীয় কথাবার্তা",
        "বাজে কথা",
        "ফালতু কথা",
        "আজেবাজে কথা",
    ],

    "উদ্বেগ": [
        "উদ্বিগ্ন",
        "অস্থির হয়ে",
        "অস্থির হয়ে",
        "দুশ্চিন্তা",
    ],

    "কষ্ট": [
        "খুব কষ্ট দিয়েছে",
        "খুব কষ্ট দিয়েছে",
        "গভীর যন্ত্রণা",
        "মনে গভীর যন্ত্রণা",
    ],

    "বিভ্রান্তি": [
        "কী করবে বুঝতে পারছে না",
        "কি করবে বুঝতে পারছে না",
        "বুঝতে পারছে না",
        "হতবুদ্ধি",
        "বিভ্রান্ত হয়ে",
        "বিভ্রান্ত হয়ে",
    ],

    "রাগ": [
        "খুব রেগে গেছে",
        "প্রচণ্ড রাগান্বিত",
        "প্রচণ্ড রাগ",
        "রাগান্বিত",
        "furious",
    ],

    "বিপদ": [
        "বিপদে পড়েছে",
        "বিপদে পড়েছে",
        "বের হওয়ার কোনো পথ",
        "বের হওয়ার কোনো পথ",
        "সংকট থেকে বের",
    ],

    "অলস": [
        "খুব অলস",
        "অলস",
        "কাজ করতে চায় না",
        "কাজ করতে চায় না",
        "কাজকর্মে অনীহা",
    ],

    "একগুঁয়েমি": [
        "নিজের অবস্থান থেকে সরছে না",
        "নিজের অবস্থান থেকে সরছে না",
        "অনেক বোঝালেও",
        "সরছে না",
    ],

    "অহংকার": [
        "অহংকার করে",
        "নিজের ক্ষমতা নিয়ে",
        "নিজের ক্ষমতা নিয়ে",
        "নিজের কৃতিত্ব নিয়ে",
        "নিজের কৃতিত্ব নিয়ে",
    ],
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_bengali(text):
    if text is None:
        return ""

    text = str(text).lower().strip()

    replacements = {
        "ড়": "ড়",
        "ঢ়": "ঢ়",
        "য়": "য়",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "’": "'",
        "\u200c": "",
        "\u200d": "",
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    return re.findall(
        r"[\u0980-\u09FF]+|[a-zA-Z]+",
        normalize_bengali(text)
    )


# ============================================================
# SAFE QUERY CORRECTION
# ============================================================

def correct_query(query):
    """
    IMPORTANT:
    Never perform substring replacement on Bengali verbs.
    That caused:
        করছে -> করেছেে
        গেছে -> গেছেএ
        পড়েছি -> পড়েছিি

    Only exact whole-token corrections are allowed.
    """

    if query is None:
        return ""

    text = str(query).strip()

    safe = {
        "পেয়েছ": "পেয়েছে",
        "পেয়েছ": "পেয়েছে",
        "গেছ": "গেছে",
        "হইছে": "হয়েছে",
        "হয়েছ": "হয়েছে",
        "হয়েছ": "হয়েছে",
    }

    for wrong, right in safe.items():
        text = re.sub(
            rf"(?<![\u0980-\u09FF]){re.escape(wrong)}(?![\u0980-\u09FF])",
            right,
            text
        )

    return text


# ============================================================
# DATASET HELPERS
# ============================================================

def _first_nonempty(record, keys):
    for key in keys:
        value = record.get(key)

        if value is None:
            continue

        if isinstance(value, (list, dict)):
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


def get_record_phrase(record):
    """
    Robust phrase extraction.

    The dataset primarily uses bengali_phrase, but this function
    supports the field names used across the project's earlier
    dataset versions.
    """

    preferred = [
        "bengali_phrase",
        "phrase",
        "expression",
        "idiom",
        "idiom_phrase",
        "idiom_bn",
        "expression_bn",
        "phrase_bn",
        "cultural_expression",
        "cultural_phrase",
        "headword",
        "title",
        "name",
        "term",
        "word",
        "entry",
        "lemma",
    ]

    phrase = _first_nonempty(record, preferred)

    if phrase:
        return phrase

    # Fallback: detect phrase-like keys automatically.
    for key, value in record.items():

        key_n = str(key).lower()

        if not any(
            marker in key_n
            for marker in [
                "phrase",
                "expression",
                "idiom",
                "headword",
                "lemma",
                "term",
            ]
        ):
            continue

        if value is None or isinstance(value, (list, dict)):
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


def get_record_meaning(record):

    fields = [
        "literal_meaning",
        "cultural_meaning",
        "meaning",
    ]

    values = []

    for field in fields:

        value = record.get(field)

        if value is None:
            continue

        if isinstance(value, list):
            values.extend(
                str(x).strip()
                for x in value
                if str(x).strip()
            )
        else:
            value = str(value).strip()

            if value:
                values.append(value)

    return " ".join(dict.fromkeys(values))


def get_display_meaning(record):

    value = record.get("cultural_meaning")

    if value:
        return str(value).strip()

    value = record.get("meaning")

    if value:
        return str(value).strip()

    value = record.get("literal_meaning")

    if value:
        return str(value).strip()

    return ""


def get_record_tone(record):

    fields = [
        "intended_emotion_tone",
        "primary_tone",
        "secondary_tone",
        "tone",
        "raw_tones",
    ]

    values = []

    for field in fields:

        value = record.get(field)

        if value is None:
            continue

        if isinstance(value, list):
            values.extend(
                str(x).strip()
                for x in value
                if str(x).strip()
            )
        else:
            value = str(value).strip()

            if value:
                values.append(value)

    return " / ".join(dict.fromkeys(values))


def get_display_tone(record):

    value = record.get("intended_emotion_tone")

    if value:
        return str(value).strip()

    primary = record.get("primary_tone")
    secondary = record.get("secondary_tone")

    values = [
        str(x).strip()
        for x in [primary, secondary]
        if x
    ]

    return " / ".join(dict.fromkeys(values))


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("BENGALI CULTURAL RAG V6.26")
print("MEANING-GROUNDED DATASET-FIRST RETRIEVAL")
print("NO GEMINI | NO GROQ | NO API KEY")
print("=" * 70)

print("\nLoading final dataset...")

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:
    dataset = json.load(f)

if not isinstance(dataset, list):
    raise RuntimeError(
        "final_cultural_dataset.json must contain a JSON list."
    )

print(
    f"Dataset records: {len(dataset)}"
)


# ============================================================
# LOAD E5 EMBEDDINGS
# ============================================================

print("\nLoading E5 embeddings...")

embeddings = np.load(
    EMBEDDING_PATH
)

if len(embeddings) != len(dataset):
    raise RuntimeError(
        f"Embedding count ({len(embeddings)}) does not match "
        f"dataset count ({len(dataset)})."
    )

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)

norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

norms[norms == 0] = 1.0

embeddings = embeddings / norms

print(
    f"Embedding matrix: {embeddings.shape}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded.")


# ============================================================
# BM25 TEXT
# ============================================================

print("\nBuilding BM25 index...")


def build_bm25_text(record):

    phrase = get_record_phrase(record)
    meaning = get_record_meaning(record)

    variants = record.get(
        "variants",
        []
    )

    if isinstance(variants, list):
        variants_text = " ".join(
            str(x)
            for x in variants
            if x
        )
    else:
        variants_text = str(variants)

    # Tone deliberately excluded.
    text = " ".join([
        phrase,
        variants_text,
        meaning,
    ])

    return normalize_bengali(text)


bm25_documents = [
    tokenize(build_bm25_text(record))
    for record in dataset
]

bm25 = BM25Okapi(
    bm25_documents
)

print("BM25 index ready.")


# ============================================================
# QUERY ENCODING
# ============================================================

def encode_query(query):

    text = "query: " + normalize_bengali(query)

    vector = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    return vector


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    if len(scores) == 0:
        return scores

    minimum = float(scores.min())
    maximum = float(scores.max())

    if maximum - minimum < 1e-9:

        if maximum > 0:
            return np.ones_like(scores)

        return np.zeros_like(scores)

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


# ============================================================
# CONCEPT DETECTION
# ============================================================

def detect_concepts(query):

    q = normalize_bengali(query)

    found = []

    # Longer phrases first to avoid weak partial matches.
    for concept, terms in CONCEPT_TERMS.items():

        ordered_terms = sorted(
            terms,
            key=lambda x: len(normalize_bengali(x)),
            reverse=True
        )

        for term in ordered_terms:

            term_n = normalize_bengali(term)

            if term_n and term_n in q:
                found.append(concept)
                break

    return list(dict.fromkeys(found))


# ============================================================
# PRIMARY INTENT
# ============================================================

def detect_primary_concept(query, concepts):

    q = normalize_bengali(query)

    matches = []

    for concept in concepts:

        patterns = PRIMARY_PATTERNS.get(
            concept,
            []
        )

        for pattern in patterns:

            pattern_n = normalize_bengali(pattern)

            if pattern_n and pattern_n in q:
                matches.append(
                    (
                        concept,
                        len(pattern_n)
                    )
                )

    if matches:

        # Specific/longer phrase wins.
        matches.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return matches[0][0]

    # If no explicit sentence pattern is found,
    # use the first detected concept.
    return concepts[0] if concepts else None


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query, concepts):

    queries = [query]

    for concept in concepts:

        for term in CONCEPT_TERMS.get(
            concept,
            []
        ):

            term = str(term).strip()

            if term and term not in queries:
                queries.append(term)

    return queries[:30]


# ============================================================
# BM25 QUERY
# ============================================================

def build_bm25_query(query, concepts=None):

    parts = [query]

    if concepts:

        for concept in concepts:

            parts.extend(
                CONCEPT_TERMS.get(
                    concept,
                    []
                )
            )

    return tokenize(
        " ".join(parts)
    )


# ============================================================
# MEANING SCORE
# ============================================================

def meaning_concept_score(record, concepts):

    if not concepts:
        return 0.0

    meaning = normalize_bengali(
        get_record_meaning(record)
    )

    if not meaning:
        return 0.0

    best = 0.0

    for concept in concepts:

        profile = MEANING_PROFILES.get(
            concept,
            {}
        )

        direct = profile.get(
            "direct",
            []
        )

        related = profile.get(
            "related",
            []
        )

        direct_hits = 0
        related_hits = 0

        for term in direct:

            term_n = normalize_bengali(term)

            if term_n and term_n in meaning:
                direct_hits += 1

        for term in related:

            term_n = normalize_bengali(term)

            if term_n and term_n in meaning:
                related_hits += 1

        if direct_hits:

            score = min(
                1.0,
                0.82
                + 0.06 * min(
                    direct_hits - 1,
                    3
                )
            )

            best = max(
                best,
                score
            )

        elif related_hits:

            best = max(
                best,
                0.35
            )

    return float(best)


# ============================================================
# CONCEPT SCORE
# ============================================================

def concept_score(record, concepts):

    if not concepts:
        return 0.0

    phrase = normalize_bengali(
        get_record_phrase(record)
    )

    meaning = normalize_bengali(
        get_record_meaning(record)
    )

    best = 0.0

    for concept in concepts:

        for term in CONCEPT_TERMS.get(
            concept,
            []
        ):

            term_n = normalize_bengali(term)

            if not term_n:
                continue

            if term_n in phrase:
                best = max(best, 1.0)

            elif term_n in meaning:
                best = max(best, 0.95)

    return float(best)


# ============================================================
# LEXICAL SCORE
# ============================================================

def lexical_score(record, query):

    q_tokens = set(
        tokenize(query)
    )

    if not q_tokens:
        return 0.0

    text = " ".join([
        get_record_phrase(record),
        get_record_meaning(record),
        " ".join(
            str(x)
            for x in record.get("variants", [])
        )
        if isinstance(record.get("variants", []), list)
        else str(record.get("variants", "")),
    ])

    text_tokens = set(
        tokenize(text)
    )

    if not text_tokens:
        return 0.0

    overlap = q_tokens & text_tokens

    return float(
        len(overlap) / max(len(q_tokens), 1)
    )


# ============================================================
# PHRASE PATTERN BONUS
# ============================================================

def phrase_pattern_bonus(record, query):

    phrase = normalize_bengali(
        get_record_phrase(record)
    )

    q = normalize_bengali(query)

    if not phrase or not q:
        return 0.0

    bonus = 0.0

    # Exact phrase occurrence.
    if phrase in q:
        bonus += 0.12

    # Variant occurrence.
    variants = record.get(
        "variants",
        []
    )

    if isinstance(variants, list):

        for variant in variants:

            variant_n = normalize_bengali(
                variant
            )

            if variant_n and variant_n in q:
                bonus = max(
                    bonus,
                    0.10
                )

    return min(
        bonus,
        0.12
    )


# ============================================================
# INTENT BONUS
# ============================================================

def intent_bonus(record, concepts):

    if not concepts:
        return 0.0

    phrase = normalize_bengali(
        get_record_phrase(record)
    )

    meaning = normalize_bengali(
        get_record_meaning(record)
    )

    text = phrase + " " + meaning

    bonus = 0.0

    for concept in concepts:

        profile = MEANING_PROFILES.get(
            concept,
            {}
        )

        direct = profile.get(
            "direct",
            []
        )

        for term in direct:

            term_n = normalize_bengali(term)

            if term_n and term_n in text:

                bonus = max(
                    bonus,
                    0.04
                )

    return bonus


# ============================================================
# PENALTIES
# ============================================================

def mismatch_penalty(
    record,
    concepts
):

    if not concepts:
        return 0.0

    meaning_score = meaning_concept_score(
        record,
        concepts
    )

    semantic_text = normalize_bengali(
        get_record_phrase(record)
        + " "
        + get_record_meaning(record)
    )

    # Known concept + no meaning evidence is dangerous.
    if meaning_score == 0.0 and semantic_text:
        return 0.10

    # Related-only evidence gets a smaller penalty.
    if 0.0 < meaning_score < 0.80:
        return 0.04

    return 0.0


def severity_mismatch_penalty(
    record,
    concepts
):

    # Keep this conservative. We do not reject candidates
    # solely because tone differs.
    return 0.0


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    original_query,
    corrected_query,
    concepts,
    expanded_queries
):

    # --------------------------------------------------------
    # Original query semantic evidence
    # --------------------------------------------------------

    original_vector = encode_query(
        original_query
    )

    original_cosine = np.dot(
        embeddings,
        original_vector
    )

    original_semantic = (
        original_cosine + 1.0
    ) / 2.0

    # --------------------------------------------------------
    # Expanded semantic evidence
    # --------------------------------------------------------

    expanded_vectors = []

    for q in expanded_queries[1:]:

        try:

            vec = encode_query(q)
            expanded_vectors.append(vec)

        except Exception:
            continue

    if expanded_vectors:

        expanded_matrix = np.vstack(
            expanded_vectors
        )

        expanded_cosine = np.max(
            np.dot(
                embeddings,
                expanded_matrix.T
            ),
            axis=1
        )

        expanded_semantic = (
            expanded_cosine + 1.0
        ) / 2.0

    else:

        expanded_semantic = np.zeros(
            len(dataset),
            dtype=np.float32
        )

    # Original query dominates.
    semantic = (
        0.80 * original_semantic
        + 0.20 * expanded_semantic
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_query = build_bm25_query(
        corrected_query,
        concepts
    )

    raw_bm25 = bm25.get_scores(
        bm25_query
    )

    bm25_scores = normalize_scores(
        raw_bm25
    )

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    candidate_indices = set()

    semantic_top = np.argsort(
        semantic
    )[::-1][:TOP_K_SEMANTIC]

    bm25_top = np.argsort(
        bm25_scores
    )[::-1][:TOP_K_BM25]

    candidate_indices.update(
        int(x)
        for x in semantic_top
    )

    candidate_indices.update(
        int(x)
        for x in bm25_top
    )

    # Add records with direct meaning evidence.
    if concepts:

        for idx, record in enumerate(dataset):

            if meaning_concept_score(
                record,
                concepts
            ) >= 0.82:

                candidate_indices.add(
                    idx
                )

    primary_concept = detect_primary_concept(
        corrected_query,
        concepts
    )

    # Add direct primary-intent records too.
    if primary_concept:

        for idx, record in enumerate(dataset):

            if meaning_concept_score(
                record,
                [primary_concept]
            ) >= 0.82:

                candidate_indices.add(
                    idx
                )

    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    results = []

    for idx in candidate_indices:

        record = dataset[idx]

        phrase = get_record_phrase(
            record
        )

        meaning = get_record_meaning(
            record
        )

        sem = float(
            semantic[idx]
        )

        bm = float(
            bm25_scores[idx]
        )

        lexical = lexical_score(
            record,
            corrected_query
        )

        concept = concept_score(
            record,
            concepts
        )

        meaning_score = meaning_concept_score(
            record,
            concepts
        )

        primary_meaning = 0.0

        if primary_concept:

            primary_meaning = (
                meaning_concept_score(
                    record,
                    [primary_concept]
                )
            )

        bonus = intent_bonus(
            record,
            concepts
        )

        pattern_bonus = phrase_pattern_bonus(
            record,
            corrected_query
        )

        penalty = mismatch_penalty(
            record,
            concepts
        )

        severity_penalty = severity_mismatch_penalty(
            record,
            concepts
        )

        # ----------------------------------------------------
        # PRIMARY INTENT WEIGHTING
        # ----------------------------------------------------

        if primary_concept:

            final_score = (

                0.25 * sem
                + 0.15 * bm
                + 0.06 * lexical
                + 0.10 * concept
                + 0.12 * meaning_score
                + 0.27 * primary_meaning
                + 0.05 * pattern_bonus
                + bonus
                - penalty
                - severity_penalty
            )

        else:

            final_score = (

                0.40 * sem
                + 0.20 * bm
                + 0.08 * lexical
                + 0.10 * concept
                + 0.17 * meaning_score
                + 0.05 * pattern_bonus
                + bonus
                - penalty
                - severity_penalty
            )

        # Strong direct meaning evidence.
        if primary_meaning >= 0.90:
            final_score += 0.10

        elif primary_meaning >= 0.82:
            final_score += 0.06

        elif meaning_score >= 0.90:
            final_score += 0.04

        # Very high semantic evidence gets only a small boost.
        if sem >= 0.92:
            final_score += 0.02

        final_score = float(
            np.clip(
                final_score,
                0.0,
                1.0
            )
        )

        results.append({

            "index": int(idx),

            "record": record,

            "phrase": phrase,

            "meaning": meaning,

            "display_meaning":
                get_display_meaning(record),

            "tone":
                get_display_tone(record),

            "semantic": sem,

            "bm25": bm,

            "lexical": lexical,

            "concept": concept,

            "meaning_score":
                meaning_score,

            "meaning_similarity":
                sem,

            "meaning_concept":
                meaning_score,

            "primary_concept":
                primary_concept,

            "primary_meaning":
                primary_meaning,

            "bonus": bonus,

            "pattern_bonus":
                pattern_bonus,

            "severity_penalty":
                severity_penalty,

            "penalty":
                penalty,

            "final":
                final_score,
        })

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["final"],
        reverse=True
    )

    # --------------------------------------------------------
    # Keep reasonable candidates.
    # Do NOT apply an aggressive 0.30 filter before
    # benchmark inspection.
    # --------------------------------------------------------

    return results[:TOP_K_FINAL]


# ============================================================
# FINAL ANSWER SELECTION
# ============================================================

def choose_answer(
    results,
    concepts
):

    if not results:
        return None

    best = results[0]

    # No concept = rely more on semantic evidence.
    if not concepts:

        if best["final"] < MIN_CONFIDENCE:
            return None

        return best

    primary = best.get(
        "primary_concept"
    )

    # --------------------------------------------------------
    # If a direct primary-intent match exists, prefer it.
    # --------------------------------------------------------

    if primary:

        primary_meaning = best.get(
            "primary_meaning",
            0.0
        )

        if primary_meaning >= 0.82:

            return best

    # --------------------------------------------------------
    # Known intent needs direct meaning evidence OR
    # a very strong semantic + BM25 combination.
    # --------------------------------------------------------

    if best["meaning_score"] < 0.80:

        strong_retrieval = (
            best["semantic"] >= 0.91
            and best["bm25"] >= 0.75
        )

        if not strong_retrieval:
            return None

    if best["final"] < MIN_CONFIDENCE:
        return None

    # --------------------------------------------------------
    # Margin check
    # --------------------------------------------------------

    if len(results) >= 2:

        second = results[1]

        margin = (
            best["final"]
            - second["final"]
        )

        # Do not reject a direct high-confidence match.
        if (
            margin < 0.025
            and best["final"] < STRONG_CONFIDENCE
            and best.get("primary_meaning", 0.0) < 0.82
        ):
            return None

    return best


# ============================================================
# RESULT DISPLAY
# ============================================================

def print_results(results):

    print()
    print("=" * 70)
    print("LOCAL DATASET RETRIEVAL RESULTS")
    print("=" * 70)

    if not results:

        print("\nNO RELIABLE DATASET CANDIDATE")
        return

    for i, r in enumerate(
        results,
        start=1
    ):

        phrase = r.get(
            "phrase",
            ""
        )

        if not phrase:
            phrase = "[UNKNOWN PHRASE]"

        print()
        print(
            f"[{i}] {phrase}"
        )

        print(
            f"Semantic           : "
            f"{r['semantic']:.4f}"
        )

        print(
            f"Meaning Similarity : "
            f"{r.get('meaning_similarity', r['semantic']):.4f}"
        )

        print(
            f"BM25               : "
            f"{r['bm25']:.4f}"
        )

        print(
            f"Lexical            : "
            f"{r['lexical']:.4f}"
        )

        print(
            f"Concept            : "
            f"{r['concept']:.4f}"
        )

        print(
            f"Meaning Concept    : "
            f"{r['meaning_score']:.4f}"
        )

        print(
            f"Primary Concept    : "
            f"{r.get('primary_concept') or 'None'}"
        )

        print(
            f"Primary Meaning    : "
            f"{r.get('primary_meaning', 0.0):.4f}"
        )

        print(
            f"Bonus              : "
            f"{r['bonus']:.4f}"
        )

        print(
            f"Pattern Bonus      : "
            f"{r.get('pattern_bonus', 0.0):.4f}"
        )

        print(
            f"Penalty            : "
            f"{r['penalty']:.4f}"
        )

        print(
            f"FINAL              : "
            f"{r['final']:.4f}"
        )

        print(
            f"Meaning            : "
            f"{r['display_meaning']}"
        )

        print(
            f"Tone               : "
            f"{r['tone']}"
        )


# ============================================================
# FINAL ANSWER DISPLAY
# ============================================================

def print_answer(answer):

    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    if answer is None:

        print()
        print(
            "বিশ্বস্ত কোনো বাংলা অভিব্যক্তি পাওয়া যায়নি।"
        )

        print()
        print(
            "ডেটাসেটে এই অর্থের সঙ্গে সরাসরি মিলে "
            "এমন নির্ভরযোগ্য evidence পাওয়া যায়নি।"
        )

        print()
        print(
            "ভুল অভিব্যক্তি অনুমান করে দেখানো হয়নি।"
        )

        return

    phrase = answer.get(
        "phrase",
        ""
    ).strip()

    if not phrase:

        phrase = "[Dataset phrase unavailable]"

    print()
    print(
        "বাংলা অভিব্যক্তি:"
    )

    print(
        phrase
    )

    print()
    print(
        "অর্থ:"
    )

    print(
        answer.get(
            "display_meaning",
            ""
        )
    )

    print()
    print(
        "ভাব/প্রয়োগের ধরন:"
    )

    print(
        answer.get(
            "tone",
            ""
        )
    )

    print()
    print(
        "Source:"
    )

    print(
        "local dataset + E5/BM25 + "
        "meaning semantic matching + "
        "concept grounding"
    )

    print()
    print(
        f"Confidence: "
        f"{answer['final']:.2f}"
    )


# ============================================================
# INTERACTIVE MODE
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V6.26 INTERACTIVE MODE")
    print("=" * 70)

    print(
        f"Dataset records: {len(dataset)}"
    )

    print(
        "LLM/API: NONE"
    )

    print(
        "Retriever: E5 + BM25 + "
        "Concept + Meaning Grounding + "
        "Primary Intent"
    )

    print()
    print(
        "Type 'exit' to quit."
    )

    while True:

        try:

            user_query = input(
                "\nআপনার প্রশ্ন লিখুন:\n> "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print()
            break

        if user_query.lower() == "exit":
            break

        if not user_query:
            continue

        try:

            print()
            print("=" * 70)
            print("BENGALI CULTURAL RAG V6.26")
            print("=" * 70)

            print()
            print(
                f"User Query: {user_query}"
            )

            # ----------------------------------------------------
            # Correction
            # ----------------------------------------------------

            corrected = correct_query(
                user_query
            )

            if corrected != user_query:

                print(
                    f"Corrected Query: {corrected}"
                )

            # ----------------------------------------------------
            # Concepts
            # ----------------------------------------------------

            concepts = detect_concepts(
                corrected
            )

            print()
            print(
                "DETECTED CONCEPTS"
            )

            if concepts:

                for concept in concepts:

                    print(
                        f"- {concept}"
                    )

            else:

                print(
                    "- No explicit concept"
                )

            primary = detect_primary_concept(
                corrected,
                concepts
            )

            print()
            print(
                f"PRIMARY CONCEPT: "
                f"{primary if primary else 'None'}"
            )

            # ----------------------------------------------------
            # Expansion
            # ----------------------------------------------------

            expanded = expand_query(
                corrected,
                concepts
            )

            print()
            print(
                "EXPANDED QUERIES"
            )

            for i, q in enumerate(
                expanded,
                start=1
            ):

                print(
                    f"{i}. {q}"
                )

            # ----------------------------------------------------
            # Retrieval
            # ----------------------------------------------------

            results = retrieve(
                user_query,
                corrected,
                concepts,
                expanded
            )

            # ----------------------------------------------------
            # Results
            # ----------------------------------------------------

            print_results(
                results
            )

            # ----------------------------------------------------
            # Final answer
            # ----------------------------------------------------

            answer = choose_answer(
                results,
                concepts
            )

            print_answer(
                answer
            )

        except Exception as e:

            print(
                "\nError:",
                type(e).__name__,
                str(e)
            )


if __name__ == "__main__":
    main()