# ============================================================
# BENGALI CULTURAL RAG
# INTELLIGENT RELEVANCE SELECTOR
# ============================================================

import re


# ============================================================
# STOP WORDS
# ============================================================

BANGLA_STOPWORDS = {
    "আমি",
    "আমার",
    "আমাকে",
    "সে",
    "তিনি",
    "তারা",
    "খুব",
    "একটি",
    "একটা",
    "করে",
    "হয়েছে",
    "হয়েছে",
    "আছে",
    "ছিল",
    "পড়েছি",
    "পড়েছি",
    "পড়েছে",
    "পড়েছে",
    "করছে",
    "হয়ে",
    "হয়ে",
    "যে",
    "এবং",
    "বা",
    "এর",
    "এ",
    "ও",
    "তো",
    "কি",
    "কী",
    "জন্য",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    text = text.lower().strip()

    text = re.sub(
        r"[^\w\s\u0980-\u09FF]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str):

    text = normalize_text(text)

    tokens = text.split()

    return {
        token
        for token in tokens
        if token not in BANGLA_STOPWORDS
        and len(token) > 1
    }


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(
    query: str,
    item: dict
) -> float:

    query_tokens = tokenize(query)

    candidate_text = " ".join([
        item.get("bengali_phrase", ""),
        item.get("cultural_meaning", ""),
        item.get("intended_emotion_tone", ""),
    ])

    candidate_tokens = tokenize(candidate_text)

    if not query_tokens:
        return 0.0

    overlap = query_tokens.intersection(
        candidate_tokens
    )

    return len(overlap) / len(query_tokens)


# ============================================================
# DANGER / TROUBLE SIGNALS
# ============================================================

DANGER_WORDS = {
    "বিপদ",
    "বিপদে",
    "সমস্যা",
    "সমস্যায়",
    "সমস্যায়",
    "ঝামেলা",
    "সঙ্কট",
    "সংকট",
    "danger",
    "trouble",
}


ANGER_WORDS = {
    "রাগ",
    "রেগে",
    "ক্ষোভ",
    "angry",
    "furious",
}


LAZY_WORDS = {
    "অলস",
    "অলসতা",
    "lazy",
}


HESITATION_WORDS = {
    "দ্বিধা",
    "দ্বিধায়",
    "দ্বিধায়",
    "সিদ্ধান্ত",
    "হয়তো",
    "হয়তো",
    "hesitate",
}


WEALTH_WORDS = {
    "ধনী",
    "ধনী হয়ে",
    "ধনী হয়ে",
    "টাকা",
    "সম্পদ",
    "rich",
}


RAIN_WORDS = {
    "বৃষ্টি",
    "বৃষ্টির",
    "গুঁড়ি",
    "গুঁড়ি গুঁড়ি",
    "বৃষ্টিপাত",
    "rain",
}


# ============================================================
# DOMAIN SIGNAL DETECTION
# ============================================================

def detect_signals(query: str):

    normalized = normalize_text(query)

    signals = set()

    word_groups = {
        "danger": DANGER_WORDS,
        "anger": ANGER_WORDS,
        "lazy": LAZY_WORDS,
        "hesitation": HESITATION_WORDS,
        "wealth": WEALTH_WORDS,
        "rain": RAIN_WORDS,
    }

    for category, words in word_groups.items():

        for word in words:

            if normalize_text(word) in normalized:

                signals.add(category)

                break

    return signals


# ============================================================
# MEANING SIGNAL
# ============================================================

MEANING_SIGNAL_MAP = {

    "danger": [
        "danger",
        "trouble",
        "distress",
        "disaster",
        "fearful",
        "desperate",
        "dangerous",
    ],

    "anger": [
        "angry",
        "furious",
        "anger",
        "irritated",
    ],

    "lazy": [
        "lazy",
        "inactive",
    ],

    "hesitation": [
        "hesitate",
        "uncertain",
        "reluctant",
        "confused",
    ],

    "wealth": [
        "rich",
        "wealth",
        "prosperous",
        "cash",
        "money",
    ],

    "rain": [
        "rain",
        "drizzling",
    ],
}


def meaning_signal_score(
    query: str,
    item: dict
) -> float:

    signals = detect_signals(query)

    if not signals:

        return 0.0

    meaning = normalize_text(
        item.get(
            "cultural_meaning",
            ""
        )
    )

    tone = normalize_text(
        item.get(
            "intended_emotion_tone",
            ""
        )
    )

    combined = (
        meaning
        + " "
        + tone
    )

    matched = 0

    for signal in signals:

        keywords = MEANING_SIGNAL_MAP.get(
            signal,
            []
        )

        if any(
            normalize_text(keyword)
            in combined
            for keyword in keywords
        ):

            matched += 1

    return matched / len(signals)


# ============================================================
# PHRASE RELEVANCE
# ============================================================

def phrase_relevance(
    query: str,
    item: dict
) -> float:

    query_normalized = normalize_text(
        query
    )

    phrase = normalize_text(
        item.get(
            "bengali_phrase",
            ""
        )
    )

    if not phrase:

        return 0.0

    if phrase in query_normalized:

        return 1.0

    return 0.0


# ============================================================
# FINAL RELEVANCE SCORE
# ============================================================

def calculate_relevance(
    query: str,
    item: dict
) -> dict:

    semantic_score = float(
        item.get(
            "similarity",
            0.0
        )
    )

    keyword_score = keyword_overlap(
        query,
        item
    )

    meaning_score = meaning_signal_score(
        query,
        item
    )

    phrase_score = phrase_relevance(
        query,
        item
    )

    # --------------------------------------------------------
    # Weighted score
    # --------------------------------------------------------

    final_score = (
        0.70 * semantic_score
        + 0.10 * keyword_score
        + 0.20 * meaning_score
    )

    result = dict(item)

    result["semantic_score"] = round(
        semantic_score,
        4
    )

    result["keyword_score"] = round(
        keyword_score,
        4
    )

    result["meaning_score"] = round(
        meaning_score,
        4
    )

    result["phrase_score"] = round(
        phrase_score,
        4
    )

    result["relevance_score"] = round(
        final_score,
        4
    )

    return result


# ============================================================
# SELECT BEST CONTEXT
# ============================================================

def select_relevant_context(
    query: str,
    retrieved_items: list,
    max_results: int = 4,
    min_score: float = 0.45
):

    scored = []

    for item in retrieved_items:

        scored.append(
            calculate_relevance(
                query,
                item
            )
        )

    # Highest relevance first
    scored.sort(
        key=lambda x: x["relevance_score"],
        reverse=True
    )

    selected = [
        item
        for item in scored
        if item["relevance_score"] >= min_score
    ]

    selected = selected[
        :max_results
    ]

    return scored, selected