# scripts/query_expansion.py

import os
import json
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai


# ============================================================
# GEMINI CLIENT
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. "
        "Please set your Gemini API key in the environment."
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query: str):
    """
    Expand a Bengali natural-language query into
    semantically related Bengali concepts.

    The expansion does NOT generate cultural expressions.
    It only identifies the meaning/intention of the query.
    """

    prompt = f"""
তুমি একটি Bengali Cultural Retrieval System-এর Query Expansion Module।

ব্যবহারকারীর query:
"{query}"

তোমার কাজ হলো query-টির মূল অর্থ পরিবর্তন না করে
retrieval-এর জন্য ৫-৮টি semantically related concept তৈরি করা।

বিশেষভাবে:
- Bengali natural language ব্যবহার করবে।
- সমার্থক বা কাছাকাছি অর্থের phrase ব্যবহার করবে।
- query-এর situation/emotion/action বুঝে related concepts দেবে।
- কোনো Bengali idiom, proverb বা cultural expression নিজে থেকে তৈরি করবে না।
- কোনো explanation দেবে না।
- JSON format ছাড়া অন্য কিছু লিখবে না।

Output format:

{{
  "original_query": "{query}",
  "expanded_queries": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    # Remove markdown code fences if Gemini adds them
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        print("Warning: Gemini returned invalid JSON.")
        print("Raw response:")
        print(text)

        return {
            "original_query": query,
            "expanded_queries": [query]
        }

    expanded = result.get("expanded_queries", [])

    if not isinstance(expanded, list):
        expanded = []

    # Clean values
    expanded = [
        str(x).strip()
        for x in expanded
        if str(x).strip()
    ]

    # Always keep original query
    final_queries = [query]

    for q in expanded:
        if q not in final_queries:
            final_queries.append(q)

    return {
        "original_query": query,
        "expanded_queries": final_queries
    }


# ============================================================
# INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("BENGALI CULTURAL RAG - GEMINI QUERY EXPANSION")
    print("=" * 70)

    query = input("\nআপনার প্রশ্ন লিখুন:\n> ").strip()

    if not query:
        print("Empty query.")
        raise SystemExit

    print("\n" + "=" * 70)
    print("QUERY EXPANSION")
    print("=" * 70)

    result = expand_query(query)

    print("\nOriginal:")
    print(result["original_query"])

    print("\nExpanded Queries:")

    for i, q in enumerate(result["expanded_queries"], 1):
        print(f"{i}. {q}")

    print("\n" + "=" * 70)
    print("QUERY EXPANSION COMPLETE")
    print("=" * 70)