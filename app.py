# ============================================================
# BENGALI CULTURAL RAG V6.26
# STREAMLIT USER INTERFACE
# ============================================================
#
# Architecture
#
# User Bengali Situation
#          |
#          v
#     Query Correction
#          |
#          v
#    Concept Detection
#          |
#          v
#     Query Expansion
#          |
#          v
# E5 + BM25 + Concept + Meaning Grounding
#          |
#          v
# Cultural Expression
#
# NO GEMINI
# NO GROQ
# NO EXTERNAL LLM
# DATASET-FIRST
# ============================================================

import sys
import json
import time
import importlib
import textwrap
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
EVAL_DIR = PROJECT_ROOT / "evaluation"
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"


# ============================================================
# PYTHON PATH
# ============================================================

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Bengali Cultural RAG V6.26",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HTML RENDER HELPER
# ============================================================

def render_html(html: str):
    """
    Render HTML safely in Streamlit.

    textwrap.dedent() is important because otherwise
    Streamlit may interpret indented HTML as a code block.
    """

    st.markdown(
        textwrap.dedent(html).strip(),
        unsafe_allow_html=True,
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
    ===================================================== */

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* =====================================================
       HERO
    ===================================================== */

    .hero {
        padding: 38px 42px;
        border-radius: 26px;
        background:
            linear-gradient(
                135deg,
                #fff7ed 0%,
                #fef2f2 45%,
                #eff6ff 100%
            );
        border: 1px solid #e5e7eb;
        margin-bottom: 28px;
        box-shadow:
            0 12px 40px rgba(15, 23, 42, 0.08);
    }

    .hero h1 {
        margin: 0 0 12px 0;
        font-size: 2.25rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.25;
    }

    .hero p {
        margin: 0;
        color: #475569;
        font-size: 1.05rem;
        line-height: 1.8;
    }

    /* =====================================================
       ANSWER CARD
    ===================================================== */

    .answer-card {
        padding: 30px;
        border-radius: 22px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow:
            0 10px 35px rgba(15, 23, 42, 0.07);
        margin-bottom: 15px;
    }

    .answer-label {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .expression {
        font-size: 2.2rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.4;
        margin-bottom: 10px;
    }

    .answer-description {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* =====================================================
       INFO CARD
    ===================================================== */

    .info-card {
        padding: 22px;
        border-radius: 18px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        min-height: 110px;
    }

    .info-title {
        font-weight: 750;
        color: #334155;
        margin-bottom: 8px;
        font-size: 1rem;
    }

    .info-text {
        color: #475569;
        line-height: 1.7;
    }

    /* =====================================================
       CONCEPT CARD
    ===================================================== */

    .concept-card {
        padding: 16px 20px;
        border-radius: 16px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        font-weight: 650;
        font-size: 1.05rem;
        margin-bottom: 8px;
    }

    /* =====================================================
       METRIC CARD
    ===================================================== */

    .metric-card {
        text-align: center;
        padding: 20px 14px;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        background: #ffffff;
        box-shadow:
            0 5px 20px rgba(15, 23, 42, 0.04);
    }

    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #111827;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 5px;
    }

    /* =====================================================
       STATUS
    ===================================================== */

    .status-online {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-size: 0.8rem;
        font-weight: 750;
    }

    .status-offline {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: #fee2e2;
        color: #991b1b;
        font-size: 0.8rem;
        font-weight: 750;
    }

    /* =====================================================
       PIPELINE
    ===================================================== */

    .pipeline-card {
        padding: 20px;
        border-radius: 18px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        text-align: center;
        height: 100%;
    }

    .pipeline-number {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 700;
    }

    .pipeline-title {
        font-weight: 800;
        color: #1e293b;
        margin-top: 5px;
    }

    /* =====================================================
       FOOTER
    ===================================================== */

    .footer {
        text-align: center;
        color: #64748b;
        padding: 20px 10px 5px 10px;
        line-height: 1.7;
    }

    .footer-title {
        font-weight: 800;
        color: #475569;
    }

    /* =====================================================
       SIDEBAR
    ===================================================== */

    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f8fafc;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD V6.26 RETRIEVAL ENGINE
# ============================================================

@st.cache_resource(
    show_spinner="বাংলা Cultural RAG engine লোড হচ্ছে..."
)
def load_rag_engine():

    try:

        scripts_path = str(SCRIPTS_DIR)

        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

        # Dynamic import avoids Pylance unresolved-import warning
        rag = importlib.import_module("v6_rag")

        return rag, None

    except Exception as exc:

        return None, exc


engine, engine_error = load_rag_engine()


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_string(value):
    """
    Convert value to a clean string.
    """

    if value is None:
        return ""

    return str(value).strip()


def phrase(item):
    """
    Extract cultural expression from retrieval result.
    """

    if isinstance(item, dict):

        keys = [
            "phrase",
            "expression",
            "cultural_expression",
            "cultural_phrase",
            "name",
            "answer",
            "selected_phrase",
        ]

        for key in keys:

            value = item.get(key)

            if value:
                return safe_string(value)

    return safe_string(item)


def score(item):
    """
    Extract retrieval score.
    """

    if not isinstance(item, dict):
        return None

    keys = [
        "final_score",
        "score",
        "final",
        "similarity",
        "combined_score",
        "retrieval_score",
    ]

    for key in keys:

        value = item.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            continue

    return None


def extract_meaning(item):
    """
    Extract cultural meaning.
    """

    if not isinstance(item, dict):
        return ""

    keys = [
        "meaning",
        "cultural_meaning",
        "culturalMeaning",
        "definition",
        "meaning_en",
        "meaning_bn",
    ]

    for key in keys:

        value = item.get(key)

        if value:
            return safe_string(value)

    return ""


def extract_tone(item):
    """
    Extract tone / usage information.
    """

    if not isinstance(item, dict):
        return ""

    keys = [
        "tone",
        "usage_tone",
        "usage",
        "context",
        "emotional_tone",
    ]

    for key in keys:

        value = item.get(key)

        if value:
            return safe_string(value)

    return ""


# ============================================================
# NORMALIZE RETRIEVAL RESULTS
# ============================================================

def extract_results(raw):

    if isinstance(raw, list):
        return raw

    if isinstance(raw, tuple):

        for item in raw:

            if isinstance(item, list):
                return item

            if isinstance(item, dict):

                keys = [
                    "results",
                    "candidates",
                    "ranked_results",
                    "retrieval_results",
                    "top_results",
                ]

                for key in keys:

                    value = item.get(key)

                    if isinstance(value, list):
                        return value

    if isinstance(raw, dict):

        keys = [
            "results",
            "candidates",
            "ranked_results",
            "retrieval_results",
            "top_results",
        ]

        for key in keys:

            value = raw.get(key)

            if isinstance(value, list):
                return value

    return []


# ============================================================
# EXTRACT PREDICTION
# ============================================================

def extract_prediction(raw, results):

    prediction = ""
    confidence = 0.0

    if isinstance(raw, dict):

        prediction_keys = [
            "prediction",
            "answer",
            "selected_phrase",
            "selected_expression",
            "phrase",
            "expression",
        ]

        for key in prediction_keys:

            value = raw.get(key)

            if value:

                prediction = safe_string(value)

                break

        confidence_keys = [
            "confidence",
            "final_confidence",
            "retrieval_confidence",
        ]

        for key in confidence_keys:

            value = raw.get(key)

            if value is None:
                continue

            try:

                confidence = float(value)

                break

            except (
                TypeError,
                ValueError,
            ):
                continue

    # Fallback to first candidate
    if not prediction and results:

        prediction = phrase(results[0])

        first_score = score(results[0])

        if first_score is not None:

            confidence = first_score

    return prediction, confidence


# ============================================================
# FIND SELECTED DATASET RECORD
# ============================================================

def find_selected_result(
    results,
    prediction,
):

    if not prediction:
        return None

    prediction = safe_string(
        prediction
    )

    # Exact match
    for item in results:

        current = phrase(item)

        if current == prediction:

            return item

    # Partial match
    for item in results:

        current = phrase(item)

        if (
            prediction in current
            or current in prediction
        ):

            return item

    return None


# ============================================================
# LOAD JSON
# ============================================================

def load_json(filename):

    path = EVAL_DIR / filename

    if not path.exists():
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return None


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    dataset_path = (
        DATA_DIR
        / "processed"
        / "final_cultural_dataset.json"
    )

    if not dataset_path.exists():
        return []

    try:

        with open(
            dataset_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if isinstance(data, list):

            return data

        if isinstance(data, dict):

            for key in [
                "records",
                "data",
                "dataset",
                "items",
            ]:

                value = data.get(key)

                if isinstance(
                    value,
                    list,
                ):

                    return value

    except Exception:

        return []

    return []


dataset = load_dataset()


# ============================================================
# RUN V6.26 QUERY
# ============================================================

def run_query(query):

    if engine is None:

        raise RuntimeError(
            "V6.26 retrieval engine could not be loaded."
        )

    query = safe_string(query)

    if not query:

        raise ValueError(
            "Query cannot be empty."
        )

    # --------------------------------------------------------
    # 1. Query correction
    # --------------------------------------------------------

    corrected_query = engine.correct_query(
        query
    )

    # --------------------------------------------------------
    # 2. Concept detection
    # --------------------------------------------------------

    concepts = engine.detect_concepts(
        corrected_query
    )

    # --------------------------------------------------------
    # 3. Query expansion
    # --------------------------------------------------------

    expanded_queries = engine.expand_query(
        corrected_query,
        concepts,
    )

    # --------------------------------------------------------
    # 4. REAL RETRIEVAL
    # --------------------------------------------------------

    start = time.perf_counter()

    raw = engine.retrieve(
        query,
        corrected_query,
        concepts,
        expanded_queries,
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    results = extract_results(
        raw
    )

    prediction, confidence = (
        extract_prediction(
            raw,
            results,
        )
    )

    return {
        "query": query,
        "corrected": corrected_query,
        "concepts": concepts,
        "expanded": expanded_queries,
        "raw": raw,
        "results": results,
        "prediction": prediction,
        "confidence": confidence,
        "latency": latency_ms,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-title">
            📚 বাংলা Cultural RAG
        </div>
        """
    )

    st.caption(
        "V6.26 • Meaning-Grounded Cultural Retrieval"
    )

    mode = st.radio(
        "Mode",
        [
            "Search",
            "Evidence",
            "Benchmark",
        ],
    )

    st.divider()

    st.markdown(
        "### ⚙️ System"
    )

    if engine is not None:

        render_html(
            """
            <span class="status-online">
                ● ENGINE ONLINE
            </span>
            """
        )

    else:

        render_html(
            """
            <span class="status-offline">
                ● ENGINE ERROR
            </span>
            """
        )

    st.write("")

    st.write(
        f"**Dataset:** {len(dataset)} records"
    )

    st.write(
        "**Embedding:** E5"
    )

    st.write(
        "**Lexical:** BM25"
    )

    st.write(
        "**Grounding:** Meaning + Concept + Primary Intent"
    )

    st.write(
        "**LLM/API:** None"
    )

    st.divider()

    st.markdown(
        "### 💡 Example Queries"
    )

    examples = [

        "সে খুব অলস এবং কোনো কাজ করতে চায় না।",

        "সে খুব রেগে গেছে এবং প্রচণ্ড রাগান্বিত।",

        "সে এমন বিপদে পড়েছে যেখান থেকে বের হওয়ার কোনো পথ পাচ্ছে না।",

        "সে কী করবে বুঝতে পারছে না এবং খুব বিভ্রান্ত হয়ে গেছে।",

        "শেষ পর্যন্ত সে সফল হয়েছে এবং তার লক্ষ্য অর্জন করেছে।",

        "সে কাজটি বারবার পিছিয়ে দিচ্ছে এবং সময় নষ্ট করছে।",

        "লোকটি কৌশল করে অন্যদের ঠকিয়েছে।",

        "সে সফল হওয়ার জন্য খুব কঠোর পরিশ্রম করছে।",

        "লোকটি খুব দরিদ্র এবং অর্থকষ্টে জীবনযাপন করছে।",

        "সে সারাদিন অপ্রয়োজনীয় ও অর্থহীন কথাবার্তা বলে।",

        "পরিস্থিতির কারণে সে খুব উদ্বিগ্ন ও অস্থির হয়ে পড়েছে।",

        "ঘটনাটি তাকে খুব কষ্ট দিয়েছে এবং তার মনে গভীর যন্ত্রণা হয়েছে।",
    ]

    for index, example in enumerate(
        examples
    ):

        if st.button(
            example,
            key=f"example_{index}",
            use_container_width=True,
        ):

            st.session_state[
                "query"
            ] = example

            st.session_state.pop(
                "result",
                None,
            )

            st.rerun()


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">

        <h1>
            বাংলা সাংস্কৃতিক অভিব্যক্তি অনুসন্ধান
        </h1>

        <p>
            আপনার পরিস্থিতি বাংলায় লিখুন —
            Bengali Cultural RAG V6.26
            অর্থ, concept, semantic similarity
            এবং cultural meaning অনুযায়ী
            উপযুক্ত সাংস্কৃতিক অভিব্যক্তি খুঁজে দেবে।
        </p>

    </div>
    """
)


# ============================================================
# ENGINE ERROR
# ============================================================

if engine_error is not None:

    st.error(
        "V6.26 retrieval engine load করা যায়নি।"
    )

    with st.expander(
        "Technical Error"
    ):

        st.code(
            str(engine_error),
            language="text",
        )

    st.stop()


# ============================================================
# SEARCH / EVIDENCE
# ============================================================

if mode in [
    "Search",
    "Evidence",
]:

    query = st.text_area(
        "আপনার পরিস্থিতি লিখুন",
        value=st.session_state.get(
            "query",
            "",
        ),
        height=125,
        placeholder=(
            "যেমন: সে খুব অলস এবং কোনো কাজ করতে চায় না।"
        ),
    )

    search_clicked = st.button(
        "🔎 খুঁজুন",
        type="primary",
    )

    if search_clicked:

        if not query.strip():

            st.warning(
                "দয়া করে একটি বাংলা পরিস্থিতি লিখুন।"
            )

        else:

            try:

                with st.spinner(
                    "Cultural meaning অনুযায়ী retrieval চলছে..."
                ):

                    result = run_query(
                        query.strip()
                    )

                st.session_state[
                    "result"
                ] = result

                st.session_state[
                    "query"
                ] = query.strip()

            except Exception as exc:

                st.error(
                    "Retrieval-এর সময় সমস্যা হয়েছে।"
                )

                with st.expander(
                    "Technical Details"
                ):

                    st.code(
                        str(exc),
                        language="text",
                    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    result = st.session_state.get(
        "result"
    )

    if result:

        st.divider()

        # ====================================================
        # RESULT HEADER
        # ====================================================

        col_left, col_right = st.columns(
            [2.2, 1]
        )

        with col_left:

            st.markdown(
                "## 🎯 Recommended Expression"
            )

            prediction = result.get(
                "prediction",
                "",
            )

            if prediction:

                render_html(
                    f"""
                    <div class="answer-card">

                        <div class="answer-label">
                            সবচেয়ে উপযুক্ত সাংস্কৃতিক অভিব্যক্তি
                        </div>

                        <div class="expression">
                            {prediction}
                        </div>

                        <div class="answer-description">
                            V6.26 meaning-grounded
                            retrieval pipeline থেকে নির্বাচিত।
                        </div>

                    </div>
                    """
                )

            else:

                st.warning(
                    "এই query-এর জন্য নির্ভরযোগ্য "
                    "সাংস্কৃতিক অভিব্যক্তি পাওয়া যায়নি।"
                )

        with col_right:

            st.markdown(
                "## 📊 Retrieval"
            )

            confidence = result.get(
                "confidence",
                0.0,
            )

            latency = result.get(
                "latency",
                0.0,
            )

            st.metric(
                "Confidence",
                f"{confidence:.3f}",
            )

            st.metric(
                "Latency",
                f"{latency:.0f} ms",
            )

        # ====================================================
        # CONCEPTS
        # ====================================================

        st.markdown(
            "## 🧠 Detected Concepts"
        )

        concepts = result.get(
            "concepts",
            [],
        )

        if concepts:

            concept_cols = st.columns(
                min(
                    len(concepts),
                    4,
                )
            )

            for index, concept in enumerate(
                concepts
            ):

                with concept_cols[
                    index
                    % len(concept_cols)
                ]:

                    render_html(
                        f"""
                        <div class="concept-card">
                            {safe_string(concept)}
                        </div>
                        """
                    )

        else:

            st.info(
                "No explicit concept detected."
            )

        # ====================================================
        # CULTURAL INFORMATION
        # ====================================================

        selected = find_selected_result(
            result.get(
                "results",
                [],
            ),
            prediction,
        )

        st.markdown(
            "## 📖 Cultural Information"
        )

        info1, info2 = st.columns(2)

        meaning = ""

        tone = ""

        if selected:

            meaning = extract_meaning(
                selected
            )

            tone = extract_tone(
                selected
            )

        with info1:

            render_html(
                """
                <div class="info-card">

                    <div class="info-title">
                        📖 অর্থ / Meaning
                    </div>

                </div>
                """
            )

            if meaning:

                st.write(
                    meaning
                )

            else:

                st.caption(
                    "Dataset-এ আলাদা meaning field পাওয়া যায়নি।"
                )

        with info2:

            render_html(
                """
                <div class="info-card">

                    <div class="info-title">
                        🎭 Tone / Usage
                    </div>

                </div>
                """
            )

            if tone:

                st.write(
                    tone
                )

            else:

                st.caption(
                    "Dataset-এ tone/usage field পাওয়া যায়নি।"
                )

        # ====================================================
        # EVIDENCE MODE
        # ====================================================

        if mode == "Evidence":

            st.divider()

            st.markdown(
                "## 🔬 Retrieval Evidence"
            )

            # ------------------------------------------------
            # Pipeline
            # ------------------------------------------------

            st.markdown(
                "### Retrieval Pipeline"
            )

            p1, p2, p3, p4 = st.columns(4)

            with p1:

                render_html(
                    """
                    <div class="pipeline-card">

                        <div class="pipeline-number">
                            STEP 01
                        </div>

                        <div class="pipeline-title">
                            Query Correction
                        </div>

                    </div>
                    """
                )

            with p2:

                render_html(
                    """
                    <div class="pipeline-card">

                        <div class="pipeline-number">
                            STEP 02
                        </div>

                        <div class="pipeline-title">
                            Concept Detection
                        </div>

                    </div>
                    """
                )

            with p3:

                render_html(
                    """
                    <div class="pipeline-card">

                        <div class="pipeline-number">
                            STEP 03
                        </div>

                        <div class="pipeline-title">
                            Query Expansion
                        </div>

                    </div>
                    """
                )

            with p4:

                render_html(
                    """
                    <div class="pipeline-card">

                        <div class="pipeline-number">
                            STEP 04
                        </div>

                        <div class="pipeline-title">
                            E5 + BM25 Retrieval
                        </div>

                    </div>
                    """
                )

            st.write("")

            # ------------------------------------------------
            # Original Query
            # ------------------------------------------------

            st.markdown(
                "### Original Query"
            )

            st.info(
                result.get(
                    "query",
                    "",
                )
            )

            # ------------------------------------------------
            # Corrected Query
            # ------------------------------------------------

            st.markdown(
                "### Corrected Query"
            )

            st.write(
                result.get(
                    "corrected",
                    "",
                )
            )

            # ------------------------------------------------
            # Concepts
            # ------------------------------------------------

            st.markdown(
                "### Detected Concepts"
            )

            if concepts:

                for concept in concepts:

                    st.write(
                        f"• {concept}"
                    )

            else:

                st.write(
                    "None"
                )

            # ------------------------------------------------
            # Expanded Queries
            # ------------------------------------------------

            expanded = result.get(
                "expanded",
                [],
            )

            with st.expander(
                "🔄 Expanded Queries"
            ):

                if expanded:

                    for index, item in enumerate(
                        expanded,
                        start=1,
                    ):

                        st.write(
                            f"{index}. {item}"
                        )

                else:

                    st.write(
                        "No expanded queries."
                    )

            # ------------------------------------------------
            # Top 5
            # ------------------------------------------------

            st.markdown(
                "### 🏆 Top-5 Retrieval Candidates"
            )

            results = result.get(
                "results",
                [],
            )

            if results:

                for index, item in enumerate(
                    results[:5],
                    start=1,
                ):

                    current_phrase = phrase(
                        item
                    )

                    current_score = score(
                        item
                    )

                    if current_score is not None:

                        st.write(
                            f"**{index}. "
                            f"{current_phrase}** "
                            f"— score: "
                            f"`{current_score:.4f}`"
                        )

                    else:

                        st.write(
                            f"**{index}. "
                            f"{current_phrase}**"
                        )

            else:

                st.info(
                    "No retrieval candidates returned."
                )

            # ------------------------------------------------
            # Raw
            # ------------------------------------------------

            with st.expander(
                "Developer / Raw Retrieval Output"
            ):

                st.write(
                    result.get(
                        "raw"
                    )
                )


# ============================================================
# BENCHMARK MODE
# ============================================================

else:

    st.markdown(
        "## 📊 V6.26 Benchmark Dashboard"
    )

    st.caption(
        "শুধু actual saved benchmark results "
        "ব্যবহার করা হচ্ছে। কোনো fabricated metric নেই।"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = load_json(
        "evaluation_summary.json"
    )

    if summary:

        def summary_value(*keys):

            for key in keys:

                if key in summary:

                    return summary[key]

            return None


        top1 = summary_value(
            "top1_accuracy",
            "top1",
            "Top-1 Accuracy",
        )

        top3 = summary_value(
            "top3_accuracy",
            "top3",
            "Top-3 Accuracy",
        )

        top5 = summary_value(
            "top5_accuracy",
            "top5",
            "Top-5 Accuracy",
        )

        mrr = summary_value(
            "mrr",
            "MRR",
        )

        ndcg = summary_value(
            "ndcg_at_5",
            "ndcg@5",
            "NDCG@5",
        )

        mean_latency = summary_value(
            "mean_latency_ms",
            "mean_latency",
            "Mean Latency",
        )

        p95_latency = summary_value(
            "p95_latency_ms",
            "p95_latency",
            "P95 Latency",
        )

        mean_confidence = summary_value(
            "mean_confidence",
            "confidence",
            "Mean Confidence",
        )

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        st.markdown(
            "### 🎯 Retrieval Performance"
        )

        metric_cols = st.columns(4)

        metric_data = [
            (
                "Top-1 Accuracy",
                top1,
            ),
            (
                "Top-3 Accuracy",
                top3,
            ),
            (
                "Top-5 Accuracy",
                top5,
            ),
            (
                "MRR",
                mrr,
            ),
        ]

        for column, (
            label,
            value,
        ) in zip(
            metric_cols,
            metric_data,
        ):

            if isinstance(
                value,
                (int, float),
            ):

                if 0 <= value <= 1:

                    display_value = (
                        f"{value:.2%}"
                    )

                else:

                    display_value = (
                        f"{value:.4f}"
                    )

            else:

                display_value = "—"

            with column:

                render_html(
                    f"""
                    <div class="metric-card">

                        <div class="metric-value">
                            {display_value}
                        </div>

                        <div class="metric-label">
                            {label}
                        </div>

                    </div>
                    """
                )

        # ----------------------------------------------------
        # System performance
        # ----------------------------------------------------

        st.write("")

        st.markdown(
            "### ⚡ System Performance"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "NDCG@5",
                (
                    f"{ndcg:.4f}"
                    if isinstance(
                        ndcg,
                        (int, float),
                    )
                    else "—"
                ),
            )

        with c2:

            st.metric(
                "Mean Latency",
                (
                    f"{mean_latency:.2f} ms"
                    if isinstance(
                        mean_latency,
                        (int, float),
                    )
                    else "—"
                ),
            )

        with c3:

            st.metric(
                "P95 Latency",
                (
                    f"{p95_latency:.2f} ms"
                    if isinstance(
                        p95_latency,
                        (int, float),
                    )
                    else "—"
                ),
            )

        with c4:

            st.metric(
                "Mean Confidence",
                (
                    f"{mean_confidence:.4f}"
                    if isinstance(
                        mean_confidence,
                        (int, float),
                    )
                    else "—"
                ),
            )

    else:

        st.warning(
            "evaluation_summary.json পাওয়া যায়নি।"
        )

    # ========================================================
    # QUERY LEVEL RESULTS
    # ========================================================

    benchmark_data = load_json(
        "benchmark_results_clean.json"
    )

    if isinstance(
        benchmark_data,
        dict,
    ):

        rows = (
            benchmark_data.get(
                "results"
            )
            or benchmark_data.get(
                "queries"
            )
            or benchmark_data.get(
                "records"
            )
            or []
        )

    elif isinstance(
        benchmark_data,
        list,
    ):

        rows = benchmark_data

    else:

        rows = []

    # ========================================================
    # TABLE / QUERY RESULTS
    # ========================================================

    if rows:

        st.divider()

        st.markdown(
            "### 🧪 Query-level Benchmark Results"
        )

        for index, row in enumerate(
            rows,
            start=1,
        ):

            if not isinstance(
                row,
                dict,
            ):

                continue

            query_text = (
                row.get("query")
                or row.get("original_query")
                or ""
            )

            prediction = (
                row.get("prediction")
                or "NONE"
            )

            gold = (
                row.get("gold_phrase")
                or row.get("gold")
                or ""
            )

            top1_value = row.get(
                "top1",
                row.get(
                    "top1_correct",
                    "—",
                ),
            )

            top3_value = row.get(
                "top3",
                row.get(
                    "top3_correct",
                    "—",
                ),
            )

            top5_value = row.get(
                "top5",
                row.get(
                    "top5_correct",
                    "—",
                ),
            )

            mrr_value = row.get(
                "mrr",
                "—",
            )

            ndcg_value = row.get(
                "ndcg_at_5",
                row.get(
                    "ndcg@5",
                    "—",
                ),
            )

            latency_value = row.get(
                "latency_ms",
                row.get(
                    "latency",
                    "—",
                ),
            )

            confidence_value = row.get(
                "confidence",
                "—",
            )

            with st.expander(
                f"{index}. {query_text}"
            ):

                left, right = st.columns(2)

                with left:

                    st.markdown(
                        "**Gold Expression**"
                    )

                    st.success(
                        safe_string(
                            gold
                        )
                    )

                with right:

                    st.markdown(
                        "**Prediction**"
                    )

                    if prediction in [
                        gold,
                        safe_string(gold),
                    ]:

                        st.success(
                            safe_string(
                                prediction
                            )
                        )

                    else:

                        st.warning(
                            safe_string(
                                prediction
                            )
                        )

                st.write("")

                metric_cols = st.columns(
                    6
                )

                metric_cols[0].metric(
                    "Top-1",
                    safe_string(
                        top1_value
                    ),
                )

                metric_cols[1].metric(
                    "Top-3",
                    safe_string(
                        top3_value
                    ),
                )

                metric_cols[2].metric(
                    "Top-5",
                    safe_string(
                        top5_value
                    ),
                )

                metric_cols[3].metric(
                    "MRR",
                    safe_string(
                        mrr_value
                    ),
                )

                metric_cols[4].metric(
                    "NDCG@5",
                    safe_string(
                        ndcg_value
                    ),
                )

                metric_cols[5].metric(
                    "Confidence",
                    safe_string(
                        confidence_value
                    ),
                )

                st.caption(
                    f"Latency: "
                    f"{safe_string(latency_value)}"
                )

    else:

        st.info(
            "benchmark_results_clean.json "
            "পাওয়া যায়নি।"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

render_html(
    """
    <div class="footer">

        <div class="footer-title">
            Bengali Cultural RAG V6.26
        </div>

        <div>
            E5 + BM25 + Concept +
            Meaning Grounding + Primary Intent
        </div>

        <div>
            Dataset-first • No Gemini • No Groq • No external LLM
        </div>

    </div>
    """
)