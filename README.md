# 🇧🇩 Bengali Cultural RAG

### Meaning-Grounded Retrieval for Culturally Appropriate Bengali Expressions

A dataset-grounded retrieval framework that maps a natural-language description of a situation to an appropriate Bengali cultural expression. The system combines **E5-based semantic retrieval**, **BM25 lexical retrieval**, **concept detection**, **query expansion**, and **meaning-grounded ranking**.

<p align="center">
<a href="https://bengali-cultural-rag.streamlit.app/"><img src="https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge" alt="Live Demo"></a>
<a href="https://github.com/pritom05-crypto/bengali-cultural-rag"><img src="https://img.shields.io/badge/💻_Source_Code-GitHub-181717?style=for-the-badge&logo=github" alt="Source Code"></a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Dataset-408%20Records-2E7D32?style=flat-square" alt="Dataset">
<img src="https://img.shields.io/badge/Top--1-83.33%25-1565C0?style=flat-square" alt="Top 1">
<img src="https://img.shields.io/badge/Top--3-100%25-1565C0?style=flat-square" alt="Top 3">
<img src="https://img.shields.io/badge/Top--5-100%25-1565C0?style=flat-square" alt="Top 5">
<img src="https://img.shields.io/badge/MRR-0.9028-6A1B9A?style=flat-square" alt="MRR">
<img src="https://img.shields.io/badge/NDCG%405-0.9213-6A1B9A?style=flat-square" alt="NDCG">
</p>

---

## 🔗 Quick Links

| | Link |
|---|---|
| 🚀 **Live Application** | https://bengali-cultural-rag.streamlit.app/ |
| 💻 **GitHub Repository** | https://github.com/pritom05-crypto/bengali-cultural-rag |

---

## 🎯 Overview

Traditional lexical retrieval can struggle when a user describes the **meaning of a situation** without using words that directly occur in the target expression.

For example:

> **সে খুব অলস এবং কোনো কাজ করতে চায় না।**

The system retrieves the culturally appropriate expression:

> **কুঁড়ের বাঘ**

The proposed framework therefore treats the task as **situation-to-expression retrieval**, combining semantic meaning, lexical evidence, conceptual compatibility, and cultural meaning.

---

## 🧠 Retrieval Pipeline

```text
Natural-Language Situation
            │
            ▼
      Preprocessing
            │
            ▼
     Concept Detection
            │
            ▼
      Query Expansion
            │
       ┌────┴────┐
       ▼         ▼
   E5 Semantic  BM25 Lexical
    Retrieval    Retrieval
       │         │
       └────┬────┘
            ▼
     Candidate Merging
            │
            ▼
 Meaning-Grounded Ranking
            │
            ▼
 Confidence + Margin Gate
        ┌───┴───┐
        ▼       ▼
   Expression  Abstain
```

---

## 🔍 Main Components

### 1. Self-Constructed Bengali Cultural Dataset
The framework uses a dataset constructed specifically for this study containing **408 expression records**, including Bengali phrases, variants, literal meaning, cultural meaning, expression type, tone, and review metadata.

### 2. Concept Detection
The query is analyzed to identify its primary situation concept, moving beyond individual words toward the underlying meaning.

### 3. Query Expansion
Related terms, synonyms, and concept-associated representations improve retrieval when the user's wording differs from the dataset wording.

### 4. E5 Semantic Retrieval
E5 embeddings provide meaning-level matching between situation queries and cultural expression candidates.

### 5. BM25 Lexical Retrieval
BM25 provides complementary exact and near-exact lexical evidence.

### 6. Meaning-Grounded Ranking
Candidates are ranked using semantic, lexical, conceptual, cultural-meaning, linguistic-pattern, and mismatch evidence.

### 7. Confidence-Based Decision
The highest-ranked candidate is returned when the configured evidence requirements are satisfied; otherwise, the system can abstain.

---

## 📊 Evaluation Results

The verified benchmark contains **12 situation-oriented queries**.

| Metric | Result |
|---|---:|
| Dataset Records | 408 |
| Benchmark Queries | 12 |
| Top-1 Accuracy | **83.33%** |
| Top-3 Accuracy | **100.00%** |
| Top-5 Accuracy | **100.00%** |
| MRR | **0.9028** |
| NDCG@5 | **0.9213** |
| Mean Confidence | **0.9149** |
| Mean Latency | **1117.55 ms** |
| P95 Latency | **1848.26 ms** |

The remaining Top-1 errors still retain relevant candidates within the top-ranked retrieval set, indicating that the main challenge is fine-grained discrimination among culturally related expressions.

---

## 🖥️ Live Demo

<p align="center">
<a href="https://bengali-cultural-rag.streamlit.app/"><img src="https://img.shields.io/badge/▶_OPEN_LIVE_APP-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open Live App"></a>
</p>

### Example Queries

**Query 1**
```text
সে খুব অলস এবং কোনো কাজ করতে চায় না।
```

**Retrieved expression:** `কুঁড়ের বাঘ`

**Query 2**
```text
সে খুব রেগে গেছে এবং প্রচণ্ড রাগান্বিত।
```

**Retrieved expression:** `অগ্নিশর্মা`

---

## 📁 Project Structure

```text
bengali-cultural-rag/
│
├── .streamlit/
│   └── config.toml
├── app.py
├── data/
│   ├── chroma_db/
│   ├── embeddings/
│   ├── processed/
│   └── raw/
├── evaluation/
├── figures/
│   └── paper/
├── scripts/
│   ├── build_embeddings.py
│   ├── hybrid_retrieval.py
│   ├── meaning_aware_retrieval.py
│   ├── query_expansion.py
│   ├── rag_pipeline.py
│   └── ...
├── tables/
│   └── paper/
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Local Installation

### 1. Clone

```bash
git clone https://github.com/pritom05-crypto/bengali-cultural-rag.git
cd bengali-cultural-rag
```

### 2. Create environment

```bash
conda create -n bengali-rag python=3.11
conda activate bengali-rag
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
streamlit run app.py
```

---

## 🛠️ Technology Stack

| Area | Technology |
|---|---|
| Interface | Streamlit |
| Language | Python |
| Semantic Retrieval | E5 Embeddings |
| Lexical Retrieval | BM25 |
| Vector Database | ChromaDB |
| Data Processing | Pandas / NumPy |
| Evaluation | Top-k Accuracy, MRR, NDCG@5 |
| Deployment | Streamlit Community Cloud |

---

## 📌 Key Characteristics

- **Dataset-first design**
- **Meaning-oriented retrieval**
- **Hybrid semantic + lexical retrieval**
- **Concept-aware candidate selection**
- **Meaning-grounded reranking**
- **Confidence-aware decision making**
- **Abstention for insufficient evidence**
- **No external generative API required for final expression selection**

---

## 🔬 Research Contribution

The project addresses a specific retrieval problem:

> **Given a natural-language description of a situation, retrieve a culturally appropriate Bengali expression from a curated cultural knowledge base.**

The framework integrates semantic, lexical, conceptual, and cultural-meaning evidence within a unified retrieval pipeline.

---

## 🚀 Future Work

- Expand the Bengali cultural expression dataset
- Increase benchmark size and diversity
- Conduct component-level ablation studies
- Improve meaning-aware reranking
- Improve confidence calibration
- Evaluate larger and more context-rich Bengali queries

---

## 📄 Research

This repository contains the implementation, dataset resources, evaluation scripts, figures, and result tables associated with the Bengali cultural expression retrieval study.

---

## 👤 Author

**Pritom**

GitHub: https://github.com/pritom05-crypto

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐.

<p align="center">
<a href="https://github.com/pritom05-crypto/bengali-cultural-rag"><img src="https://img.shields.io/badge/⭐_Star_this_repository-GitHub-181717?style=for-the-badge&logo=github" alt="Star Repository"></a>
</p>
