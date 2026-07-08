<div align="center">

# RAG Pipeline with Hallucination Detection

[![Python](https://shields.io)](https://python.org)
[![LangChain](https://shields.io)](https://langchain.com)
[![Hugging Face](https://shields.io🤗_Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co)
[![FAISS](https://shields.ioFAISS-Vector_DB-0467DF?style=for-the-badge)](https://github.com)
[![Apache Spark](https://shields.ioApache_Spark-E25A1B?style=for-the-badge&logo=apachespark&logoColor=white)](https://apache.org)
[![Hadoop](https://shields.ioHadoop-66CCFF?style=for-the-badge&logo=apachehadoop&logoColor=black)](https://apache.org)

---
</div>

## 📌 Overview
This project addresses LLM hallucination in RAG systems by implementing a **four-layer detection system** (Word Overlap, Semantic Similarity, LLM-as-Judge, Specificity Check). It ensures high-fidelity, grounded answers by validating output against source contexts.

Additionally, it outlines a production architecture for processing **500,000+ documents** using **Apache Spark and Hadoop HDFS** for efficient, parallelized vector ingestion.

---

## ⚡ Features
- 🛡️ **Four-Layer Hallucination Detection** — Multi-signal verification matrix.
- 🔍 **Semantic Retrieval** — FAISS-powered vector search.
- 🐘 **Big Data Processing** — Spark/Hadoop architecture for distributed preprocessing.
- ⚙️ **Modular Design** — Highly customizable, running on CPU via `flan-t5-base`.

---

### 🛡️ The Four Detection Layers

| Layer | Method | Catches |
|:-----:|:-------|:--------|
| **1** | **Word Overlap** | Token mismatches. |
| **2** | **Semantic Similarity** | Cosine distance mismatches. |
| **3** | **LLM-as-Judge** | Self-evaluation prompt loops. |
| **4** | **Specificity Check** | Empty/evasive answers. |

---

## 🐘 Production Scaling (Hadoop & Spark Architecture)
The system is designed for massive scale, moving from local ingestion to distributed processing:
*   **Hadoop HDFS:** Manages 500k+ documents for high-availability storage.
*   **Apache Spark:** Executes distributed cleaning, tokenization, and embedding generation via `all-mpnet-base-v2`.

---

## 🚀 Quickstart

### Installation
```bash
git clone https://github.com
cd rag-hallucination-detection
python -m venv .venv
source .venv/bin/activate  # Or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

### Run
```bash
python main.py
```

---

## 📊 Evaluation Results

| Question | Verdict | Insight |
|:---------|:-------:|:--------|
| Factual Query | 🟢 Valid | High similarity & specificity. |
| Low-Info Query | 🟣 Low_Info | Caught by Specificity Check. |

---

## 🛠️ Tech Stack & Configuration
*   **Orchestration:** LangChain
*   **Data Engine:** Spark/Hadoop
*   **Vector DB:** FAISS
*   **LLM:** `flan-t5-base`

*Thresholds can be tuned in `main.py` (`SIMILARITY_HALLUCINATION`, etc.).*
