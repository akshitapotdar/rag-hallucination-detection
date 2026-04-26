<div align="center">

# 🧠 RAG Pipeline with Hallucination Detection

### *A Retrieval-Augmented Generation system that doesn't just answer — it knows when it's wrong.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-0467DF?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)



---

</div>

##  Overview

Large Language Models hallucinate. They generate confident-sounding answers that aren't grounded in any real source — a critical failure mode in retrieval systems where users trust the output.

This project implements a **four-layer hallucination detection system** on top of a standard RAG pipeline. Instead of just generating answers, it scores their reliability across multiple independent dimensions and combines them into a single verdict.


---

##  Features

- 🔍 **Semantic Retrieval** — FAISS-powered vector search over your knowledge base
- 🤖 **Lightweight Generation** — `flan-t5-base` for fast, focused answers
- 🛡️ **Four-Layer Hallucination Detection** — independent signals combined into a robust verdict
- 📊 **Transparent Scoring** — every answer comes with its full diagnostic breakdown
- ⚙️ **Modular Design** — swap models, embeddings, or detection methods without touching the core pipeline
- 🧪 **No External APIs Required** — runs entirely locally on Hugging Face open models

---

### The Four Detection Layers


Each signal has different failure modes — by requiring agreement across all four, the system catches issues no single check could detect alone.

| Layer | Method | Catches |
|:-----:|:-------|:--------|
| **1** | **Word Overlap** | Vocabulary mismatch between answer and source — classic hallucination signal |
| **2** | **Semantic Similarity** | Cosine distance between answer and best-matching chunk — catches paraphrases *and* semantic emptiness |
| **3** | **LLM-as-Judge** | Asks the model to verify its own output against the context |
| **4** | **Specificity Check** | Flags ultra-short, low-information answers that game the other metrics |

---

##  Quickstart

### Prerequisites

- Python 3.11+
- ~2 GB free disk space (for model weights)
- Windows / macOS / Linux

### Installation

```bash
# Clone the repo
git clone https://github.com/YourUsername/llm-project.git
cd llm-project

# Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

That's it. First run will download the models (~500 MB); subsequent runs are instant.

---
## Results
 
Sample run on a small AI/ML knowledge base:
 
| Question | Generated Answer | Overlap | Similarity | Judge | Specificity | Verdict |
|:---------|:-----------------|:-------:|:----------:|:-----:|:-----------:|:-------:|
| What is AI? | *simulation of human intelligence in machines* | 0.83 | 0.56 | ✅ YES | HIGH | 🟡 **UNCERTAIN** |
| What is machine learning? | *allows systems to learn from data* | 0.83 | 0.48 | ✅ YES | HIGH | 🟡 **UNCERTAIN** |
| Who invented AI? | *human* | 1.00 | 0.26 | ✅ YES | LOW | 🟣 **LOW_INFO** |
| What is neural network? | *computational models inspired by the human brain* | 0.86 | 0.41 | ✅ YES | HIGH | 🟡 **UNCERTAIN** |
 
### Key Insight
 
Look at the third row. The answer **"human"** has *perfect* word overlap (1.00) and even passes the LLM judge — single-metric systems would mark this as grounded. But the **specificity check** catches it as `LOW_INFO`, and the **low semantic similarity** (0.26) confirms the answer is semantically thin.
 
This is exactly why multi-layer detection matters: **each layer catches what the others miss.**
 
The other three answers land at `UNCERTAIN` — high word overlap and judge approval, but the answer-to-context vectors aren't tightly aligned because answers are short (~6 words) compared to the joined retrieved chunks. Tuning the similarity threshold downward, or comparing against the best individual chunk instead of joined context, reliably promotes these to `GROUNDED`. See the [Configuration](#%EF%B8%8F-configuration) section for tuning guidance.
 
---


##  Project Structure

```
llm-project/
├── main.py             # Full pipeline: RAG + hallucination detection
├── data.txt            # Source knowledge base
├── requirements.txt    # Locked dependencies
├── assets/
│   ├── architecture.svg        # System architecture diagram
│   └── detection-layers.svg    # Detection layers diagram
├── .gitignore
└── README.md
```

---

##  Tech Stack

| Component | Choice | Why |
|:----------|:-------|:----|
| **Orchestration** | LangChain | Mature ecosystem, clean abstractions |
| **Vector Store** | FAISS | Fast, local, no infra needed |
| **Embeddings** | `all-mpnet-base-v2` | Strong semantic quality, reasonable size |
| **Generator** | `google/flan-t5-base` | Instruction-tuned, runs on CPU |
| **Math** | NumPy | Cosine similarity, vector ops |

---

##  Configuration

Detection thresholds live in `hallucination_flag()` in `main.py`:

```python
# Tunable thresholds
SIMILARITY_HALLUCINATION = 0.30   # below this → likely hallucination
SIMILARITY_GROUNDED      = 0.45   # above this → semantic match passes
OVERLAP_GROUNDED         = 0.50   # word overlap floor for GROUNDED
SPECIFICITY_MIN_WORDS    = 3      # answers shorter than this → LOW_INFO
```

These were tuned for short answers against medium-length chunks. If your context shape differs (longer chunks, longer answers, different domain), retune them empirically:

1. Run on 15–20 questions where you know the ground truth
2. Look at score distributions for grounded vs. hallucinated cases
3. Pick thresholds that best separate them

---

##  Try Breaking It

Want to verify the detection actually works? Add adversarial questions to the `questions` list in `main.py`:

```python
questions = [
    "What is AI?",
    "Is AI purple?",                          # nonsense → should flag
    "Did Einstein invent machine learning?",  # false attribution → should flag
    "Tell me about quantum cryptography.",    # off-topic → should flag
]
```

If the system marks these as `GROUNDED`, your thresholds need tightening.

---

