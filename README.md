# RAG Pipeline with Multi-Layer Hallucination Detection

## Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system with a four-layer hallucination detection framework. The goal is to improve the reliability of LLM-generated answers by grounding responses in retrieved documents and validating them using multiple independent verification signals.

Unlike a standard RAG pipeline, this system does not simply return an answer. Every response is evaluated for factual grounding and assigned a confidence verdict using lexical, semantic, model-based, and specificity-based validation. During development, I also explored distributed data processing concepts using Apache Spark and Apache Hadoop to better understand how document preprocessing workflows can scale prior to embedding generation.

## Objectives

- Build an end-to-end Retrieval-Augmented Generation pipeline using open-source models.
- Retrieve relevant documents using dense semantic search.
- Generate grounded answers using an instruction-tuned language model.
- Detect hallucinations using multiple complementary validation techniques.
- Benchmark retrieval and generation quality.
- Design a modular architecture that can be extended to larger datasets.

## System Architecture

```text
Raw Documents
      │
      ▼
Document Loading
      │
      ▼
Text Cleaning & Normalization
      │
      ▼
Document Chunking
      │
      ▼
SentenceTransformer Embeddings
      │
      ▼
FAISS Vector Index
      │
      ▼
User Query
      │
      ▼
Semantic Retrieval
      │
      ▼
FLAN-T5 Answer Generation
      │
      ▼
Hallucination Detection
 ├── Word Overlap
 ├── Semantic Similarity
 ├── LLM-as-Judge
 └── Specificity Check
      │
      ▼
Grounded Response + Confidence Verdict
```

## Features

- End-to-end Retrieval-Augmented Generation pipeline
- Dense semantic retrieval using FAISS
- SentenceTransformer (`all-mpnet-base-v2`) embeddings
- Response generation using `google/flan-t5-base`
- Four-layer hallucination detection
- Transparent confidence scoring
- Modular architecture for experimentation
- Fully local execution using open-source models

## Data Processing Pipeline

The preprocessing workflow consists of:

1. Load raw documents.
2. Clean and normalize text.
3. Remove duplicate or redundant content.
4. Chunk documents for semantic retrieval.
5. Generate embeddings using SentenceTransformers.
6. Store embeddings in FAISS.
7. Retrieve relevant context.
8. Generate answers using FLAN-T5.
9. Validate answers using the four-layer hallucination detection framework.

### Exploring Scalable Data Processing

To better understand scalable retrieval pipelines, I explored Apache Spark and Apache Hadoop concepts.

- **Apache Spark:** explored for distributed preprocessing workflows such as text cleaning, normalization, duplicate removal, and document chunking.
- **Apache Hadoop:** explored to understand distributed storage concepts for managing document collections prior to embedding generation.

After preprocessing, document chunks are embedded using SentenceTransformers and indexed in FAISS for efficient semantic retrieval.

## Machine Learning Models

| Model | Purpose | Usage |
|------|---------|-------|
| SentenceTransformer (`all-mpnet-base-v2`) | Dense semantic embeddings | Pre-trained inference |
| Google FLAN-T5 Base | Answer generation | Pre-trained inference |
| FAISS | Vector similarity search | Vector indexing |
| NumPy | Similarity calculations | Utility |
| LangChain | Pipeline orchestration | Framework |

## Hallucination Detection

The generated response is validated using four complementary techniques:

1. **Word Overlap** – Measures lexical agreement between the generated answer and retrieved context.
2. **Semantic Similarity** – Computes cosine similarity between embeddings.
3. **LLM-as-Judge** – Verifies whether the answer is supported by the retrieved evidence.
4. **Specificity Check** – Flags vague or low-information responses.

The combined output produces a final confidence verdict.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Programming | Python 3.11 |
| Framework | LangChain |
| Deep Learning | PyTorch |
| Embeddings | SentenceTransformers |
| Language Model | Google FLAN-T5 Base |
| Vector Database | FAISS |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Distributed Processing | Apache Spark |
| Distributed Storage Concepts | Apache Hadoop |
| Version Control | Git |

## Installation

```bash
git clone https://github.com/akshitapotdar/rag-hallucination-detection.git
cd rag-hallucination-detection

python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Example Results

| Question | Generated Answer | Overlap | Semantic Similarity | LLM Judge | Specificity | Final Verdict |
|----------|------------------|:-------:|:-------------------:|:---------:|:-----------:|:-------------:|
| What is AI? | Simulation of human intelligence in machines | 0.83 | 0.56 | ✅ Yes | High | 🟡 Uncertain |
| What is Machine Learning? | Allows systems to learn from data | 0.83 | 0.48 | ✅ Yes | High | 🟡 Uncertain |
| Who invented AI? | Human | 1.00 | 0.26 | ✅ Yes | Low | 🟣 Low Information |
| What is a Neural Network? | Computational models inspired by the human brain | 0.86 | 0.41 | ✅ Yes | High | 🟡 Uncertain |

### Analysis

The multi-layer validation framework demonstrates that relying on a single metric is insufficient for hallucination detection. The answer **"Human"** achieves perfect lexical overlap but is correctly identified as **Low Information** due to low semantic similarity and limited informational content.

## Project Structure

```text
rag-hallucination-detection/
├── main.py
├── data.txt
├── requirements.txt
├── README.md
├── assets/
│   ├── architecture.svg
│   └── detection-layers.svg
```

## Evaluation and Benchmarking

| Metric | Purpose |
|---------|---------|
| Word Overlap | Lexical grounding |
| Semantic Similarity | Embedding-based grounding |
| LLM-as-Judge | Context verification |
| Specificity | Detect low-information responses |
| Final Verdict | Combined confidence score |

## Future Improvements

- Hybrid retrieval (BM25 + Dense Retrieval)
- Cross-Encoder reranking
- ONNX/TensorRT optimization
- Quantized inference
- GPU benchmarking
- Docker deployment
- AWS deployment
- Multi-document reasoning

## Key Learnings

- Retrieval-Augmented Generation architecture
- Semantic search with FAISS
- Prompt engineering
- Hallucination detection
- End-to-end ML pipeline design
- Model benchmarking
- Document preprocessing
- Scalable preprocessing concepts using Apache Spark and Hadoop

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Large Language Models
- Prompt Engineering
- Sentence Embeddings
- Semantic Search
- FAISS
- Hallucination Detection
- PyTorch
- LangChain
- Machine Learning Pipelines
- Model Evaluation
- Benchmarking
- Technical Documentation
