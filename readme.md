# Corrective Retrieval-Augmented Generation (CRAG)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-blue)

**A research-oriented implementation of Corrective Retrieval-Augmented Generation (CRAG) built from scratch using PyTorch, Hugging Face, FAISS, BM25, and Streamlit.**

</div>

---

# Overview

Large Language Models (LLMs) often generate incorrect or incomplete responses when retrieval quality is poor. Traditional Retrieval-Augmented Generation (RAG) systems assume retrieved documents are always relevant, which frequently leads to hallucinations.

This project implements the **Corrective Retrieval-Augmented Generation (CRAG)** pipeline, where retrieval quality is first evaluated before generation. When retrieved documents are insufficient, the system performs corrective retrieval using query refinement and web search to improve the final context supplied to the language model.

The project is implemented as a modular research codebase with separate components for retrieval, refinement, generation, evaluation, and visualization.

---

# Features

* Dense vector retrieval using Sentence Transformers + FAISS
* BM25 sparse retrieval
* Hybrid retrieval pipeline
* Retrieval quality evaluation
* Query refinement
* Web search fallback using DuckDuckGo
* Context-aware response generation
* Interactive Streamlit interface
* Modular architecture for research and experimentation
* Configuration-driven design

---

# Architecture

```text
                   User Query
                        │
                        ▼
              ┌──────────────────┐
              │ Query Processing │
              └──────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 Dense Retrieval                  BM25 Retrieval
        │                               │
        └───────────────┬───────────────┘
                        ▼
                 Hybrid Retrieval
                        │
                        ▼
          Retrieval Quality Evaluation
                        │
          ┌─────────────┴─────────────┐
          │                           │
      High Quality               Low Quality
          │                           │
          ▼                           ▼
      Generation              Query Refinement
                                      │
                                      ▼
                               Web Search
                                      │
                                      ▼
                             Updated Context
                                      │
                                      ▼
                                Final Generation
```

---

# Project Structure

```text
CRAG/
│
├── ANALYSIS/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── index/
│
├── src/
│   ├── core/
│   ├── evaluator/
│   ├── generator/
│   ├── pipeline/
│   ├── refinement/
│   ├── retrieval/
│   ├── search/
│   └── utils/
│
├── pyproject.toml
└── README.md
```

---

# Tech Stack

| Category         | Technologies              |
| ---------------- | ------------------------- |
| Language         | Python 3.12               |
| Deep Learning    | PyTorch                   |
| LLM Framework    | Hugging Face Transformers |
| Dense Retrieval  | Sentence Transformers     |
| Vector Database  | FAISS                     |
| Sparse Retrieval | BM25                      |
| Web Search       | DuckDuckGo Search         |
| Configuration    | Pydantic                  |
| UI               | Streamlit                 |
| Visualization    | Plotly                    |

---

# Installation


## Create a virtual environment

```bash
uv venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install dependencies

```bash
uv sync
```

---

# Running the Project

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

# Workflow

1. Receive a user query.
2. Retrieve candidate documents using dense and sparse retrieval.
3. Combine retrieval results using a hybrid strategy.
4. Evaluate retrieval quality.
5. If retrieval confidence is high:

   * Generate the response directly.
6. Otherwise:

   * Refine the query.
   * Perform web search.
   * Retrieve additional evidence.
   * Generate the final response using the improved context.

---

# Dependencies

Major libraries used in this implementation include:

* PyTorch
* Transformers
* Sentence Transformers
* FAISS
* Rank-BM25
* Streamlit
* Plotly
* DuckDuckGo Search
* BeautifulSoup
* Pydantic
* Datasets
* NumPy

---

# Current Implementation Status

| Module                  | Status |
| ----------------------- | :----: |
| Dense Retrieval         |    ✅   |
| BM25 Retrieval          |    ✅   |
| Hybrid Retrieval        |    ✅   |
| Retrieval Evaluation    |    ✅   |
| Query Refinement        |    ✅   |
| Web Search Integration  |    ✅   |
| Context Generation      |    ✅   |
| Streamlit UI            |    ✅   |

---

# Research Goals

This project focuses on understanding and implementing the core concepts behind Corrective Retrieval-Augmented Generation rather than relying solely on existing frameworks.

Key objectives include:

* Improving retrieval quality before generation.
* Reducing errors caused by poor retrieval.
* Building a modular RAG research framework.
* Providing an extensible architecture for future experimentation.

---

# Future Improvements

* Hallucination detection module
* Reranking with cross-encoders
* Multi-query retrieval
* ColBERT integration
* Knowledge graph retrieval
* Agentic retrieval workflows
* Comprehensive benchmark evaluation
* Docker deployment
* CI/CD pipeline

---

# License

This project is released under the MIT License.

---

# Acknowledgements

This implementation is inspired by research on **Corrective Retrieval-Augmented Generation (CRAG)** and leverages several open-source projects, including:

* PyTorch
* Hugging Face Transformers
* Sentence Transformers
* FAISS
* Streamlit
* Rank-BM25

---

## Author

**Hem Khokhani**

Research interests:

* Retrieval-Augmented Generation (RAG)
* Information Retrieval
* Large Language Models
* Natural Language Processing
* Machine Learning
