# RAG-Based Sales Analysis System
## Project Report

**Author:** Mohan Giri
**Date:** April 2026
**Repository:** https://github.com/girimohan/rag-sales-analysis

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Dataset](#3-dataset)
4. [System Architecture](#4-system-architecture)
5. [Technical Implementation](#5-technical-implementation)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Results and Validation](#7-results-and-validation)
8. [Challenges and Solutions](#8-challenges-and-solutions)
9. [Limitations](#9-limitations)
10. [Future Work](#10-future-work)
11. [Conclusion](#11-conclusion)

---

## 1. Introduction

This project builds a fully local, privacy-preserving question-answering system over a retail sales dataset using Retrieval-Augmented Generation (RAG). A user types a plain-English question — such as "Which region had the highest profit?" or "Who are the top customers by sales?" — and the system retrieves relevant data from a vector database and passes it to a local large language model (LLM) to generate a concise, grounded answer.

The entire pipeline runs on a local machine with no internet dependency: the vector store, the embedding model, and the LLM are all self-hosted. This makes the system suitable for scenarios where data privacy and offline operation are required.

---

## 2. Problem Statement

Business analysts working with tabular sales data typically rely on SQL queries, pivot tables, or BI dashboards to extract insights. These tools require technical knowledge and cannot answer free-form natural language questions. Conversely, general-purpose LLMs (ChatGPT, etc.) cannot answer dataset-specific questions without being given the data — and uploading sensitive business data to external APIs raises privacy concerns.

**Goal:** Build a system that allows non-technical users to ask natural language questions about a sales dataset and receive accurate, data-grounded answers — entirely on-premises, with no external API calls.

---

## 3. Dataset

| Property | Value |
|---|---|
| Name | Sample - Superstore |
| Source | Tableau / Kaggle |
| Rows | 9,994 orders |
| Columns | 21 (order ID, date, customer, region, product, sales, profit, etc.) |
| Time Range | January 2014 to December 2017 |
| Geography | United States only (4 regions, 49 states, 531 cities) |
| Encoding | windows-1252 |
| Size | ~2.5 MB |

The dataset covers retail orders across three product categories: **Furniture**, **Office Supplies**, and **Technology**. Key numeric fields used in the analysis are `sales`, `profit`, `quantity`, and `discount`.

---

## 4. System Architecture

```
User Question
      |
      v
[Streamlit UI]  ──POST /ask/stream──>  [FastAPI Backend]
                                              |
                                    [RAG Pipeline (pipeline.py)]
                                       /                \
                              [Query Classifier]   [Context Builder]
                                       |                  |
                              [ChromaDB Vector Store]     |
                              (9,994 row chunks +         |
                               26 summary docs)           |
                                       |                  |
                              [SentenceTransformer]       |
                              (all-MiniLM-L6-v2)          |
                                       |                  |
                                  [top-20 docs] ─────────>|
                                                          |
                                              [Ollama LLM (llama3)]
                                                          |
                                              Streaming tokens
                                                          |
                                              [Streamlit UI renders answer]
```

**Data flow:**
1. The user submits a question in the Streamlit UI.
2. The UI sends a POST request to the FastAPI `/ask/stream` endpoint with the question and selected model.
3. The pipeline embeds the query using `all-MiniLM-L6-v2` and retrieves the top-20 most semantically similar documents from ChromaDB.
4. An analytical query classifier determines whether the question is aggregate (e.g., "highest profit") or transactional (e.g., "find order CA-2017-...").
5. Documents are reordered — SUMMARY documents placed first for analytical queries, ROW-LEVEL documents first for transactional ones.
6. The assembled context and question are sent to the local Ollama LLM with a strict system prompt.
7. The LLM streams tokens back through FastAPI as a `StreamingResponse`; the UI accumulates and displays them in real time.

---

## 5. Technical Implementation

### 5.1 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.13 | Core runtime |
| Data processing | pandas, numpy | Dataset loading, aggregation |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` | Semantic vector encoding |
| GPU acceleration | PyTorch 2.6.0 + CUDA 12.4 | Embedding on GTX 1650 Ti |
| Vector database | ChromaDB (persistent) | Document storage and similarity search |
| LLM | Ollama + llama3 (default) / phi3:mini | Local inference, no internet needed |
| API backend | FastAPI + Uvicorn | REST endpoints for the pipeline |
| Frontend | Streamlit | Interactive web UI |
| Caching | Python `functools.lru_cache` | Query and embedding result reuse |

### 5.2 Vector Store — ChromaDB

The vector store contains two document types:

**Row-level chunks (9,994 documents)**
Each order row is serialised into a plain-text sentence, for example:
```
Order CA-2017-152156 | 2017-11-08 | Customer: Claire Gute | Segment: Consumer |
Region: South | State: Kentucky | City: Henderson | Category: Furniture |
Sub-Category: Bookcases | Product: Bush Somerset Collection Bookcase |
Sales: $261.96 | Quantity: 2 | Discount: 0.00 | Profit: $41.91
```
These documents answer order-specific lookups but are a small, non-representative sample for aggregation.

**Pre-computed summary documents (26 documents)**
Because row-level retrieval cannot reliably reconstruct totals from 9,994 rows using only 20 samples, 26 aggregate summary documents were pre-computed using pandas and injected into the same ChromaDB collection. These cover:

| Summary Type | Count |
|---|---|
| Profit/sales by region (all-time) | 1 |
| Profit by region per year (2014-2017) | 4 |
| Sales by region per year (2014-2017) | 4 |
| Profit by product category | 1 |
| Profit by sub-category (full list) | 1 |
| Sales and profit by year (annual totals) | 1 |
| Profit by state (top 10 + bottom 10) | 1 |
| Top city profits and losses | 1 |
| Top 10 customers by profit | 1 |
| Top 10 customers by sales | 1 |
| Top 5 customers per region (4 regions) | 4 |
| Profit by customer segment | 1 |
| Profit by ship mode | 1 |
| Sub-category sales by year | 4 |
| **Total** | **26** |

### 5.3 Analytical Query Classifier

A keyword-based classifier (`is_analytical()`) detects whether a question is analytical or transactional:

```python
ANALYTICAL_KEYWORDS = {
    "profit", "sales", "revenue", "region", "category", "highest", "lowest",
    "top", "best", "worst", "most", "least", "compare", "trend", "year", ...
}

def is_analytical(query: str) -> bool:
    tokens = query.lower().replace("-", " ").split()
    return bool(ANALYTICAL_KEYWORDS.intersection(tokens))
```

For analytical queries, summary documents are placed at the top of the retrieved context before row-level records. This ensures the LLM reads authoritative totals first.

### 5.4 LLM Prompt Engineering

The system prompt is designed to strictly confine the LLM to the retrieved context:

- The LLM is framed as a **data lookup tool**, not a reasoning agent, to suppress general-knowledge responses.
- A numbered **MANDATORY RULES** section requires the LLM to use only information present word-for-word in the context.
- An explicit **FORBIDDEN** section lists banned topics (climate, policies, non-US countries, filler phrases) to prevent hallucination patterns observed during testing.
- `temperature = 0` is set in Ollama options for deterministic, non-creative output.
- Output is capped at 2 sentences and 256 tokens.

### 5.5 Performance Optimisations

| Optimisation | Mechanism | Benefit |
|---|---|---|
| Embedding cache | `@lru_cache(maxsize=256)` on `_embed()` | Repeated queries skip GPU re-encoding |
| Answer cache | `@lru_cache(maxsize=128)` on `run_rag()` | Identical questions return instantly |
| Singleton model | `_model` initialised once, reused | Avoids reloading 90 MB model per request |
| Singleton ChromaDB client | `_client` initialised once, reused | Avoids disk re-open per request |
| GPU offloading | `num_gpu = -1` (all layers) on CUDA | LLM inference on GTX 1650 Ti |
| Streaming response | `ollama.chat(stream=True)` + `StreamingResponse` | First token appears in ~1 s |

### 5.6 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ask` | Returns complete answer as JSON (cached) |
| POST | `/ask/stream` | Streams answer tokens as plain text |

Both endpoints accept a JSON body `{"query": "...", "model": "llama3"}`. The `model` field defaults to `"llama3"` but accepts `"phi3:mini"` for faster, lighter inference.

### 5.7 User Interface

The Streamlit UI provides:
- A text area for entering questions
- A sidebar model selector (`llama3` / `phi3:mini`)
- Real-time streaming display using a `placeholder.text()` accumulator (plain text rendering to avoid Markdown/LaTeX conflicts with dollar signs in currency values)

---

## 6. Key Design Decisions

### Why RAG instead of fine-tuning?
Fine-tuning an LLM on the Superstore dataset would require significant compute and would bake the data into the model weights — making updates impossible without retraining. RAG keeps the data external and updatable: adding new orders requires only re-running `ingest.py`.

### Why pre-computed summaries instead of SQL-over-LLM?
Text-to-SQL approaches require the LLM to generate correct SQL and a live database connection. For a fixed dataset, pre-computing aggregates with pandas is more reliable and faster. The 26 summary documents cover all analytically relevant dimensions.

### Why local LLM (Ollama) instead of OpenAI API?
The entire motivation is data privacy and offline operation. Sending sales data to an external API is not appropriate for business data. Ollama runs llama3 and phi3:mini fully locally with no data leaving the machine.

### Why `temperature = 0`?
During testing, llama3 at default temperature (~0.8) hallucinated extensively — introducing climate data, government policies, and non-existent countries when it could not find an answer. Setting `temperature = 0` forces the model to output the highest-probability tokens, eliminating creative deviation from the context.

---

## 7. Results and Validation

The following questions were tested and validated against known ground-truth values computed directly in pandas:

| Question | System Answer | Ground Truth | Status |
|---|---|---|---|
| Which region had the highest profit? | West ($108,418.45) | West ($108,418.45) | Correct |
| Which city had the most losses? | Philadelphia (-$13,837.83) | Philadelphia (-$13,837.83) | Correct |
| Which sub-category has the lowest profit? | Tables (-$17,725.48) | Tables (-$17,725.48) | Correct |
| Which city had the highest profit? | New York City ($62,036.98) | New York City ($62,036.98) | Correct |
| Which category is most profitable? | Technology | Technology | Correct |

**Streaming:** First token appears in approximately 1 second; full 2-sentence answers complete in 3-6 seconds on a GTX 1650 Ti with llama3.

**Cache performance:** A repeated identical question returns in under 10 ms (served from `lru_cache`, no LLM call).

---

## 8. Challenges and Solutions

| Challenge | Root Cause | Solution |
|---|---|---|
| Wrong aggregate answers (e.g., wrong region) | top_k=5 retrieved only West region rows; no full-dataset totals | Injected 26 pre-computed summary documents; raised top_k to 20 |
| `TypeError: unsupported operand type(s) for \|: 'function' and 'NoneType'` | `chromadb.PersistentClient` is a factory function, not a class; `function \| None` is invalid in Python 3.10+ union syntax | Removed type annotations from singleton module-level variables |
| 404 on `/ask/stream` after adding endpoint | Old uvicorn process (started with `--reload`) still running on port 8000 | Killed old process; restarted uvicorn without `--reload` |
| Garbled `$` signs in streaming UI | `st.write_stream()` renders Markdown; `$24,051` is parsed as a LaTeX math delimiter | Switched to `placeholder.text(full_response)` accumulator (plain text) |
| LLM hallucinating climate, policies, non-US countries | Default temperature gave creative freedom; soft prompt guidelines were ignored | Rewrote prompt with numbered MANDATORY RULES + FORBIDDEN section; set `temperature = 0` |
| Verbose multi-paragraph answers | llama3 over-explains context and adds caveats | Added explicit 2-sentence output cap and "Stop after the answer" instruction |

---

## 9. Limitations

- **Static dataset:** The vector store reflects a snapshot of the Superstore CSV. New orders require re-running `ingest.py` and `add_summaries.py`.
- **Summary coverage:** The 26 summary documents cover pre-defined dimensions. A question about an unsupported combination (e.g., profit by sub-category within a specific state) will fall back to row-level documents, which may give an incomplete answer.
- **LLM instruction following:** Even with `temperature = 0` and a strict prompt, smaller models (phi3:mini) occasionally deviate from instructions on complex multi-part questions. llama3 is more reliable but slower.
- **No conversation memory:** Each question is answered independently. Follow-up questions ("What about the East region?") are not supported.
- **Hardware dependency:** Performance is optimised for CUDA. On CPU-only machines, embedding and inference will be significantly slower.

---

## 10. Future Work

- **Conversational memory:** Add a chat history buffer so users can ask follow-up questions in context.
- **Dynamic summaries:** Auto-detect when new data is ingested and regenerate affected summary documents.
- **Text-to-SQL fallback:** For queries outside summary coverage, generate and execute a SQL query against the raw data.
- **Evaluation framework:** Implement automated answer quality scoring using a ground-truth QA set and metrics such as answer faithfulness and context recall.
- **Multi-dataset support:** Extend the pipeline to support multiple uploaded CSV files with dynamic collection switching.
- **Authentication:** Add API key validation to the FastAPI endpoints before any production deployment.

---

## 11. Conclusion

This project demonstrates that a fully local, privacy-preserving question-answering system over tabular business data is practical and effective. The key insight is that pure retrieval from row-level chunks is insufficient for aggregate analytical questions — pre-computing and injecting summary documents as first-class vector store entries is a simple but critical design choice that makes the difference between wrong and correct answers.

The combination of ChromaDB for semantic retrieval, SentenceTransformers for embeddings, Ollama for local LLM inference, FastAPI for the backend, and Streamlit for the frontend provides a complete, deployable stack that requires no external services or internet connectivity. With `temperature = 0` and a rigorously structured system prompt, llama3 stays reliably within the retrieved context and produces concise, accurate answers.

---

*Report generated: April 2026*
