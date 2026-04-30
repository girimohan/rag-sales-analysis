# RAG-Based Sales Analysis System
## Technical Report

**Team Members:** Mohan Giri · Razib Hasan
**Date:** April 2026
**Repository:** https://github.com/girimohan/rag-sales-analysis

---

## Table of Contents

1. [Introduction and Problem Statement](#1-introduction-and-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Data Preprocessing and Chunking](#3-data-preprocessing-and-chunking)
4. [Embedding Model and Vector Database](#4-embedding-model-and-vector-database)
5. [LLM Selection and Prompt Engineering](#5-llm-selection-and-prompt-engineering)
6. [Sample Queries and Responses](#6-sample-queries-and-responses)
7. [Challenges Encountered and Solutions](#7-challenges-encountered-and-solutions)
8. [Division of Work](#8-division-of-work)
9. [AI Tool Usage Disclosure](#9-ai-tool-usage-disclosure)

---

## 1. Introduction and Problem Statement

This project builds a fully local, privacy-preserving question-answering system over a retail sales dataset using Retrieval-Augmented Generation (RAG). A user types a plain-English question — such as "Which region had the highest profit?" — and the system retrieves relevant data from a vector database, passes it to a local LLM, and streams a concise, grounded answer in real time.

**Problem:** Business analysts relying on SQL or BI dashboards need technical knowledge for every new query. Cloud LLMs (e.g., ChatGPT) cannot answer dataset-specific questions without receiving the data — raising serious privacy concerns. **Goal:** allow non-technical users to query sales data accurately and privately, with no external API calls.

**Dataset:** The Superstore dataset (Tableau/Kaggle) — 9,994 US retail orders, January 2014–December 2017, 21 columns (order ID, customer, region, category, sales, profit, discount, etc.), three product categories, 4 regions, 49 states, 531 cities. File encoding: `windows-1252`.

---

## 2. System Architecture

```
[Streamlit UI] --(POST /ask/stream)--> [FastAPI + Uvicorn]
                                               |
                                      [RAG Pipeline]
                                    /               \
                        [Query Classifier]    [Context Builder]
                                |                    |
                        [ChromaDB]           assembled context
                   (9,994 row chunks +               |
                     26 summary docs)                |
                                |                    |
                  [SentenceTransformer embed]         |
                    top-20 docs retrieved ---------->|
                                                     |
                                         [Ollama LLM (llama3)]
                                            temperature=0
                                                     |
                                         streaming tokens --> UI
```

**Data flow:** (1) User submits a question and selects a model in the Streamlit sidebar. (2) `POST /ask/stream` is sent to FastAPI. (3) The query is embedded with `all-MiniLM-L6-v2` (CUDA) and top-20 documents are retrieved from ChromaDB. (4) An analytical keyword classifier determines if the question is aggregate or transactional. (5) Summary documents are placed first for aggregate queries. (6) Context and question are sent to Ollama with `temperature=0`. (7) Tokens stream back through `StreamingResponse`; the UI renders them live in plain text.

| Component | Technology |
|---|---|
| Data processing | Python 3.13, pandas, numpy |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2`, PyTorch 2.6 + CUDA 12.4 |
| Vector database | ChromaDB (persistent, local) |
| LLM | Ollama + llama3 / phi3:mini |
| API + UI | FastAPI + Uvicorn, Streamlit |
| Caching | `functools.lru_cache` (embeddings + answers) |

---

## 3. Data Preprocessing and Chunking

**Loading and normalisation:** The CSV is read with `pandas.read_csv(encoding="windows-1252")`. Column names are normalised to lowercase with underscores via a regex pipeline in `clean_columns()` (e.g., `"Sub-Category"` → `"sub_category"`).

**Row-to-text serialisation (`row_to_text.py`):** Each row is converted into a self-contained natural-language sentence embedding all 21 fields — order ID, date, customer, location, product, category, sales, discount, and profit. Missing values are replaced with "N/A". This format makes every field semantically searchable.

**Chunking (`chunker.py`):** Text is split at 500-character boundaries. Because each serialised row is ~300–350 characters, all 9,994 rows produce exactly one chunk each — no mid-sentence splits, no information loss.

**Pre-computed summary documents (`add_summaries.py`):** Pure row-level retrieval cannot reconstruct aggregate totals from a 20-document sample. To fix this, 26 pandas `groupby` summary documents were pre-computed and injected into ChromaDB as first-class documents, covering: profit/sales by region (all-time + per year 2014–2017), category, sub-category, state, top cities, top customers, segment, ship mode, and sub-category sales by year. One-time ingestion of all 9,994 chunks takes ~20 seconds on a GTX 1650 Ti.

---

## 4. Embedding Model and Vector Database

**Embedding model — `all-MiniLM-L6-v2`:** Chosen for its 384-dimensional vectors optimised for semantic retrieval, small footprint (~90 MB), proven benchmark performance, and full local/offline operation. CUDA is used automatically when available; CPU fallback is included. A singleton pattern loads the model once per server lifetime.

**Vector database — ChromaDB (persistent):** Chosen because it is purpose-built for embedding storage and nearest-neighbour search, requires no external server process (the store lives in a local `chroma_db/` folder), supports persistent storage across restarts, and allows document ID prefixing (`chunk_*` vs. `summary_*`) used by the analytical classifier to distinguish document types. Batching at 5,000 documents per insert respects ChromaDB's maximum batch size of 5,461.

---

## 5. LLM Selection and Prompt Engineering

**LLM — llama3 via Ollama (default):** Runs fully locally (8B parameters), follows structured instructions reliably with `temperature=0`, and offloads all layers to GPU via `num_gpu=-1`. `phi3:mini` (3.8B) is offered as a faster alternative via the UI sidebar, though it deviates from strict instructions more often. No external API keys or internet connection are needed.

**Prompt engineering:** Early testing revealed that llama3 at default temperature hallucinated extensively — inventing climate data, government policies, and non-US countries when context was thin. Four techniques were applied iteratively to fix this:

1. **Role framing:** The LLM is declared a "data lookup tool", not an analyst — suppressing world-knowledge reasoning.
2. **Numbered MANDATORY RULES:** Requires the model to use only text present word-for-word in the context, output a fallback phrase if the answer is absent, and stop after 2 sentences.
3. **Explicit FORBIDDEN section:** Bans climate, policies, non-US countries, and filler phrases ("However", "Additionally", "In summary") that were observed as hallucination starters.
4. **`temperature=0` + `num_predict=256`:** Deterministic output, no creative generation, hard token cap.

---

## 6. Sample Queries and Responses

All responses were verified against pandas ground truth. Screenshots of the running UI are in the `screenshots/` folder of the repository (see screenshot guidance at end of report).

| Query | System Response | Ground Truth |
|---|---|---|
| Which region had the highest profit? | West — $108,418.45 | West — $108,418.45 ✓ |
| Which city had the most losses? | Philadelphia — -$13,837.83 | Philadelphia — -$13,837.83 ✓ |
| Which sub-category has the lowest profit? | Tables — -$17,725.48 | Tables — -$17,725.48 ✓ |
| Which city had the highest profit? | New York City — $62,036.98 | New York City — $62,036.98 ✓ |
| Which category is most profitable? | Technology — $145,454.95 | Technology ✓ |

**Performance:** first streaming token ~1 s; full answer 3–6 s (llama3, GTX 1650 Ti); cached repeated query < 10 ms.

---

## 7. Challenges Encountered and Solutions

**Wrong aggregate answers:** top_k=5 retrieved a biased row sample, not full-dataset totals. *Fix:* injected 26 pre-computed summary documents; raised top_k to 20; added analytical classifier to prioritise summaries.

**`TypeError` on ChromaDB singleton:** `chromadb.PersistentClient | None` raised a runtime `TypeError` because `PersistentClient` is a factory function, not a class — Python 3.10+ union syntax cannot be applied to a function. *Fix:* removed type annotation from the singleton variables.

**404 on `/ask/stream`:** An old uvicorn process (started with `--reload`) was still occupying port 8000 after the new endpoint was added. *Fix:* killed the old process; restarted without `--reload`.

**Garbled dollar signs in Streamlit:** `st.write_stream()` renders Markdown, interpreting `$24,051` as a LaTeX math delimiter. *Fix:* replaced with a `placeholder.text()` accumulator — plain text, no Markdown parsing.

**LLM hallucinating climate/policies/countries:** llama3 at default temperature filled thin context with general world knowledge (eco-districts, carbon policies, seasonal temperatures). *Fix:* rewrote system prompt with numbered MANDATORY RULES and explicit FORBIDDEN section; set `temperature=0`.

**Verbose multi-paragraph answers:** llama3 added caveats and explanations beyond what was asked. *Fix:* hard 2-sentence cap in both system and user prompts; `num_predict=256`; "Stop after the answer."

---

## 8. Division of Work

Both members explored and downloaded the dataset together. All major architectural decisions, prompt engineering, debugging, and final validation were done jointly.

**Mohan Giri** set up the GitHub repository, virtual environment, and project structure. He implemented the RAG pipeline, ChromaDB ingestion, pre-computed summaries, retrieval module, and performance optimisations.

**Razib Hasan** handled data loading and preprocessing, built the EDA notebook, developed the FastAPI backend (including the streaming endpoint), and created the Streamlit UI with the model selector.

The report and presentation slides were written and reviewed together.

---

## 9. AI Tool Usage Disclosure

*(This section does not count toward the page limit.)*

### 9.1 AI Tools Used

**GitHub Copilot (Claude Sonnet 4.6)** was used throughout the project via VS Code as an AI pair programmer. The following table documents which parts involved AI assistance and the nature of the prompts given:

| Component | AI Assistance | Example Prompts / Instructions Given |
|---|---|---|
| `row_to_text.py` | Code generation | "Write a function that converts a pandas row from the Superstore dataset into a natural language string including all fields" |
| `chunker.py` | Code generation | "Write a chunker that splits text into 500-character chunks" |
| `build_store.py` | Code generation | "Write a function to embed text chunks with SentenceTransformers and store them in ChromaDB persistently" |
| `add_summaries.py` | Code generation | "Pre-compute profit/sales aggregates by region, category, sub-category, state, city, segment, ship mode, and top customers using pandas groupby, and inject them as ChromaDB documents" |
| `pipeline.py` | Code generation + debugging | "Add an analytical keyword classifier that puts summary documents first in context; fix TypeError on PersistentClient type annotation" |
| `app/main.py` | Code generation | "Add a /ask/stream endpoint using FastAPI StreamingResponse with ollama.chat(stream=True)" |
| `ui/app.py` | Code generation + debugging | "Add a model selector sidebar; fix dollar sign rendering in streaming output" |
| `prompts.py` | Iterative prompt engineering | Multiple rounds: "Tighten the system prompt to prevent hallucination; add FORBIDDEN section; add numbered MANDATORY RULES; set temperature=0" |
| Report and slides | Drafting + refinement | "Write a technical report for this RAG project covering architecture, chunking, embeddings, LLM, challenges" |

### 9.2 Problems and Errors in AI-Generated Code

1. **`TypeError` on ChromaDB type annotation:** The AI annotated `_client: chromadb.PersistentClient | None = None`. This raised a runtime `TypeError` because `PersistentClient` is a factory function, not a class. We fixed it by removing the annotation.
2. **`st.write_stream()` rendering bug:** The AI suggested `st.write_stream()` for streaming display. Streamlit rendered it as Markdown, breaking dollar signs as LaTeX. We identified and fixed it with a `placeholder.text()` accumulator.
3. **Hallucination through soft prompt instructions:** Early AI-drafted prompts used polite language that llama3 ignored. We had to observe specific failure outputs, diagnose the pattern, and design the FORBIDDEN section ourselves before the AI could help encode it.
4. **ChromaDB batch size crash:** The AI generated a single `collection.add()` for all 9,994 chunks, exceeding the 5,461 limit. We added a `batch_size=5000` loop.

### 9.3 Running Results

The system was successfully deployed. All five validation queries returned correct answers matching pandas ground truth. Ingestion took ~20 seconds. Streaming delivered first tokens in ~1 second. Screenshots of the Streamlit UI, FastAPI `/docs` page, and terminal ingestion output are in the `screenshots/` folder.

### 9.4 Student Contributions Beyond AI-Generated Code

- **Two-tier document architecture:** The insight that row-level retrieval is insufficient for aggregates — and the design of pre-computed summaries in the same collection — was our own. The AI implemented it; we designed it.
- **Prompt engineering through observed failure:** We collected real bad outputs (climate, eco-districts, carbon policies) and built the FORBIDDEN section directly from them. The AI encoded the rules; we identified them.
- **Bug diagnosis:** Both the `TypeError` and `st.write_stream()` bugs required understanding chromadb's and Streamlit's internal APIs. We diagnosed root causes independently.
- **`temperature=0` decision:** We decided this after observing that soft guidelines were insufficient — an inference configuration decision requiring understanding of token sampling.
- **Integration and validation:** We integrated all components end-to-end, cross-checked all query answers against pandas, and resolved import path, port conflict, and encoding issues that arose during integration.

---

*Report submitted: April 2026 — Mohan Giri & Razib Hasan*


---

## 1. Introduction and Problem Statement

This project builds a fully local, privacy-preserving question-answering system over a retail sales dataset using Retrieval-Augmented Generation (RAG). A user types a plain-English question — such as "Which region had the highest profit?" or "Who are the top customers by sales?" — and the system retrieves relevant data from a vector database, then passes it to a local large language model (LLM) to generate a concise, grounded answer.

**Problem:** Business analysts working with tabular sales data typically rely on SQL queries or BI dashboards, which require technical knowledge and cannot answer free-form natural language questions. General-purpose cloud LLMs (e.g., ChatGPT) cannot answer dataset-specific questions without being given the data — and uploading sensitive business data to external APIs raises serious privacy concerns.

**Goal:** Allow non-technical users to ask natural language questions about a sales dataset and receive accurate, data-grounded answers — entirely on-premises, with no external API calls.

**Dataset:** The Superstore dataset (Tableau/Kaggle) contains 9,994 retail orders across the United States from January 2014 to December 2017, covering three product categories (Furniture, Office Supplies, Technology), 4 regions, 49 states, and 531 cities. Key fields are `sales`, `profit`, `quantity`, and `discount`. The file uses `windows-1252` encoding.

---

## 2. System Architecture

The system consists of five components connected in a pipeline:

```
[Streamlit UI] --> POST /ask/stream --> [FastAPI + Uvicorn]
                                              |
                                     [RAG Pipeline]
                                    /              \
                        [Query Classifier]    [Context Builder]
                                |                   |
                        [ChromaDB]           assembled context
                    (9,994 row chunks +             |
                      26 summary docs)              |
                                |                   |
                  [SentenceTransformer embed]        |
                     top-20 docs retrieved -------->|
                                                    |
                                        [Ollama LLM (llama3)]
                                           temperature=0
                                                    |
                                      streaming tokens --> UI
```

**Data flow:**
1. User submits a question in the Streamlit UI, selecting a model (llama3 or phi3:mini) in the sidebar.
2. The UI sends `POST /ask/stream` to FastAPI with `{"query": "...", "model": "llama3"}`.
3. The pipeline embeds the query using `all-MiniLM-L6-v2` (on GPU via CUDA) and retrieves the top-20 most semantically similar documents from ChromaDB.
4. A keyword-based analytical query classifier checks whether the question is aggregate ("highest profit", "top region") or transactional ("find order ID...").
5. For analytical queries, pre-computed SUMMARY documents are placed first in the context; row-level documents follow. For transactional queries, the order is reversed.
6. The assembled context and question are sent to the Ollama LLM with a strict system prompt and `temperature=0`.
7. The LLM streams tokens back through FastAPI as a `StreamingResponse`. The Streamlit UI accumulates tokens in a `placeholder.text()` display updated on each chunk.

**Technology stack:**

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Data processing | pandas, numpy |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| GPU acceleration | PyTorch 2.6.0 + CUDA 12.4 (GTX 1650 Ti) |
| Vector database | ChromaDB (persistent, local) |
| LLM | Ollama + llama3 (default) / phi3:mini |
| API backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Caching | `functools.lru_cache` |

---

## 3. Data Preprocessing and Chunking

**Loading and column normalisation (`load_data.py`, `preprocess.py`):**
The CSV is loaded with `pandas.read_csv(path, encoding="windows-1252")`. Column names are then normalised to lowercase with underscores (e.g., `"Sub-Category"` becomes `"sub_category"`) using a regex-based strip-and-replace pipeline in `clean_columns()`. This makes field access uniform across the codebase.

**Row-to-text conversion (`row_to_text.py`):**
Each row of the 9,994-row DataFrame is serialised into a self-contained natural-language sentence that includes all fields a user might ask about:

```
Order CA-2017-152156 was placed on 2017-11-08 with ship mode 'Second Class'.
Customer 'Claire Gute' (Consumer segment) is located in Henderson, Kentucky,
South region. They ordered 'Bush Somerset Collection Bookcase'
(Category: Furniture, Sub-Category: Bookcases). Quantity: 2, Sales: $261.96,
Discount: 0.0, Profit: $41.91.
```

This format embeds all semantically relevant context into a single retrievable unit. Missing values are replaced with "N/A".

**Chunking (`chunker.py`):**
Each text string is split into chunks of at most 500 characters using character-based sliding window chunking. Because each row serialises to approximately 300–350 characters, nearly all rows produce exactly one chunk — meaning the 9,994 rows become 9,994 chunks with no information loss from mid-sentence splits.

**Pre-computed summary documents (`add_summaries.py`):**
A critical design decision was to inject 26 pre-computed aggregate summary documents into the same ChromaDB collection. These are computed using pandas `groupby` over the full dataset and cover profit/sales by region (all-time and per year 2014–2017), category, sub-category, state, city, segment, ship mode, top customers, and sub-category sales by year. This is explained further in Section 4.

**Vector store construction (`build_store.py`):**
`all-MiniLM-L6-v2` embeds all chunks in one batch on CUDA, then ChromaDB stores them in batches of 5,000 (ChromaDB's maximum batch size is 5,461). The one-time ingestion of 9,994 chunks takes approximately 20 seconds on a GTX 1650 Ti.

---

## 4. Embedding Model and Vector Database Choices

### Embedding Model: `all-MiniLM-L6-v2`

**Choice:** `sentence-transformers/all-MiniLM-L6-v2`

**Justification:**
- Produces 384-dimensional dense vectors optimised for semantic similarity and retrieval tasks — precisely the use case here.
- Runs efficiently on modest GPU hardware (GTX 1650 Ti, 4 GB VRAM). The full model is ~90 MB.
- Widely validated on benchmark retrieval tasks; strikes a good balance between speed and accuracy compared to larger alternatives (e.g., `all-mpnet-base-v2`).
- Fully open-source and runs locally — consistent with the project's privacy-first design goal.

CUDA acceleration is detected automatically (`device="cuda"` if `torch.cuda.is_available()`), with CPU fallback for collaborators without a compatible GPU. A singleton pattern ensures the model is loaded once and reused across all requests, avoiding the ~2-second reload overhead per query.

### Vector Database: ChromaDB (persistent)

**Choice:** ChromaDB with `PersistentClient(path="chroma_db")`

**Justification:**
- Designed specifically for embedding storage and approximate nearest-neighbour search — no SQL overhead for vector operations.
- Persistent storage means the vector store survives server restarts without re-embedding.
- Simple Python API with no external server process required — the entire DB lives in a local folder, keeping the stack self-contained.
- Supports metadata filtering, which enables the two-document-type design (row-level `chunk_*` IDs vs. summary `summary_*` IDs) used by the analytical classifier.

**The two-document-type design:**
Pure row-level retrieval is insufficient for aggregate queries. When asked "Which region had the highest profit?", retrieving top-20 rows by semantic similarity returns rows that happen to mention "profit" and "West" — a biased sample, not a total. The solution is to pre-compute complete aggregates with pandas and inject them as 26 summary documents into the same collection. The analytical classifier then prioritises these documents in the context, ensuring the LLM reads a ground-truth total like "West: profit=$108,418.45, orders=3,203" rather than inferring from a handful of rows.

---

## 5. LLM Selection and Prompt Engineering

### LLM Selection

**Primary model: llama3 (via Ollama)**
- 8B parameter model running fully locally — no data leaves the machine.
- Instruction-following capability is strong enough to adhere to the MANDATORY RULES prompt structure.
- With `temperature=0`, responses are deterministic and consistent.
- Runs on GPU via Ollama's `num_gpu=-1` option (all layers offloaded to VRAM).

**Alternative: phi3:mini**
- 3.8B parameter Microsoft model — faster, lower memory footprint.
- Available in the UI as a selectable alternative for users who prioritise speed over accuracy.
- Found to deviate from strict instructions more often than llama3 on complex queries during testing.

Both models are served locally via Ollama, which handles model loading, GPU scheduling, and HTTP inference. No OpenAI or external API keys are required.

### Prompt Engineering

The system prompt went through multiple iterations after observing hallucination behaviour during testing. The final design uses four techniques:

**1. Role framing as a data lookup tool (not a reasoning agent):**
```
You are a data lookup tool for a US Superstore sales dataset (2014-2017).
Your ONLY job is to read the retrieved context below and extract the answer.
```
Framing the model as a lookup tool suppresses its tendency to reason, explain, or apply general world knowledge.

**2. Numbered MANDATORY RULES:**
```
1. Use ONLY information that appears word-for-word in the retrieved context. Zero exceptions.
2. If the answer is not in the context, output exactly: "The context does not contain
   enough information to answer this question."
3. Never compute, estimate, infer, or reason beyond what is explicitly written.
4. Never use outside knowledge. Ignore everything you know about the real world.
5. The dataset is 100% United States data.
6. Output at most 2 sentences. First sentence = direct answer with exact number. Then STOP.
```
Numbered rules are followed more reliably than bullet-point guidelines.

**3. Explicit FORBIDDEN section:**
```
FORBIDDEN:
- Mentioning climate, weather, temperature, seasons, or environment
- Mentioning policies, regulations, laws, or government programs
- Any country other than the United States
- Sentences beginning with "However", "Additionally", "In summary", "While"
- Reasoning chains or explanations longer than 2 sentences
- Any number not copied directly from the context
```
This directly addresses the specific hallucination patterns observed (see Section 7).

**4. `temperature=0`:**
Set in the Ollama call options: `{"temperature": 0, "num_predict": 256, "num_gpu": -1}`. This forces deterministic output — the model consistently picks the highest-probability token at each step, eliminating creative elaboration.

---

## 6. Sample Queries and Responses

The following queries were run against the deployed system and verified against pandas-computed ground truth values:

**Query 1:** *"Which region had the highest profit?"*
> System: "According to the summary, the West region had the highest total profit of $108,418.45 across all years (2014-2017)."
> Ground truth (pandas): West — $108,418.45 ✓

**Query 2:** *"Which city had the most losses?"*
> System: "Philadelphia had the most losses with a total profit of -$13,837.83 according to the summary data."
> Ground truth (pandas): Philadelphia — -$13,837.83 ✓

**Query 3:** *"Which sub-category has the lowest profit?"*
> System: "Tables has the lowest profit among all sub-categories, with a total of -$17,725.48."
> Ground truth (pandas): Tables — -$17,725.48 ✓

**Query 4:** *"Which city had the highest profit?"*
> System: "New York City had the highest profit with $62,036.98 according to the summary."
> Ground truth (pandas): New York City — $62,036.98 ✓

**Query 5:** *"Which category is most profitable?"*
> System: "Technology is the most profitable category with a total profit of $145,454.95."
> Ground truth (pandas): Technology ✓

**Performance observations:**
- First streaming token arrives in approximately 1 second (llama3 on GTX 1650 Ti).
- Full 2-sentence answers complete in 3–6 seconds.
- Repeated identical questions return in under 10 ms (served from `lru_cache` — no LLM call).

*Note: Screenshots of the running Streamlit UI showing these query responses are included as separate attachments (screenshots/ folder in the repository).*

---

## 7. Challenges Encountered and Solutions

**Challenge 1 — Wrong answers for aggregate questions**
Initial implementation retrieved top-5 row-level documents. For "Which region had the highest profit?", the retrieval happened to return West region rows, giving a superficially correct but unreliable answer — on a different query phrasing it returned "Central". Root cause: 5 rows from 9,994 cannot represent full-dataset totals.
*Solution:* Pre-computed 26 pandas aggregate summary documents and injected them as first-class ChromaDB documents. Added keyword classifier to prioritise summaries for analytical queries. Raised top_k to 20.

**Challenge 2 — `TypeError` on ChromaDB singleton initialisation**
`chromadb.PersistentClient` is a factory function in chromadb's API, not a class. Using Python 3.10+ union type annotation `chromadb.PersistentClient | None` on a module-level variable raised `TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'`.
*Solution:* Removed type annotations from the `_client` and `_model` singleton module-level variables. They are initialised to `None` without annotation.

**Challenge 3 — 404 on `/ask/stream` after adding the endpoint**
The streaming endpoint was added to `app/main.py`, but a uvicorn process started earlier (with `--reload`) was still occupying port 8000. The new code was never loaded.
*Solution:* Identified and killed the old process; restarted uvicorn without `--reload` to ensure the updated module was loaded.

**Challenge 4 — Garbled dollar signs in Streamlit streaming**
Used `st.write_stream()` initially for real-time token display. Streamlit renders this output as Markdown, so `$24,051` was parsed as an opening LaTeX math delimiter — causing text to split character-by-character and appear as garbled symbols mid-stream.
*Solution:* Replaced `st.write_stream()` with a `placeholder = st.empty()` / `placeholder.text(full_response)` accumulator pattern. `placeholder.text()` renders plain text with no Markdown parsing.

**Challenge 5 — LLM hallucinating climate data, government policies, and non-US countries**
When asked "Which city sees most profits? and which country?", llama3 at default temperature generated multiple paragraphs about eco-friendly business districts, seasonal temperatures, carbon footprint policies, and tax incentives for green practices — none of which exist in the dataset. The model filled the gap in its retrieved context with general world knowledge.
*Solution:* Rewrote the system prompt with numbered MANDATORY RULES and an explicit FORBIDDEN section listing the specific hallucination categories observed. Set `temperature=0` to eliminate creative generation.

**Challenge 6 — Verbose multi-paragraph answers**
Even with a system prompt requesting conciseness, llama3 would produce 4–6 sentence answers with caveats ("It is worth noting that...", "However, this data only covers...").
*Solution:* Added a hard 2-sentence output cap to both the system and user prompts, added `num_predict=256` token cap in Ollama options, and ended the user prompt with "Stop after the answer."

---

## 8. Division of Work

| Area | Mohan Giri | Razib Hasan |
|---|---|---|
| Dataset sourcing and initial exploration | Primary | Reviewer |
| Data preprocessing pipeline (`load_data.py`, `preprocess.py`, `row_to_text.py`, `chunker.py`) | Primary | Reviewer |
| ChromaDB vector store setup and ingestion (`build_store.py`, `ingest.py`) | Primary | Reviewer |
| Pre-computed summary documents (`add_summaries.py`) | Primary | Reviewer |
| RAG pipeline — retrieval and context assembly (`query_store.py`, `pipeline.py`) | Primary | Reviewer |
| Analytical query classifier and document reordering | Primary | Reviewer |
| Prompt engineering and hallucination debugging | Joint | Joint |
| FastAPI backend (`app/main.py`) | Primary | Reviewer |
| Streamlit UI (`ui/app.py`) | Primary | Reviewer |
| Streaming implementation (`stream_rag`, `StreamingResponse`) | Primary | Reviewer |
| Performance optimisations (caching, singletons, GPU) | Primary | Reviewer |
| Testing and validation against ground truth | Joint | Joint |
| Debugging (TypeError, 404, dollar sign rendering) | Joint | Joint |
| Report writing | Joint | Joint |
| Presentation slides | Joint | Joint |

Both team members participated in architecture design discussions, debugging sessions, and validation of all query results against pandas ground truth.

---

## 9. AI Tool Usage Disclosure

*(This section does not count toward the page limit.)*

### 9.1 AI Tools Used

**GitHub Copilot (Claude Sonnet 4.6)** was used throughout the project via VS Code as an AI pair programmer. The following table documents which parts involved AI assistance and the nature of the prompts given:

| Component | AI Assistance | Example Prompts / Instructions Given |
|---|---|---|
| `row_to_text.py` | Code generation | "Write a function that converts a pandas row from the Superstore dataset into a natural language string including all fields" |
| `chunker.py` | Code generation | "Write a chunker that splits text into 500-character chunks" |
| `build_store.py` | Code generation | "Write a function to embed text chunks with SentenceTransformers and store them in ChromaDB persistently" |
| `add_summaries.py` | Code generation | "Pre-compute profit/sales aggregates by region, category, sub-category, state, city, segment, ship mode, and top customers using pandas groupby, and inject them as ChromaDB documents" |
| `pipeline.py` | Code generation + debugging | "Add an analytical keyword classifier that puts summary documents first in context; fix TypeError on PersistentClient type annotation" |
| `app/main.py` | Code generation | "Add a /ask/stream endpoint using FastAPI StreamingResponse with ollama.chat(stream=True)" |
| `ui/app.py` | Code generation + debugging | "Add a model selector sidebar; fix dollar sign rendering in streaming output" |
| `prompts.py` | Iterative prompt engineering | Multiple rounds: "Tighten the system prompt to prevent hallucination; add FORBIDDEN section; add numbered MANDATORY RULES; set temperature=0" |
| Report and slides | Drafting + refinement | "Write a technical report for this RAG project covering architecture, chunking, embeddings, LLM, challenges" |

### 9.2 Problems and Errors in AI-Generated Code

AI-generated code required debugging and correction in several cases:

1. **`TypeError` on ChromaDB type annotation:** The AI initially annotated the singleton variable as `_client: chromadb.PersistentClient | None = None`. This raised a runtime `TypeError` because `PersistentClient` is a factory function, not a class, and Python cannot construct a union type with a function. The AI did not know this chromadb API detail. We fixed it by removing the annotation.

2. **`st.write_stream()` rendering bug:** The AI suggested using `st.write_stream()` for the streaming display. This caused dollar signs in currency values (`$24,051`) to be interpreted as LaTeX math delimiters by Streamlit's Markdown renderer, producing garbled output. We identified the root cause and replaced it with a `placeholder.text()` accumulator.

3. **Hallucinating prompt instructions:** Early AI-drafted system prompts used polite language ("keep answers concise", "do not speculate"). These were consistently ignored by llama3. We had to iteratively redesign the prompt structure ourselves — identifying specific forbidden phrases and patterns from actual bad outputs before the AI could help encode them as explicit rules.

4. **ChromaDB batch size:** The AI initially generated a single `collection.add()` call for all 9,994 chunks. This crashed with a ChromaDB error (maximum batch size is 5,461). We identified the limit and added the `batch_size=5000` loop.

### 9.3 Running Results

The system was successfully deployed and all five validation queries returned correct answers matching pandas ground truth (see Section 6). The ingestion pipeline ran in approximately 20 seconds. The streaming API delivered first tokens in approximately 1 second. Screenshots of the running Streamlit UI, FastAPI `/docs` endpoint, and terminal output from `ingest.py` and `add_summaries.py` are included in the `screenshots/` folder of the repository.

### 9.4 Student Contributions Beyond AI-Generated Code

The following contributions represent work, decisions, and problem-solving that went beyond accepting AI output:

- **Architectural decision — two document types:** The insight that pure row-level retrieval is fundamentally insufficient for aggregate queries was our own. We designed the two-tier architecture (row chunks + pre-computed summaries in the same collection) and wrote the analytical classifier independently after observing wrong answers. The AI helped implement it, but the design was ours.

- **Prompt engineering through observed failure:** We ran the system, observed specific hallucination patterns (climate data, government policies, non-US countries), and iteratively diagnosed and fixed them. The FORBIDDEN section of the prompt was built directly from real bad outputs we collected during testing — not from AI suggestion.

- **Debugging `TypeError` and `st.write_stream()`:** Both bugs required understanding the internals of chromadb's API and Streamlit's rendering behaviour. We identified root causes independently before seeking AI help for the fix.

- **Validation methodology:** We cross-checked all system answers against pandas ground-truth aggregates computed independently in a notebook, confirming accuracy before documenting results.

- **Temperature=0 decision:** We decided to set `temperature=0` after observing that soft prompt guidelines were insufficient. This is an LLM inference configuration decision that required understanding of how temperature affects token sampling.

- **Integration and testing:** We integrated all components (data prep, vector store, pipeline, API, UI) manually, ran end-to-end tests, and resolved import path issues, port conflicts, and encoding errors that arose during integration — none of which appeared in AI-generated scaffolding.

---

*Report submitted: April 2026 — Mohan Giri & Razib Hasan*

