# RAG Sales Analysis

A Retrieval-Augmented Generation (RAG) system for querying the Superstore sales dataset (2014–2017) using natural language. Ask business questions and get answers grounded in real data — no hallucination.

## Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (CUDA) |
| Vector Store | ChromaDB (persistent, local) |
| LLM | Ollama + `llama3` (local) |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Dataset | [Superstore Sales](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) — 9,994 orders |

## Project Structure

```
rag-sales-analysis/
├── data/
│   ├── raw/                        # Source CSV (git-ignored)
│   └── processed/                  # Cleaned data (git-ignored)
├── notebooks/
│   └── 01_exploration.ipynb        # EDA: distributions, trends, charts
├── src/
│   ├── data_prep/                  # load_data, preprocess, row_to_text, chunker
│   ├── vectorstore/                # build_store, query_store (ChromaDB)
│   └── rag/                        # pipeline, prompts
├── app/
│   └── main.py                     # FastAPI backend  →  POST /ask
├── ui/
│   └── app.py                      # Streamlit frontend
├── ingest.py                       # One-time: build chroma_db/ from CSV
├── add_summaries.py                # One-time: inject aggregate summary docs
└── requirements.txt
```

## How It Works

1. **Ingestion** — Each row of the CSV is converted to a natural-language sentence, chunked, embedded, and stored in ChromaDB (`ingest.py`).
2. **Summaries** — Pre-computed aggregate statistics (profit/sales by region, category, state, segment, year, city, customer) are embedded and stored alongside row-level chunks (`add_summaries.py`). This is what makes analytical questions work correctly.
3. **Query** — A question is embedded, top-20 similar documents are retrieved, summary docs are prioritised for analytical questions via a keyword classifier, and llama3 generates a grounded answer.

## Getting Started

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- (Optional) NVIDIA GPU with CUDA 12.x for faster embeddings

### 2. Clone and set up

```bash
git clone https://github.com/girimohan/rag-sales-analysis.git
cd rag-sales-analysis
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 3. Get the dataset

Download from Kaggle and place at `data/raw/Sample - Superstore.csv`:

> https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

### 4. Pull the LLM

```bash
ollama pull llama3
```

### 5. Build the vector store

```bash
# Build row-level chunks (run once)
python ingest.py

# Inject aggregate summaries (run once, or after data changes)
python add_summaries.py
```

### 6. Run the app

```bash
# Terminal 1 — Backend
uvicorn app.main:app --port 8000

# Terminal 2 — Frontend
streamlit run ui/app.py
```

Open **http://localhost:8501** to chat with your data.

---

## Questions the Chatbot Can Answer

### Profit & Sales Performance
- Which region had the highest profit last year?
- Which region had the highest sales in 2015?
- Which category generated the most profit?
- Which sub-category has the lowest profit?
- Which state contributed the most to total profit?
- What was the total profit in 2014 / 2015 / 2016 / 2017?
- What is the profit trend over the years?

### Customer Insights
- Who are the top 10 customers by profit?
- Which customer generated the highest sales?
- Which customers are most profitable in the West region?

### Product & Category Insights
- Which product category has the highest sales?
- Which sub-category is growing the fastest?
- How do Technology and Furniture compare in profitability?

### Regional & Geographic Insights
- Which region performs best overall?
- Which city saw the most loss?
- Which state has the highest sales?
- How do profits compare across regions?

### Order-Level Questions
- What is the profit for Order ID CA-2017-115805?
- Which region does this order belong to?
- What category is this product in?

---

## Notes

- `chroma_db/` is git-ignored. Run `ingest.py` and `add_summaries.py` to rebuild locally.
- CUDA is used automatically if available; falls back to CPU.
- The RAG pipeline uses a keyword classifier to route analytical questions to summary documents and order-specific questions to row-level chunks.

