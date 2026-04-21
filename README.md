# RAG Sales Analysis

A question-answering system built on the Superstore sales dataset (2014-2017). It uses a local language model to answer business questions from retrieved sales data, without sending data to any external service.

---

## Stack

| Component   | Technology                                    |
|-------------|-----------------------------------------------|
| Embeddings  | `sentence-transformers/all-MiniLM-L6-v2`      |
| Vector store| ChromaDB (persistent, local)                 |
| LLM         | Ollama + `llama3` (local)                    |
| Backend API | FastAPI + Uvicorn                            |
| Frontend UI | Streamlit                                    |
| Dataset     | [Superstore Sales](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) (9,994 orders) |

---

## Architecture

```
User question
     |
     v
Streamlit UI  -->  POST /ask  -->  FastAPI
                                      |
                              Embed the question
                                      |
                              Query ChromaDB (top 20)
                                      |
                        ┌─────────────────────────┐
                        | Analytical query?        |
                        | Yes -> summaries first   |
                        | No  -> row chunks first  |
                        └─────────────────────────┘
                                      |
                              Build context string
                                      |
                              Ollama (llama3)
                                      |
                               Return answer
```

The vector store holds two types of documents:

- **Row-level chunks** - one natural-language sentence per order row, used for order-specific lookups.
- **Summary documents** - pre-computed aggregate statistics (totals, rankings, trends) over the full dataset, used for analytical questions.

A keyword classifier detects whether the question is analytical or order-specific and reorders the retrieved documents accordingly before passing them to the model.

---

## Why This Works

A standard RAG system that only embeds individual rows will fail on questions like "which region had the highest profit?" because no single row contains that answer. Retrieving 20 random rows from the West region and asking the model to summarise them leads to incorrect answers.

This system solves that by pre-computing the answers to common aggregate questions (totals, rankings, year-over-year figures) and storing them as first-class documents in the vector store. When an analytical question is asked, those summary documents are retrieved and placed first in the context, giving the model accurate, dataset-wide figures to work from.

---

## Summary Documents Included

| Document | Covers |
|---|---|
| Profit by region (all years) | Total profit and sales per region |
| Profit by region per year | 2014, 2015, 2016, 2017 separately |
| Sales by region per year | 2014, 2015, 2016, 2017 separately |
| Profit by category | All three product categories |
| Profit by sub-category | All 17 sub-categories, lowest to highest |
| Profit by state | Top 10 and bottom 10 states |
| Sales by state | Top 10 and bottom 10 states |
| Annual totals | Sales, profit, and orders per year |
| Profit and loss by city | Top 15 most profitable and 15 highest-loss cities |
| Customer segment performance | Consumer, Corporate, Home Office |
| Top 10 customers by profit | All four regions |
| Top 10 customers by sales | All four regions |
| Top 5 customers per region | Central, East, South, West |
| Sub-category sales by year | Year-over-year breakdown for trend questions |
| Ship mode performance | Profit, sales, and order count per mode |

---

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
│   └── main.py                     # FastAPI backend (POST /ask)
├── ui/
│   └── app.py                      # Streamlit frontend
├── ingest.py                       # One-time: build chroma_db/ from CSV
├── add_summaries.py                # One-time: inject aggregate summary docs
└── requirements.txt
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- NVIDIA GPU with CUDA 12.x (optional, but recommended for faster embeddings)

### 2. Clone and install

```bash
git clone https://github.com/girimohan/rag-sales-analysis.git
cd rag-sales-analysis
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 3. Get the dataset

Download from Kaggle and place the file at `data/raw/Sample - Superstore.csv`:

https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

### 4. Pull the model

```bash
ollama pull llama3
```

### 5. Build the vector store

Both scripts only need to be run once, or again if the dataset changes.

```bash
python ingest.py         # embeds all 9,994 rows into ChromaDB
python add_summaries.py  # injects 26 pre-computed aggregate summaries
```

### 6. Start the app

```bash
# Terminal 1 - API
uvicorn app.main:app --port 8000

# Terminal 2 - UI
streamlit run ui/app.py
```

Open http://localhost:8501 in your browser.

---

## Example Questions

**Profit and sales**
- Which region had the highest profit last year?
- Which category generated the most profit?
- Which sub-category has the lowest profit?
- What was the total profit in 2016?
- What is the profit trend from 2014 to 2017?

**Customers**
- Who are the top 10 customers by profit?
- Which customers are most profitable in the West region?

**Geography**
- Which state contributed the most to total profit?
- Which city saw the most loss?
- How do profits compare across regions?

**Order-level**
- What is the profit for order CA-2017-115805?
- Which region does this order belong to?

---

## Limitations

- The dataset covers 2014-2017 only. Questions about other time periods cannot be answered.
- Summary documents cover common aggregate dimensions. Questions about unusual combinations (e.g. profit by ship mode per state per year) may not have a matching summary and will fall back to row-level chunks, which can give incorrect totals.
- The model is llama3 running locally via Ollama. Response time depends on hardware. On CPU it is slow.
- The system does not retain conversation history. Each question is answered independently.
- Row-level chunks are a sample of the dataset, not the full data. Never use them to compute totals.

---

## Troubleshooting

**Ollama not responding**
Make sure Ollama is running: `ollama serve`. Then verify llama3 is pulled: `ollama list`.

**Port 8000 already in use**
Find and kill the process: `netstat -ano | findstr :8000`, then `Stop-Process -Id <PID> -Force`.

**chroma_db/ missing or empty**
Run `python ingest.py` then `python add_summaries.py` from the project root with the venv active.

**CUDA not detected**
Verify your PyTorch build: `python -c "import torch; print(torch.cuda.is_available())"`. If False, reinstall torch with the correct CUDA version from https://pytorch.org.

**Slow responses**
Embeddings default to CUDA if available. If running on CPU, ingestion and query will be slower. This does not affect answer quality.

---

## Future Enhancements

- Add conversation history so the model can handle follow-up questions.
- Support filtering by date range or region before retrieval.
- Replace the keyword classifier with a lightweight intent classifier trained on labelled examples.
- Add a feedback mechanism to flag incorrect answers and use them to improve prompts.
- Support uploading a different CSV without re-running the full pipeline manually.
- Add automated tests for the pipeline covering known question-answer pairs.

