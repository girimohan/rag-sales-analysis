# RAG Sales Analysis

A local question-answering system for the Superstore sales dataset (2014-2017). Ask business questions in plain English and get answers backed by pre-computed aggregates. No data leaves your machine.

## Stack

| Component    | Technology                               |
|--------------|------------------------------------------|
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persistent, local)             |
| LLM          | Ollama + `llama3`                        |
| Backend API  | FastAPI + Uvicorn                        |
| Frontend UI  | Streamlit                                |
| Dataset      | [Superstore Sales](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) (9,994 orders, 2014-2017) |

## How It Works

Questions are embedded and matched against two document types in ChromaDB:

- **Summary documents:** pre-computed dataset-wide aggregates (totals, rankings, trends). Used for analytical questions.
- **Row-level chunks:** one sentence per order row. Used for order-specific lookups.

An analytical keyword classifier routes the query to the right document type before passing context to the LLM.

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com/) running locally.

```bash
git clone https://github.com/girimohan/rag-sales-analysis.git
cd rag-sales-analysis
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Download the dataset and place it at `data/raw/Sample - Superstore.csv`:  
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

```bash
ollama pull llama3

python ingest.py          # build ChromaDB from CSV (run once)
python add_summaries.py   # inject aggregate summaries (run once)
```

```bash
# Terminal 1
uvicorn app.main:app --port 8000

# Terminal 2
streamlit run ui/app.py
```

Open http://localhost:8501.

## Example Questions

- Which region had the highest profit?
- Which category is most profitable?
- Which sub-category has the lowest profit?
- What is the sales trend from 2014 to 2017?
- Which months show the highest sales?
- Compare Technology vs Furniture sales.
- Who are the top 10 customers by profit?
- Which city had the most losses?
- What is the profit for order CA-2017-115805?

## Limitations

- Dataset covers 2014-2017 only.
- Each question is answered independently with no conversation history.
- Questions about unusual dimension combinations may fall back to row-level chunks and give inaccurate totals.
- Response speed depends on hardware; llama3 on CPU is slow.

