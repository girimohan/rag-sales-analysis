# RAG Sales Analysis

A Retrieval-Augmented Generation (RAG) system for querying and analyzing the Superstore sales dataset using natural language.

## Project Structure

```
rag-sales-analysis/
├── data/
│   ├── raw/          # Source CSV files (git-ignored)
│   └── processed/    # Cleaned / chunked data (git-ignored)
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── data_prep/    # Loading, cleaning, chunking
│   ├── vectorstore/  # ChromaDB build & query helpers
│   ├── rag/          # Pipeline and prompt templates
│   └── utils/        # Config and logging
├── demo/
│   └── screenshots/
├── report/
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **macOS / Linux:** `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the data

The `data/` folder is not included in this repo.
Download the CSV from Kaggle and place it in `data/raw/`:

1. Go to https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
2. Click **Download** and save the file.
3. Copy `Sample - Superstore.csv` into the `data/raw/` folder.

### 4. Launch the notebook

```bash
jupyter notebook notebooks/01_exploration.ipynb
```
