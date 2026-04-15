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

### 3. Add your data

Place the Superstore CSV file in `data/raw/`.

### 4. Configure environment variables (optional)

Copy `.env.example` to `.env` and adjust paths or model names as needed.

### 5. Launch the notebook

```bash
jupyter notebook notebooks/01_exploration.ipynb
```
