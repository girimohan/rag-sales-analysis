import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_prep.load_data import load_superstore
from src.data_prep.preprocess import clean_columns
from src.data_prep.row_to_text import df_to_text_list
from src.data_prep.chunker import chunk_text_list
from src.vectorstore.build_store import build_vector_store

DATA_PATH = "data/raw/Sample - Superstore.csv"

print("Loading data...")
df = load_superstore(DATA_PATH)
df = clean_columns(df)
print(f"Loaded {len(df)} rows.")

print("Converting rows to text...")
texts = df_to_text_list(df)
print(f"Created {len(texts)} text entries.")

print("Chunking text...")
chunks = chunk_text_list(texts)
print(f"Created {len(chunks)} chunks.")

print("Building vector store...")
build_vector_store(chunks)
print("Done. Vector store saved to chroma_db/")
