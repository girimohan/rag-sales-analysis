"""
add_summaries.py
Compute aggregate statistics from the Superstore dataset and inject them
as summary documents into the existing ChromaDB collection so the RAG
pipeline can answer analytical/aggregation questions correctly.
"""
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

from src.data_prep.load_data import load_superstore as load_data
from src.data_prep.preprocess import clean_columns

DATA_PATH = "data/raw/Sample - Superstore.csv"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "superstore"


def build_summary_docs(df: pd.DataFrame) -> list[dict]:
    """Return a list of {'id': str, 'text': str} summary documents."""
    docs = []

    # --- ensure date columns are parsed ---
    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=False, errors="coerce")
    df["year"] = df["order_date"].dt.year

    # ── 1. Profit by Region (all time) ────────────────────────────────────────
    r = df.groupby("region").agg(
        total_profit=("profit", "sum"),
        total_sales=("sales", "sum"),
        total_orders=("order_id", "nunique"),
    ).round(2)
    top_region = r["total_profit"].idxmax()
    lines = ["Profit summary by region (all years):"]
    for region, row in r.sort_values("total_profit", ascending=False).iterrows():
        lines.append(
            f"  {region}: profit=${row['total_profit']:,.2f}, "
            f"sales=${row['total_sales']:,.2f}, orders={row['total_orders']}"
        )
    lines.append(f"Highest-profit region overall: {top_region}.")
    docs.append({"id": "summary_profit_by_region", "text": "\n".join(lines)})

    # ── 2. Profit by Region per Year ──────────────────────────────────────────
    ry = df.groupby(["year", "region"])["profit"].sum().round(2).reset_index()
    for yr, grp in ry.groupby("year"):
        top = grp.loc[grp["profit"].idxmax(), "region"]
        lines = [f"Profit by region for year {yr}:"]
        for _, row in grp.sort_values("profit", ascending=False).iterrows():
            lines.append(f"  {row['region']}: ${row['profit']:,.2f}")
        lines.append(f"Highest-profit region in {yr}: {top}.")
        docs.append({"id": f"summary_region_profit_{yr}", "text": "\n".join(lines)})

    # ── 3. Profit by Category (all time) ──────────────────────────────────────
    ca = df.groupby("category").agg(
        total_profit=("profit", "sum"),
        total_sales=("sales", "sum"),
    ).round(2)
    top_cat = ca["total_profit"].idxmax()
    lines = ["Profit and sales by product category (all years):"]
    for cat, row in ca.sort_values("total_profit", ascending=False).iterrows():
        lines.append(f"  {cat}: profit=${row['total_profit']:,.2f}, sales=${row['total_sales']:,.2f}")
    lines.append(f"Most profitable category: {top_cat}.")
    docs.append({"id": "summary_profit_by_category", "text": "\n".join(lines)})

    # ── 4. Profit by Sub-Category (top 10) ────────────────────────────────────
    sc = df.groupby("sub_category")["profit"].sum().round(2).sort_values(ascending=False)
    lines = ["Top 10 sub-categories by profit (all years):"]
    for sub, profit in sc.head(10).items():
        lines.append(f"  {sub}: ${profit:,.2f}")
    lines.append(f"Most profitable sub-category: {sc.index[0]}.")
    docs.append({"id": "summary_profit_by_subcategory", "text": "\n".join(lines)})

    # ── 5. Sales & Profit by Year ──────────────────────────────────────────────
    yt = df.groupby("year").agg(
        total_sales=("sales", "sum"),
        total_profit=("profit", "sum"),
        total_orders=("order_id", "nunique"),
    ).round(2)
    lines = ["Annual sales and profit summary:"]
    for yr, row in yt.iterrows():
        lines.append(
            f"  {yr}: sales=${row['total_sales']:,.2f}, "
            f"profit=${row['total_profit']:,.2f}, orders={row['total_orders']}"
        )
    docs.append({"id": "summary_annual_totals", "text": "\n".join(lines)})

    # ── 6. Profit by State (top 10 and bottom 10) ────────────────────────────
    st = df.groupby("state")["profit"].sum().round(2).sort_values(ascending=False)
    lines = ["Top 10 states by profit:"]
    for state, profit in st.head(10).items():
        lines.append(f"  {state}: ${profit:,.2f}")
    lines.append("Bottom 10 states by profit (least profitable / losses):")
    for state, profit in st.tail(10).items():
        lines.append(f"  {state}: ${profit:,.2f}")
    docs.append({"id": "summary_profit_by_state", "text": "\n".join(lines)})

    # ── 7. Top 10 customers by profit ────────────────────────────────────────
    cust = df.groupby("customer_name")["profit"].sum().round(2).sort_values(ascending=False)
    lines = ["Top 10 customers by total profit:"]
    for cname, profit in cust.head(10).items():
        lines.append(f"  {cname}: ${profit:,.2f}")
    docs.append({"id": "summary_top_customers", "text": "\n".join(lines)})

    # ── 8. Segment performance ───────────────────────────────────────────────
    seg = df.groupby("segment").agg(
        total_profit=("profit", "sum"),
        total_sales=("sales", "sum"),
    ).round(2)
    lines = ["Performance by customer segment:"]
    for segment, row in seg.sort_values("total_profit", ascending=False).iterrows():
        lines.append(f"  {segment}: profit=${row['total_profit']:,.2f}, sales=${row['total_sales']:,.2f}")
    docs.append({"id": "summary_by_segment", "text": "\n".join(lines)})

    return docs


def inject_summaries():
    print("Loading data...")
    df = clean_columns(load_data(DATA_PATH))

    print("Computing summaries...")
    docs = build_summary_docs(df)

    print(f"Built {len(docs)} summary documents.")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    print("Embedding summaries...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    embeddings = model.encode([d["text"] for d in docs], show_progress_bar=True).tolist()

    # Upsert so re-running is idempotent
    collection.upsert(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        embeddings=embeddings,
    )

    print(f"Done. Injected {len(docs)} summary documents into '{COLLECTION_NAME}'.")
    for d in docs:
        print(f"  + {d['id']}")


if __name__ == "__main__":
    inject_summaries()
