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

    # 1. Profit by Region (all time)
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

    # 2. Profit by Region per Year
    ry = df.groupby(["year", "region"])["profit"].sum().round(2).reset_index()
    for yr, grp in ry.groupby("year"):
        top = grp.loc[grp["profit"].idxmax(), "region"]
        lines = [f"Profit by region for year {yr}:"]
        for _, row in grp.sort_values("profit", ascending=False).iterrows():
            lines.append(f"  {row['region']}: ${row['profit']:,.2f}")
        lines.append(f"Highest-profit region in {yr}: {top}.")
        docs.append({"id": f"summary_region_profit_{yr}", "text": "\n".join(lines)})

    # 3. Profit by Category (all time)
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

    # 4. Profit by Sub-Category (top 10)
    sc = df.groupby("sub_category")["profit"].sum().round(2).sort_values(ascending=False)
    lines = ["Top 10 sub-categories by profit (all years):"]
    for sub, profit in sc.head(10).items():
        lines.append(f"  {sub}: ${profit:,.2f}")
    lines.append(f"Most profitable sub-category: {sc.index[0]}.")
    docs.append({"id": "summary_profit_by_subcategory", "text": "\n".join(lines)})

    # 5. Sales and Profit by Year
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

    # 6. Profit by State (top 10 and bottom 10)
    st = df.groupby("state")["profit"].sum().round(2).sort_values(ascending=False)
    lines = ["Top 10 states by profit:"]
    for state, profit in st.head(10).items():
        lines.append(f"  {state}: ${profit:,.2f}")
    lines.append("Bottom 10 states by profit (least profitable / losses):")
    for state, profit in st.tail(10).items():
        lines.append(f"  {state}: ${profit:,.2f}")
    docs.append({"id": "summary_profit_by_state", "text": "\n".join(lines)})

    # 7. Top 10 customers by profit
    cust = df.groupby("customer_name")["profit"].sum().round(2).sort_values(ascending=False)
    lines = ["Top 10 customers by total profit:"]
    for cname, profit in cust.head(10).items():
        lines.append(f"  {cname}: ${profit:,.2f}")
    docs.append({"id": "summary_top_customers", "text": "\n".join(lines)})

    # 8. Segment performance
    seg = df.groupby("segment").agg(
        total_profit=("profit", "sum"),
        total_sales=("sales", "sum"),
    ).round(2)
    lines = ["Performance by customer segment:"]
    for segment, row in seg.sort_values("total_profit", ascending=False).iterrows():
        lines.append(f"  {segment}: profit=${row['total_profit']:,.2f}, sales=${row['total_sales']:,.2f}")
    docs.append({"id": "summary_by_segment", "text": "\n".join(lines)})

    # 9. City-level profit and loss
    city = df.groupby(["city", "state"]).agg(
        total_profit=("profit", "sum"),
        total_sales=("sales", "sum"),
    ).round(2)
    city_sorted = city.sort_values("total_profit")

    # Bottom 15 - cities with most losses
    loss_cities = city_sorted[city_sorted["total_profit"] < 0].head(15)
    worst_city = city_sorted.index[0]  # (city, state) tuple
    lines = ["Cities with the most losses (negative profit):"]
    for (cname, state), row in loss_cities.iterrows():
        lines.append(f"  {cname}, {state}: loss=${row['total_profit']:,.2f} (sales=${row['total_sales']:,.2f})")
    lines.append(
        f"City with the single highest loss: {worst_city[0]}, {worst_city[1]} "
        f"with profit=${city_sorted.iloc[0]['total_profit']:,.2f}."
    )
    docs.append({"id": "summary_loss_by_city", "text": "\n".join(lines)})

    # Top 15 - most profitable cities
    top_cities = city.sort_values("total_profit", ascending=False).head(15)
    lines = ["Most profitable cities:"]
    for (cname, state), row in top_cities.iterrows():
        lines.append(f"  {cname}, {state}: profit=${row['total_profit']:,.2f} (sales=${row['total_sales']:,.2f})")
    docs.append({"id": "summary_profit_by_city", "text": "\n".join(lines)})

    # 10. Ship mode performance
    ship = df.groupby("ship_mode").agg(
        total_profit=("profit", "sum"),
        total_sales=("sales", "sum"),
        total_orders=("order_id", "nunique"),
    ).round(2)
    lines = ["Performance by shipping mode:"]
    for mode, row in ship.sort_values("total_profit", ascending=False).iterrows():
        lines.append(
            f"  {mode}: profit=${row['total_profit']:,.2f}, "
            f"sales=${row['total_sales']:,.2f}, orders={row['total_orders']}"
        )
    docs.append({"id": "summary_by_ship_mode", "text": "\n".join(lines)})

    # 11. Sales by Region per Year
    ry_sales = df.groupby(["year", "region"])["sales"].sum().round(2).reset_index()
    for yr, grp in ry_sales.groupby("year"):
        top = grp.loc[grp["sales"].idxmax(), "region"]
        lines = [f"Sales by region for year {yr}:"]
        for _, row in grp.sort_values("sales", ascending=False).iterrows():
            lines.append(f"  {row['region']}: ${row['sales']:,.2f}")
        lines.append(f"Highest-sales region in {yr}: {top}.")
        docs.append({"id": f"summary_region_sales_{yr}", "text": "\n".join(lines)})

    # 12. Sales by State (top 10 and bottom 10)
    st_sales = df.groupby("state")["sales"].sum().round(2).sort_values(ascending=False)
    lines = ["Top 10 states by total sales:"]
    for state, sales in st_sales.head(10).items():
        lines.append(f"  {state}: ${sales:,.2f}")
    lines.append("Bottom 10 states by total sales:")
    for state, sales in st_sales.tail(10).items():
        lines.append(f"  {state}: ${sales:,.2f}")
    docs.append({"id": "summary_sales_by_state", "text": "\n".join(lines)})

    # 13. Top 10 customers by sales
    cust_sales = df.groupby("customer_name")["sales"].sum().round(2).sort_values(ascending=False)
    lines = ["Top 10 customers by total sales:"]
    for cname, sales in cust_sales.head(10).items():
        lines.append(f"  {cname}: ${sales:,.2f}")
    docs.append({"id": "summary_top_customers_by_sales", "text": "\n".join(lines)})

    # 14. Full sub-category profit (all, including bottom)
    sc_full = df.groupby("sub_category").agg(
        total_profit=("profit", "sum"),
        total_sales=("sales", "sum"),
    ).round(2).sort_values("total_profit")
    lines = ["All sub-categories by profit (lowest to highest):"]
    for sub, row in sc_full.iterrows():
        lines.append(f"  {sub}: profit=${row['total_profit']:,.2f}, sales=${row['total_sales']:,.2f}")
    lines.append(f"Least profitable sub-category: {sc_full.index[0]} (${sc_full.iloc[0]['total_profit']:,.2f}).")
    lines.append(f"Most profitable sub-category: {sc_full.index[-1]} (${sc_full.iloc[-1]['total_profit']:,.2f}).")
    docs.append({"id": "summary_subcategory_full", "text": "\n".join(lines)})

    # 15. Top 5 customers by profit per region
    for region, rdf in df.groupby("region"):
        rc = rdf.groupby("customer_name")["profit"].sum().round(2).sort_values(ascending=False).head(5)
        lines = [f"Top 5 most profitable customers in {region} region:"]
        for cname, profit in rc.items():
            lines.append(f"  {cname}: ${profit:,.2f}")
        docs.append({"id": f"summary_top_customers_{region.lower()}", "text": "\n".join(lines)})

    # 16. Sub-category year-over-year sales (growth signal)
    sc_yr = df.groupby(["year", "sub_category"])["sales"].sum().round(2).unstack(fill_value=0)
    lines = ["Sub-category sales by year (for trend analysis):"]
    for sub in sc_yr.columns:
        row_vals = "  |  ".join(f"{yr}: ${sc_yr.loc[yr, sub]:,.0f}" for yr in sc_yr.index)
        lines.append(f"  {sub}: {row_vals}")
    docs.append({"id": "summary_subcategory_sales_by_year", "text": "\n".join(lines)})

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
        metadatas=[{"type": "summary"} for _ in docs],
    )

    print(f"Done. Injected {len(docs)} summary documents into '{COLLECTION_NAME}'.")
    for d in docs:
        print(f"  + {d['id']}")


if __name__ == "__main__":
    inject_summaries()
