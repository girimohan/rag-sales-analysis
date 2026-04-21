import pandas as pd


def row_to_text(row: pd.Series) -> str:
    """Convert a single Superstore row into a natural-language string."""
    def val(key):
        v = row.get(key)
        if pd.isna(v) if not isinstance(v, str) else v.strip() == "":
            return "N/A"
        return str(v).strip()

    return (
        f"Order {val('order_id')} was placed on {val('order_date')} "
        f"with ship mode '{val('ship_mode')}'. "
        f"Customer '{val('customer_name')}' ({val('segment')} segment) "
        f"is located in {val('city')}, {val('state')}, {val('region')} region. "
        f"They ordered '{val('product_name')}' "
        f"(Category: {val('category')}, Sub-Category: {val('sub_category')}). "
        f"Quantity: {val('quantity')}, Sales: ${val('sales')}, "
        f"Discount: {val('discount')}, Profit: ${val('profit')}."
    )


def df_to_text_list(df: pd.DataFrame) -> list[str]:
    """Apply row_to_text to every row and return a list of non-empty strings."""
    results = []
    for _, row in df.iterrows():
        text = row_to_text(row)
        if text.strip():
            results.append(text)
    return results
