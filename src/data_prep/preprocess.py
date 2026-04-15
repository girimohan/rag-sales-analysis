import pandas as pd


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to lowercase with underscores."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df
