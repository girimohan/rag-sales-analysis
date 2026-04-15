import pandas as pd


def load_superstore(path: str) -> pd.DataFrame:
    """Load the Superstore CSV dataset from the given file path."""
    df = pd.read_csv(path, encoding="windows-1252")
    return df
