import time
import functools
import pandas as pd


def check_columns(df: pd.DataFrame, columns: list) -> list:
    """
    Returns only the columns that actually exist in the dataframe.
    Logs a warning for any that are missing.
    """
    existing = []
    for col in columns:
        if col in df.columns:
            existing.append(col)
        else:
            print(f"  [WARNING] Column '{col}' not found in dataframe — skipping.")
    return existing


def sample_data(df: pd.DataFrame, n: int = 500_000, random_state: int = 42) -> pd.DataFrame:
    """
    Returns a random sample of n rows if the dataframe is larger than n.
    Useful for faster testing during development on the 2.8M row dataset.
    """
    if len(df) > n:
        print(f"  [SAMPLE] Dataset has {len(df):,} rows — sampling down to {n:,} for faster processing.")
        return df.sample(n=n, random_state=random_state).reset_index(drop=True)
    return df


def timer(func):
    """
    Decorator that logs how long each pipeline step takes.
    Usage: add @timer above any function definition.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  [{func.__name__}] completed in {elapsed:.2f}s")
        return result
    return wrapper