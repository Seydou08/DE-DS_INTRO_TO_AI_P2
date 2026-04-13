import pandas as pd
from src.utils import timer

# Columns that are critical for modeling — rows missing these are dropped
REQUIRED_COLUMNS = ['Severity', 'Start_Time']

# Columns with too many nulls to fill — dropped entirely if missing threshold exceeded
HIGH_NULL_THRESHOLD = 0.5  # drop column if more than 50% null


@timer
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {before - len(df)}")

    # 2. Drop rows missing critical columns
    df = df.dropna(subset=REQUIRED_COLUMNS)
    print(f"  Rows after dropping missing Severity/Start_Time: {len(df)}")

    # 3. Drop columns that are mostly empty (over 50% null)
    null_frac = df.isnull().mean()
    cols_to_drop = null_frac[null_frac > HIGH_NULL_THRESHOLD].index.tolist()
    if cols_to_drop:
        print(f"  Dropping high-null columns (>{HIGH_NULL_THRESHOLD*100:.0f}% null): {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # 4. Fill remaining nulls in numeric columns with median
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df