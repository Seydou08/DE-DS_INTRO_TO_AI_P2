import os
import pandas as pd

def save_procesed_data(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

def print_summary(df: pd.DataFrame, stage_name: str):
    print(f"\n---{stage_name}---")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())


    if 'Severity' in df.columns:
        print("\nSeverity Distribution:")
        print(df['Severity'].value_counts(dropna=False).sort_index())

    print("\nMissing Values (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))