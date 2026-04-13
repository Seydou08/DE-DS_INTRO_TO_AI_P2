import pandas as pd
from src.loader import load_data
from src.cleaner import clean_data
from src.features import create_time_features, encode
from src.preprocess import save_procesed_data, print_summary
from src.utils import sample_data

RAW_DATA_PATH  = "data/raw/US_Accidents_March23.csv"
OUTPUT_PATH    = "data/processed/cleaned_data.csv"

# Set to True during development to run on 500k rows instead of 2.8M
USE_SAMPLE = False


def handle_class_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prints severity class distribution and computes class weights.
    Weights are saved alongside the dataset so the Data Scientist
    can pass them directly into sklearn models via class_weight param.
    """
    print("\n--- Class Imbalance Check ---")
    dist = df['Severity'].value_counts().sort_index()
    print(dist)

    # Compute weight for each class: inversely proportional to frequency
    total = len(df)
    n_classes = dist.shape[0]
    weights = {}
    for severity, count in dist.items():
        weights[severity] = round(total / (n_classes * count), 4)

    print("\nSuggested class weights for modeling:")
    for k, v in weights.items():
        print(f"  Severity {k}: {v}")

    # Save weights to a small file the DS can load
    import json, os
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/class_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    print("  Class weights saved to: data/processed/class_weights.json")

    return df


def main():
    print("Starting preprocessing pipeline:\n")

    # Step 1: Load
    df = load_data(RAW_DATA_PATH)
    print_summary(df, "Raw Data")

    # Step 2: Optional sample for faster dev runs
    if USE_SAMPLE:
        from src.utils import sample_data
        df = sample_data(df, n=500_000)

    # Step 3: Clean
    df = clean_data(df)
    print_summary(df, "Cleaned Data")

    # Step 4: Time features
    df = create_time_features(df)
    print_summary(df, "Data with Time Features")

    # Step 5: Encode categorical columns
    df = encode(df)
    print_summary(df, "Encoded Data")

    # Step 6: Handle class imbalance
    df = handle_class_imbalance(df)

    # Step 7: Save
    save_procesed_data(df, OUTPUT_PATH)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()