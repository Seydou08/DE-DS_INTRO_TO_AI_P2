from src.loader import load_data
from src.cleaner import clean_data
from setup_data import setup_data
from src.features import create_time_features, encode
from src.preprocess import save_procesed_data, print_summary


    
RAW_DATA_PATH = "data/raw/US_Accidents_March23.csv"
OUTPUT_PATH = "data/processed/cleaned_data.csv"

def main():
    print("Starting preprocessing pipeline:")

    df = load_data(RAW_DATA_PATH)
    print_summary(df, "Raw Data")

    df = clean_data(df)
    print_summary(df, "Cleaned Data")

    df = create_time_features(df)
    print_summary(df, "Data with Time Features")

    df = encode(df)
    print_summary(df, "Encoded Data")

    save_procesed_data(df, OUTPUT_PATH)

    print("Pipeline completed")


if __name__ == "__main__":
    main()

