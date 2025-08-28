import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
RAW_DATA_PATH = "data/raw/IMDB_Dataset.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

# Config
TEST_SIZE = 0.25
RANDOM_STATE = 42

def prepare_data():
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)

    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Use correct column names
    X = df["review"]
    y = df["sentiment"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save processed data
    pd.DataFrame({'review': X_train, 'sentiment': y_train}).to_csv(TRAIN_PATH, index=False)
    pd.DataFrame({'review': X_test, 'sentiment': y_test}).to_csv(TEST_PATH, index=False)

    print(f"Prepared data: {TRAIN_PATH} and {TEST_PATH}")

if __name__ == '__main__':
    prepare_data()
