import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    df = pd.read_csv("data/reviews.csv")
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    train = pd.DataFrame({'text': X_train, 'label': y_train})
    test = pd.DataFrame({'text': X_test, 'label': y_test})
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    print("Prepared data: data/train.csv and data/test.csv")

if __name__ == '__main__':
    prepare_data()
