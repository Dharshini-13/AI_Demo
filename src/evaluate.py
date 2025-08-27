import pandas as pd, joblib, json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate():
    model = joblib.load("model.pkl")
    df = pd.read_csv("data/test.csv")
    X = df['text']
    y = df['label']
    preds = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0))
    }
    with open("metrics/eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Eval metrics written to metrics/eval.json")

if __name__ == '__main__':
    evaluate()
