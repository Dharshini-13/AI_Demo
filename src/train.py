import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------
# 1. MLflow + DagsHub init
# ------------------------
import dagshub
dagshub.init(repo_owner="dharsh", repo_name="imdb-response-demo", mlflow=True)
mlflow.set_experiment("IMDB_Sentiment_Analysis")

# ------------------------
# 2. Load dataset
# ------------------------
df = pd.read_csv("data/raw/IMDB_Dataset.csv")
X = df['review']
y = df['sentiment'].apply(lambda x: 1 if x.lower() == 'positive' else 0)

# ------------------------
# 3. Split train/test
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Make processed folder
os.makedirs("data/processed", exist_ok=True)

# Save CSVs
pd.DataFrame({"review": X_train, "sentiment": y_train}).to_csv("data/processed/train.csv", index=False)
pd.DataFrame({"review": X_test, "sentiment": y_test}).to_csv("data/processed/test.csv", index=False)

# ------------------------
# 4. Vectorize
# ------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------
# 5. Train model
# ------------------------
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# ------------------------
# 6. Evaluate
# ------------------------
y_pred = model.predict(X_test_vec)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
}
print("✅ Training complete. Metrics:", metrics)

# ------------------------
# 7. Save artifacts
# ------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.pkl")
joblib.dump(vectorizer, "artifacts/vectorizer.pkl")

# Save metrics
import json
with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f)

# ------------------------
# 8. Log to MLflow
# ------------------------
with mlflow.start_run():
    mlflow.log_params({"model": "LogisticRegression", "vectorizer": "Tfidf"})
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    mlflow.log_artifact("artifacts/model.pkl")
    mlflow.log_artifact("artifacts/vectorizer.pkl")
    mlflow.log_artifact("artifacts/metrics.json")
