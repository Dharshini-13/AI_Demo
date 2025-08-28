# src/train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# ------------------------
# Paths
# ------------------------
RAW_DATA_PATH = "data/raw/IMDB_Dataset.csv"
PROCESSED_DIR = "data/processed"
ARTIFACT_DIR = "artifacts"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.json")

# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv(RAW_DATA_PATH)
X = df['review']
y = df['sentiment'].apply(lambda x: 1 if x.lower() == "positive" else 0)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save processed data
pd.DataFrame({"review": X_train, "sentiment": y_train}).to_csv(TRAIN_PATH, index=False)
pd.DataFrame({"review": X_test, "sentiment": y_test}).to_csv(TEST_PATH, index=False)

# ------------------------
# Vectorize
# ------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------
# Train model
# ------------------------
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# ------------------------
# Evaluate
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
# Save artifacts
# ------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f)
