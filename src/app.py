# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import json

class ReviewRequest(BaseModel):
    text: str

app = FastAPI(title="IMDb Sentiment Analysis API")

# ------------------------
# Load model + vectorizer + metrics
# ------------------------
MODEL_PATH = os.path.join("artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join("artifacts", "vectorizer.pkl")
METRICS_PATH = os.path.join("artifacts", "metrics.json")

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer not found. Run 'python src/train.py' first.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        model_accuracy = metrics.get("accuracy", None)
else:
    model_accuracy = None

# ------------------------
# Health check
# ------------------------
@app.get("/")
def read_root():
    return {
        "message": "IMDb Sentiment Analysis API is running!",
        "model_accuracy": model_accuracy
    }

# ------------------------
# Prediction
# ------------------------
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    try:
        X_input = vectorizer.transform([request.text])
        pred = model.predict(X_input)[0]
        sentiment = "positive" if pred == 1 else "negative"

        return {
            "review": request.text,
            "sentiment": sentiment,
            "model_accuracy": model_accuracy
        }
    except Exception as e:
        return {"error": str(e)}
