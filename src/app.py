# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Define request body
class ReviewRequest(BaseModel):
    text: str

# Initialize FastAPI
app = FastAPI(title="IMDb Sentiment Analysis API")

# Load model + vectorizer
model_path = os.path.join("models", "model.pkl")
vectorizer_path = os.path.join("models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.get("/")
def read_root():
    return {"message": "IMDb Sentiment Analysis API is running!"}

@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    try:
        # Transform input
        X = vectorizer.transform([request.text])
        prediction = model.predict(X)[0]

        sentiment = "positive" if prediction == 1 else "negative"

        return {
            "review": request.text,
            "sentiment": sentiment
        }
    except Exception as e:
        return {"error": str(e)}
