import streamlit as st
import requests
import json

st.title("IMDb Sentiment Analysis Demo")

# Load accuracy from metrics file
try:
    with open("metrics/accuracy.json", "r") as f:
        metrics = json.load(f)
        st.sidebar.success(f"Model Accuracy: {metrics['accuracy']*100:.2f}%")
except:
    st.sidebar.warning("Accuracy not available. Train the model first.")

review = st.text_area("Enter your review:")

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8001/predict", json={"text": review})
    if response.status_code == 200:
        result = response.json()
        sentiment = result["sentiment"]
        st.write(f"### Prediction: {sentiment}")
    else:
        st.error("Error in prediction API")
