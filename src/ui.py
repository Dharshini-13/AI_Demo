# src/ui.py
import streamlit as st
import requests

st.title("IMDb Sentiment Analysis Demo")

# ------------------------
# 1. Fetch model accuracy
# ------------------------
API_URL = "http://127.0.0.1:8001"

try:
    health_response = requests.get(f"{API_URL}/")
    if health_response.status_code == 200:
        model_accuracy = health_response.json().get("model_accuracy", None)
        if model_accuracy is not None:
            st.sidebar.success(f"Model Accuracy: {model_accuracy*100:.2f}%")
        else:
            st.sidebar.warning("Accuracy not available from API.")
    else:
        st.sidebar.warning("API not reachable.")
except Exception as e:
    st.sidebar.warning(f"API connection error: {e}")

# ------------------------
# 2. User input
# ------------------------
review = st.text_area("Enter your review:")

# ------------------------
# 3. Prediction button
# ------------------------
if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        try:
            response = requests.post(f"{API_URL}/predict", json={"text": review})
            response.raise_for_status()
            result = response.json()
            sentiment = result.get("sentiment", "N/A")
            accuracy = result.get("model_accuracy", None)

            # Display prediction
            color = "green" if sentiment.lower() == "positive" else "red"
            st.markdown(f"<h3 style='color:{color}'>Prediction: {sentiment}</h3>", unsafe_allow_html=True)

            # Display accuracy
            if accuracy is not None:
                st.info(f"Model Accuracy: {accuracy*100:.2f}%")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
