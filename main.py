import streamlit as st

from pages import heart, lstm, cnn

page = st.sidebar.selectbox(
    "Select App",
    ["Heart Attack Classifier", "LSTM Predictor", "CNN Predictor"],
)

if page == "Heart Attack Classifier":
    heart.render()
elif page == "LSTM Predictor":
    lstm.render()
elif page == "CNN Predictor":
    cnn.render()
