import pickle
import unicodedata
import numpy as np
import streamlit as st

try:
    from tensorflow.keras.models import load_model as keras_load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    keras_load_model = None
    pad_sequences = None

from config import LSTM_MODEL_PATH, TOKENIZER_PATH
from errors import ModelLoadError

MAX_LEN = 200


@st.cache_resource
def load_artifacts(model_path: str, tokenizer_path: str) -> tuple:
    """Load the LSTM model and tokenizer. Raises ModelLoadError on any failure."""
    if keras_load_model is None:
        raise ModelLoadError("lstm_model", model_path, ImportError("tensorflow is not installed"))

    try:
        model = keras_load_model(model_path)
    except Exception as e:
        raise ModelLoadError("lstm_model", model_path, e)

    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        raise ModelLoadError("tokenizer", tokenizer_path, e)

    return model, tokenizer


def validate_text(text: str) -> str | None:
    """Return an error string if text is empty or whitespace-only, None if valid.

    Strips both standard whitespace and Unicode control characters (Cc category)
    before checking if any meaningful content remains.
    """
    if not text:
        return "Input text cannot be empty or whitespace-only."
    # unicodedata.category returns 'Cc' for control chars; strip() misses some (e.g. DEL \x7f)
    cleaned = "".join(
        ch for ch in text
        if not (ch.isspace() or (len(ch) == 1 and unicodedata.category(ch) == "Cc"))
    )
    if not cleaned:
        return "Input text cannot be empty or whitespace-only."
    return None


def preprocess(text: str, tokenizer, max_len: int) -> np.ndarray:
    """Tokenize and pad text to shape (1, max_len)."""
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len)
    return padded


def render():
    st.title("LSTM Text Predictor")

    with st.spinner("Loading model and tokenizer..."):
        try:
            model, tokenizer = load_artifacts(LSTM_MODEL_PATH, TOKENIZER_PATH)
        except ModelLoadError as e:
            st.error(str(e))
            return

    text = st.text_area("Enter text for prediction")

    if st.button("Predict"):
        error = validate_text(text)
        if error:
            st.error(error)
            return

        try:
            features = preprocess(text, tokenizer, MAX_LEN)
            prediction = model.predict(features)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            st.success(f"Predicted class: {predicted_class}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
