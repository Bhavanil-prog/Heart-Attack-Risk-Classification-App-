import numpy as np
import streamlit as st

try:
    from tensorflow.keras.models import load_model as keras_load_model
except ImportError:
    keras_load_model = None

from config import CNN_MODEL_PATH
from errors import ModelLoadError


@st.cache_resource
def load_model(path: str):
    """Load the CNN .h5 model. Raises ModelLoadError on any failure."""
    if keras_load_model is None:
        raise ModelLoadError("cnn_model", path, ImportError("tensorflow is not installed"))

    try:
        model = keras_load_model(path)
    except Exception as e:
        raise ModelLoadError("cnn_model", path, e)

    return model


def validate_input(input_data: np.ndarray, expected_shape: tuple) -> str | None:
    """Return an error string if input is invalid, None if valid.

    Checks that:
    - input_data dtype is numeric
    - input_data shape matches expected_shape (ignoring the first/batch dimension)
    """
    if not np.issubdtype(input_data.dtype, np.number):
        return f"Input dtype '{input_data.dtype}' is not numeric."

    # expected_shape is e.g. (None, 28, 28, 1) — ignore the first (batch) dimension
    model_spatial_shape = expected_shape[1:]
    if input_data.shape != model_spatial_shape:
        return (
            f"Input shape {input_data.shape} does not match "
            f"expected shape {model_spatial_shape}."
        )

    return None


def preprocess(input_data: np.ndarray, expected_shape: tuple) -> np.ndarray:
    """Reshape input to match model expectations by adding a batch dimension.

    Returns a numpy array with shape (1, *expected_shape[1:]).
    """
    model_spatial_shape = expected_shape[1:]
    return input_data.reshape((1,) + model_spatial_shape)


def render():
    st.title("CNN Predictor")

    with st.spinner("Loading model..."):
        try:
            model = load_model(CNN_MODEL_PATH)
        except ModelLoadError as e:
            st.error(str(e))
            return

    expected_shape = model.input_shape  # e.g. (None, 28, 28, 1)

    uploaded_file = st.file_uploader("Upload a NumPy .npy file", type=["npy"])

    if st.button("Predict"):
        if uploaded_file is None:
            st.error("Please upload a .npy file before predicting.")
            return

        try:
            input_data = np.load(uploaded_file)
        except Exception as e:
            st.error(f"Failed to load .npy file: {e}")
            return

        error = validate_input(input_data, expected_shape)
        if error:
            st.error(error)
            return

        try:
            features = preprocess(input_data, expected_shape)
            prediction = model.predict(features)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            st.success(f"Predicted class: {predicted_class}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
