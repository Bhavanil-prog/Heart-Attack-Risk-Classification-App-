import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from config import RF_MODEL_PATH
from errors import ModelLoadError


@st.cache_resource
def load_model(path: str) -> RandomForestClassifier:
    """Load the RandomForest model from disk. Raises ModelLoadError on failure."""
    try:
        with st.spinner("Loading model..."):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        raise ModelLoadError("rf_model", path, e)


def preprocess(inputs: dict) -> pd.DataFrame:
    """Encode categorical inputs and scale numeric features."""
    Exercise_Angina = 1 if inputs["exercise_angina"] == "Y" else 0
    Sex_F = 1 if inputs["gender"] == "F" else 0
    Sex_M = 1 if inputs["gender"] == "M" else 0

    chest_pain_map = {"ASY": 3, "NAP": 2, "ATA": 1, "TA": 0}
    resting_ecg_map = {"Normal": 0, "LVH": 1, "ST": 2}
    st_slope_map = {"Down": 0, "Up": 1, "Flat": 2}

    df = pd.DataFrame({
        "Age": [inputs["age"]],
        "RestingBP": [inputs["resting_bp"]],
        "Cholesterol": [inputs["cholesterol"]],
        "FastingBS": [inputs["fasting_bs"]],
        "MaxHR": [inputs["max_hr"]],
        "Oldpeak": [inputs["oldpeak"]],
        "Exercise_Angina": [Exercise_Angina],
        "Sex_F": [Sex_F],
        "Sex_M": [Sex_M],
        "Chest_PainType": [chest_pain_map[inputs["chest_pain_type"]]],
        "Resting_ECG": [resting_ecg_map[inputs["resting_ecg"]]],
        "st_Slope": [st_slope_map[inputs["st_slope"]]],
    })

    scaler = StandardScaler()
    df[["Age", "RestingBP", "Cholesterol", "MaxHR"]] = scaler.fit_transform(
        df[["Age", "RestingBP", "Cholesterol", "MaxHR"]]
    )

    return df


def validate_inputs(inputs: dict) -> str | None:
    """Return an error string if any field is out of range, None if valid."""
    checks = [
        ("age", 20, 100, "Age must be between 20 and 100"),
        ("resting_bp", 0, 300, "RestingBP must be between 0 and 300"),
        ("cholesterol", 0, 700, "Cholesterol must be between 0 and 700"),
        ("fasting_bs", 0, 1, "FastingBS must be 0 or 1"),
        ("max_hr", 60, 250, "MaxHR must be between 60 and 250"),
        ("oldpeak", -3.0, 6.6, "Oldpeak must be between -3.0 and 6.6"),
    ]
    for field, lo, hi, msg in checks:
        val = inputs[field]
        if val < lo or val > hi:
            return msg
    return None


def display_prediction(prediction: int) -> str:
    """Return a risk message based on the predicted class."""
    if prediction == 1:
        return "⚠️ High Risk of Heart Attack❗"
    return "Low risk of Heart Attack 😎😊"


def render():
    st.title("Heart Attack Risk Classification App ❤️")

    try:
        model = load_model(RF_MODEL_PATH)
    except ModelLoadError as e:
        st.error(str(e))
        return

    age = st.number_input("Age", min_value=20, max_value=100, value=25)
    resting_bp = st.number_input("RestingBP", min_value=0, max_value=300, value=100)
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=140)
    fasting_bs = st.selectbox("FastingBS", (0, 1))
    max_hr = st.number_input("MaxHR", min_value=60, max_value=250, value=140)
    oldpeak = st.number_input("Oldpeak", min_value=-3.0, max_value=6.6, value=1.0)
    gender = st.selectbox("Gender (Male or Female)", ("M", "F"))
    chest_pain_type = st.selectbox("ChestPainType", ("ATA", "NAP", "ASY", "TA"))
    resting_ecg = st.selectbox("RestingECG", ("Normal", "ST", "LVH"))
    exercise_angina = st.selectbox("ExerciseAngina", ("N", "Y"))
    st_slope = st.selectbox("ST_Slope", ("Up", "Flat", "Down"))

    if st.button("Predict"):
        inputs = {
            "age": age,
            "resting_bp": resting_bp,
            "cholesterol": cholesterol,
            "fasting_bs": fasting_bs,
            "max_hr": max_hr,
            "oldpeak": oldpeak,
            "gender": gender,
            "chest_pain_type": chest_pain_type,
            "resting_ecg": resting_ecg,
            "exercise_angina": exercise_angina,
            "st_slope": st_slope,
        }

        error = validate_inputs(inputs)
        if error:
            st.error(error)
            return

        try:
            features = preprocess(inputs)
            prediction = model.predict(features)[0]
            message = display_prediction(prediction)
            if prediction == 1:
                st.error(message)
            else:
                st.success(message)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
