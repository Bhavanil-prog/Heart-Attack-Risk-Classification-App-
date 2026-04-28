# Feature: ml-project-deployment, Property 2: Heart input preprocessing produces correct feature vector

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

from pages.heart import preprocess

EXPECTED_COLUMNS = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "Exercise_Angina", "Sex_F", "Sex_M", "Chest_PainType", "Resting_ECG", "st_Slope",
]

valid_input_strategy = st.fixed_dictionaries({
    "age": st.integers(min_value=20, max_value=100),
    "resting_bp": st.integers(min_value=0, max_value=300),
    "cholesterol": st.integers(min_value=0, max_value=700),
    "fasting_bs": st.sampled_from([0, 1]),
    "max_hr": st.integers(min_value=60, max_value=250),
    "oldpeak": st.floats(min_value=-3.0, max_value=6.6, allow_nan=False, allow_infinity=False),
    "gender": st.sampled_from(["M", "F"]),
    "chest_pain_type": st.sampled_from(["ATA", "NAP", "ASY", "TA"]),
    "resting_ecg": st.sampled_from(["Normal", "ST", "LVH"]),
    "exercise_angina": st.sampled_from(["N", "Y"]),
    "st_slope": st.sampled_from(["Up", "Flat", "Down"]),
})


@given(valid_input_strategy)
@settings(max_examples=100)
def test_preprocess_output_columns(inputs):
    """Property 2: preprocess returns a DataFrame with exactly the 12 expected columns."""
    df = preprocess(inputs)
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Expected columns {EXPECTED_COLUMNS}, got {list(df.columns)}"
    )


@given(valid_input_strategy)
@settings(max_examples=100)
def test_preprocess_encoded_categorical_ranges(inputs):
    """Property 2: encoded categorical values are within expected integer ranges."""
    df = preprocess(inputs)

    row = df.iloc[0]

    assert row["Exercise_Angina"] in (0, 1), f"Exercise_Angina out of range: {row['Exercise_Angina']}"
    assert row["Sex_F"] in (0, 1), f"Sex_F out of range: {row['Sex_F']}"
    assert row["Sex_M"] in (0, 1), f"Sex_M out of range: {row['Sex_M']}"
    assert 0 <= row["Chest_PainType"] <= 3, f"Chest_PainType out of range: {row['Chest_PainType']}"
    assert 0 <= row["Resting_ECG"] <= 2, f"Resting_ECG out of range: {row['Resting_ECG']}"
    assert 0 <= row["st_Slope"] <= 2, f"st_Slope out of range: {row['st_Slope']}"


# Feature: ml-project-deployment, Property 3: Out-of-range heart inputs are rejected

from pages.heart import validate_inputs

# Numeric fields and their valid ranges
NUMERIC_FIELDS = [
    ("age", 20, 100),
    ("resting_bp", 0, 300),
    ("cholesterol", 0, 700),
    ("fasting_bs", 0, 1),
    ("max_hr", 60, 250),
    ("oldpeak", -3.0, 6.6),
]

valid_base_strategy = st.fixed_dictionaries({
    "age": st.integers(min_value=20, max_value=100),
    "resting_bp": st.integers(min_value=0, max_value=300),
    "cholesterol": st.integers(min_value=0, max_value=700),
    "fasting_bs": st.sampled_from([0, 1]),
    "max_hr": st.integers(min_value=60, max_value=250),
    "oldpeak": st.floats(min_value=-3.0, max_value=6.6, allow_nan=False, allow_infinity=False),
    "gender": st.sampled_from(["M", "F"]),
    "chest_pain_type": st.sampled_from(["ATA", "NAP", "ASY", "TA"]),
    "resting_ecg": st.sampled_from(["Normal", "ST", "LVH"]),
    "exercise_angina": st.sampled_from(["N", "Y"]),
    "st_slope": st.sampled_from(["Up", "Flat", "Down"]),
})


def out_of_range_value(field, lo, hi):
    """Return a strategy that generates a value strictly outside [lo, hi]."""
    if isinstance(lo, float) or isinstance(hi, float):
        return st.one_of(
            st.floats(max_value=lo - 0.01, allow_nan=False, allow_infinity=False),
            st.floats(min_value=hi + 0.01, allow_nan=False, allow_infinity=False),
        )
    else:
        return st.one_of(
            st.integers(max_value=lo - 1),
            st.integers(min_value=hi + 1),
        )


@st.composite
def out_of_range_input_strategy(draw):
    base = draw(valid_base_strategy)
    field, lo, hi = draw(st.sampled_from(NUMERIC_FIELDS))
    bad_value = draw(out_of_range_value(field, lo, hi))
    base[field] = bad_value
    return base


@given(out_of_range_input_strategy())
@settings(max_examples=100)
def test_out_of_range_inputs_are_rejected(inputs):
    """Property 3: validate_inputs returns an error string for out-of-range numeric fields."""
    # Validates: Requirements 1.7
    result = validate_inputs(inputs)
    assert isinstance(result, str), (
        f"Expected an error string, got {type(result).__name__!r} ({result!r}) for inputs {inputs}"
    )
    assert result is not None, f"Expected error string, got None for inputs {inputs}"


# Feature: ml-project-deployment, Property 4: Prediction result display maps all classes

from pages.heart import display_prediction


@given(st.integers())
@settings(max_examples=100)
def test_display_prediction_returns_nonempty_string(prediction_class):
    """Property 4: display_prediction returns a non-empty string for any integer class."""
    # Validates: Requirements 1.5, 1.6, 2.5, 3.4
    result = display_prediction(prediction_class)
    assert isinstance(result, str), f"Expected str, got {type(result).__name__!r}"
    assert len(result) > 0, f"Expected non-empty string for class {prediction_class}"


@given(st.just(1))
@settings(max_examples=100)
def test_display_prediction_class1_is_high_risk(prediction_class):
    """Property 4: class 1 maps to a high-risk message."""
    # Validates: Requirements 1.5
    result = display_prediction(prediction_class)
    assert any(kw in result for kw in ("High Risk", "high risk", "⚠️")), (
        f"Expected high-risk message for class 1, got: {result!r}"
    )


@given(st.just(0))
@settings(max_examples=100)
def test_display_prediction_class0_is_low_risk(prediction_class):
    """Property 4: class 0 maps to a low-risk message."""
    # Validates: Requirements 1.6
    result = display_prediction(prediction_class)
    assert any(kw in result for kw in ("Low risk", "low risk", "😎")), (
        f"Expected low-risk message for class 0, got: {result!r}"
    )
