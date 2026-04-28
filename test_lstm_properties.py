# Feature: ml-project-deployment, Property 5: LSTM preprocessing produces padded array of correct shape

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_TF = True
except ImportError:
    HAS_TF = False

from pages.lstm import preprocess


class MockTokenizer:
    def texts_to_sequences(self, texts):
        # Return a list of token ids (just use ord values of chars, capped at 1000)
        return [[ord(c) % 1000 for c in texts[0]]]


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not available")
@given(
    text=st.text(min_size=1),
    max_len=st.integers(min_value=50, max_value=500),
)
@settings(max_examples=100)
def test_lstm_preprocess_output_shape(text, max_len):
    """Property 5: LSTM preprocessing produces padded array of correct shape (1, max_len)."""
    # Validates: Requirements 2.4
    tokenizer = MockTokenizer()
    result = preprocess(text, tokenizer, max_len)

    assert result.shape == (1, max_len), (
        f"Expected shape (1, {max_len}), got {result.shape}"
    )


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not available")
@given(
    text=st.text(min_size=1),
    max_len=st.integers(min_value=50, max_value=500),
)
@settings(max_examples=100)
def test_lstm_preprocess_values_nonnegative_integers(text, max_len):
    """Property 5: All values in the padded array are non-negative integers."""
    # Validates: Requirements 2.4
    tokenizer = MockTokenizer()
    result = preprocess(text, tokenizer, max_len)

    assert (result >= 0).all(), (
        f"Expected all non-negative values, found negatives in {result}"
    )
    assert result.dtype.kind in ('i', 'u'), (
        f"Expected integer dtype, got {result.dtype}"
    )


# Feature: ml-project-deployment, Property 6: Empty or whitespace-only text is rejected by LSTM validator

whitespace_strategy = st.one_of(
    st.just(""),
    st.text(alphabet=st.characters(whitelist_categories=("Zs", "Cc")), min_size=1),
)


@given(text=whitespace_strategy)
@settings(max_examples=100)
def test_lstm_empty_or_whitespace_rejected(text):
    """Property 6: Empty or whitespace-only text is rejected by LSTM validator.

    Validates: Requirements 2.6
    """
    from unittest.mock import MagicMock
    from pages.lstm import validate_text

    mock_model = MagicMock()

    result = validate_text(text)

    assert result is not None, (
        f"Expected an error string for whitespace/empty input {repr(text)}, got None"
    )
    assert isinstance(result, str), (
        f"Expected error to be a string, got {type(result)}"
    )
    mock_model.predict.assert_not_called()
