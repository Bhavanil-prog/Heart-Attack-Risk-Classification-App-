# Feature: ml-project-deployment, Property 7: CNN preprocessing produces array matching expected model input shape
# Feature: ml-project-deployment, Property 8: Invalid CNN input format is rejected

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from hypothesis import given, settings, assume
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

from pages.cnn import preprocess, validate_input


@st.composite
def shape_and_array(draw):
    ndims = draw(st.integers(min_value=1, max_value=3))
    dims = draw(st.lists(st.integers(min_value=1, max_value=32), min_size=ndims, max_size=ndims))
    shape = tuple(dims)
    arr = np.zeros(shape, dtype=np.float32)
    return arr, (None,) + shape


@given(shape_and_array())
@settings(max_examples=100)
def test_cnn_preprocess_output_shape(args):
    """Property 7: preprocess returns array with shape (1,) + spatial_shape for any valid input."""
    # Validates: Requirements 3.3
    arr, expected_shape = args
    result = preprocess(arr, expected_shape)
    assert result.shape == (1,) + arr.shape, (
        f"Expected shape {(1,) + arr.shape}, got {result.shape}"
    )


# Property 8: Invalid CNN input format is rejected

@st.composite
def mismatched_shape_strategy(draw):
    ndims = draw(st.integers(min_value=1, max_value=3))
    shape_a = tuple(draw(st.lists(st.integers(min_value=1, max_value=16), min_size=ndims, max_size=ndims)))
    # Make shape_b different from shape_a
    shape_b = tuple(draw(st.lists(st.integers(min_value=1, max_value=16), min_size=ndims, max_size=ndims)))
    assume(shape_a != shape_b)
    arr = np.zeros(shape_a, dtype=np.float32)
    expected_shape = (None,) + shape_b
    return arr, expected_shape


@given(mismatched_shape_strategy())
@settings(max_examples=100)
def test_cnn_validate_input_wrong_shape(args):
    """Property 8 (wrong shape): validate_input returns a descriptive error for mismatched shapes.
    # Validates: Requirements 3.5
    """
    arr, expected_shape = args
    result = validate_input(arr, expected_shape)
    assert result is not None, (
        f"Expected an error string for shape mismatch, got None. "
        f"arr.shape={arr.shape}, expected_shape={expected_shape}"
    )
    assert isinstance(result, str) and len(result) > 0, (
        f"Expected a non-empty error string, got: {result!r}"
    )


@given(
    st.one_of(
        np_arrays(dtype=np.dtype("object"), shape=st.integers(min_value=1, max_value=16)),
        np_arrays(dtype=np.dtype("U10"), shape=st.integers(min_value=1, max_value=16)),
    )
)
@settings(max_examples=100)
def test_cnn_validate_input_non_numeric_dtype(arr):
    """Property 8 (non-numeric dtype): validate_input returns a descriptive error for object/str dtypes.
    # Validates: Requirements 3.5
    """
    # Use any expected_shape; dtype check happens before shape check
    expected_shape = (None,) + arr.shape
    result = validate_input(arr, expected_shape)
    assert result is not None, (
        f"Expected an error string for non-numeric dtype '{arr.dtype}', got None."
    )
    assert isinstance(result, str) and len(result) > 0, (
        f"Expected a non-empty error string, got: {result!r}"
    )
