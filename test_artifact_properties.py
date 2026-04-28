# Feature: ml-project-deployment, Property 1: Missing artifact raises descriptive error

import sys
import os
import contextlib
import importlib
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings
import hypothesis.strategies as st

from errors import ModelLoadError

# ---------------------------------------------------------------------------
# Import the page modules with st.cache_resource patched out so the loaders
# are plain functions (no caching side-effects between test runs).
# ---------------------------------------------------------------------------
_noop_decorator = lambda f: f

with patch("streamlit.cache_resource", _noop_decorator), \
     patch("streamlit.spinner", return_value=contextlib.nullcontext()):
    import pages.heart as heart_module
    import pages.cnn as cnn_module
    import pages.lstm as lstm_module
    importlib.reload(heart_module)
    importlib.reload(cnn_module)
    importlib.reload(lstm_module)

# ---------------------------------------------------------------------------
# Strategy: alphanumeric text prefixed with a guaranteed-nonexistent directory
# ---------------------------------------------------------------------------
path_text_strategy = st.text(
    min_size=1,
    alphabet=st.characters(whitelist_categories=("L", "N")),
)


def make_nonexistent_path(s: str) -> str:
    return f"/nonexistent_test_path_{s}"


# ---------------------------------------------------------------------------
# Property 1 tests
# ---------------------------------------------------------------------------

@given(path_text_strategy)
@settings(max_examples=50, deadline=None)
def test_heart_load_model_missing_path_raises_model_load_error(path_suffix):
    """Property 1: heart.load_model raises ModelLoadError for any missing path.
    Validates: Requirements 1.2
    """
    path = make_nonexistent_path(path_suffix)
    with patch("streamlit.spinner", return_value=contextlib.nullcontext()):
        try:
            heart_module.load_model(path)
            assert False, f"Expected ModelLoadError for path {path!r}"
        except ModelLoadError as e:
            assert path in str(e), (
                f"Error message should contain the path {path!r}, got: {str(e)!r}"
            )


@given(path_text_strategy)
@settings(max_examples=50, deadline=None)
def test_cnn_load_model_missing_path_raises_model_load_error(path_suffix):
    """Property 1: cnn.load_model raises ModelLoadError for any missing path.
    Validates: Requirements 3.2
    """
    path = make_nonexistent_path(path_suffix)
    try:
        cnn_module.load_model(path)
        assert False, f"Expected ModelLoadError for path {path!r}"
    except ModelLoadError as e:
        assert path in str(e), (
            f"Error message should contain the path {path!r}, got: {str(e)!r}"
        )


@given(path_text_strategy)
@settings(max_examples=50, deadline=None)
def test_lstm_load_artifacts_missing_paths_raises_model_load_error(path_suffix):
    """Property 1: lstm.load_artifacts raises ModelLoadError for any missing paths.
    Validates: Requirements 2.3
    """
    model_path = make_nonexistent_path(f"model_{path_suffix}")
    tokenizer_path = make_nonexistent_path(f"tokenizer_{path_suffix}")
    try:
        lstm_module.load_artifacts(model_path, tokenizer_path)
        assert False, f"Expected ModelLoadError for paths {model_path!r}, {tokenizer_path!r}"
    except ModelLoadError as e:
        assert model_path in str(e) or tokenizer_path in str(e), (
            f"Error message should contain a missing path, got: {str(e)!r}"
        )
