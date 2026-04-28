# Implementation Plan: ML Project Deployment

## Overview

Unify three ML models (Heart Attack Classifier, LSTM, CNN) into a single navigable Streamlit app. Migrate the existing `app.py` into a pages-based structure, add LSTM and CNN pages, centralize artifact paths via `config.py`, and cover all correctness properties with Hypothesis property-based tests.

## Tasks

- [x] 1. Set up project structure and config
  - Create `pages/` directory (empty `__init__.py`)
  - Create `models/` directory with a `.gitkeep` placeholder
  - Create `config.py` with `BASE_DIR`, `MODELS_DIR`, and all four artifact path constants (`RF_MODEL_PATH`, `LSTM_MODEL_PATH`, `TOKENIZER_PATH`, `CNN_MODEL_PATH`); support env-var overrides
  - Create `ModelLoadError` exception class in a shared `errors.py` module
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 2. Create `pages/heart.py` (migrate from `app.py`)
  - [x] 2.1 Implement `load_model`, `preprocess`, and `render` in `pages/heart.py`
    - `load_model(path)` loads `rf_model.pkl` via pickle; raises `ModelLoadError` on any failure
    - `preprocess(inputs: dict) -> pd.DataFrame` applies the encoding maps and `StandardScaler` from `app.py`
    - `render()` shows a loading spinner while loading, renders the input form, validates ranges, calls `preprocess` + `model.predict`, and displays high/low risk messages
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 6.3_

  - [x] 2.2 Write property test for `preprocess` (Property 2)
    - **Property 2: Heart input preprocessing produces correct feature vector**
    - Use `@given(st.builds(...))` with valid in-range values; assert output DataFrame has exactly the 12 expected columns and encoded values are within expected integer ranges
    - **Validates: Requirements 1.4**

  - [x] 2.3 Write property test for heart input validation (Property 3)
    - **Property 3: Out-of-range heart inputs are rejected**
    - Generate `HeartInput` dicts with at least one numeric field outside its valid range; assert the validator returns an error string and does not return a DataFrame
    - **Validates: Requirements 1.7**

  - [x] 2.4 Write property test for prediction result display (Property 4)
    - **Property 4: Prediction result display maps all classes**
    - Generate random integer prediction classes; assert the display helper returns a non-empty string; assert class 1 → high-risk message, class 0 → low-risk message
    - **Validates: Requirements 1.5, 1.6**

- [x] 3. Create `pages/lstm.py`
  - [x] 3.1 Implement `load_artifacts`, `preprocess`, and `render` in `pages/lstm.py`
    - `load_artifacts(model_path, tokenizer_path)` loads the `.h5` model and pickle tokenizer; raises `ModelLoadError` on any failure
    - `preprocess(text, tokenizer, max_len) -> np.ndarray` tokenizes and pads to shape `(1, max_len)`
    - `render()` shows a loading spinner, renders a text area, validates non-empty input, calls `preprocess` + `model.predict`, and displays the predicted class
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 6.3_

  - [x] 3.2 Write property test for LSTM preprocessing (Property 5)
    - **Property 5: LSTM preprocessing produces padded array of correct shape**
    - Generate random non-empty strings and `max_len` values (50–500); assert output shape is `(1, max_len)` and all values are non-negative integers
    - **Validates: Requirements 2.4**

  - [x] 3.3 Write property test for LSTM empty-input validation (Property 6)
    - **Property 6: Empty or whitespace-only text is rejected by LSTM validator**
    - Generate strings composed entirely of whitespace/control characters plus the empty string; assert the validator returns an error and does not invoke the model
    - **Validates: Requirements 2.6**

- [x] 4. Create `pages/cnn.py`
  - [x] 4.1 Implement `load_model`, `preprocess`, and `render` in `pages/cnn.py`
    - `load_model(path)` loads the CNN `.h5` file; raises `ModelLoadError` on any failure
    - `preprocess(input_data, expected_shape) -> np.ndarray` reshapes input to match model expectations
    - `render()` shows a loading spinner, renders an appropriate input widget (file upload or numeric input), validates format, calls `preprocess` + `model.predict`, and displays the predicted class
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.3_

  - [x] 4.2 Write property test for CNN preprocessing (Property 7)
    - **Property 7: CNN preprocessing produces array matching expected model input shape**
    - Generate random numeric arrays of the correct shape; assert the preprocessed output shape matches the expected model input shape
    - **Validates: Requirements 3.3**

  - [x] 4.3 Write property test for invalid CNN input (Property 8)
    - **Property 8: Invalid CNN input format is rejected**
    - Generate arrays with wrong shapes or dtypes; assert the validator returns a descriptive error string and does not invoke the model
    - **Validates: Requirements 3.5**

- [x] 5. Create `main.py` entry point and wire navigation
  - Implement sidebar `st.selectbox` with the three app names
  - Import and call `heart.render()`, `lstm.render()`, `cnn.render()` based on selection
  - _Requirements: 6.1, 6.2_

- [x] 6. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Write property test for missing artifact loading (Property 1)
  - [x] 7.1 Write property test for `ModelLoadError` on missing paths (Property 1)
    - **Property 1: Missing artifact raises descriptive error**
    - Generate random strings as artifact paths (guaranteed not to exist on disk); assert each loader (`load_model` in heart/lstm/cnn, `load_artifacts` in lstm) raises `ModelLoadError` whose message contains the artifact name and the missing path
    - **Validates: Requirements 1.2, 2.3, 3.2, 4.3**

- [x] 8. Add `requirements.txt` and `README.md`
  - [x] 8.1 Create `requirements.txt` with all dependencies pinned using `==`
    - Include: `streamlit`, `scikit-learn`, `pandas`, `numpy`, `tensorflow`/`keras`, `hypothesis`, `pytest`
    - _Requirements: 5.1, 5.2_

  - [x] 8.2 Write unit test asserting `requirements.txt` uses only pinned versions
    - Parse `requirements.txt` and assert every non-comment line contains `==`
    - _Requirements: 5.1_

  - [x] 8.3 Create `README.md` documenting required artifact filenames, `models/` directory structure, Python version, and how to run the app
    - _Requirements: 4.4, 5.3_

- [x] 9. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with a minimum of 100 examples (`settings` profile `"ci"`)
- Place a `conftest.py` at the project root to register the Hypothesis `"ci"` profile
- The existing `app.py` and `rf_model.pkl` remain untouched until `pages/heart.py` is verified working
