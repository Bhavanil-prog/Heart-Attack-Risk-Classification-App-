# ML Project Deployment

A multi-model Streamlit application serving three ML prediction services:
- Heart Attack Risk Classifier (Random Forest)
- Text Classification (LSTM)
- Image/Sequence Classification (CNN)

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Model Artifacts

Place all model artifacts in the `models/` directory before running the app.

#### Required files

| File | Description |
|------|-------------|
| `models/rf_model.pkl` | Heart Attack Classifier (Random Forest) |
| `models/lstm_model.h5` | LSTM text model |
| `models/tokenizer.pkl` | LSTM tokenizer |
| `models/cnn_model.h5` | CNN model |

#### Directory structure

```
models/
├── rf_model.pkl       # Heart Attack Classifier (Random Forest)
├── lstm_model.h5      # LSTM text model
├── tokenizer.pkl      # LSTM tokenizer
└── cnn_model.h5       # CNN model
```

If any artifact is missing, the app will display a descriptive error and halt inference for that service.

## Running the app

```bash
streamlit run main.py
```

## Running tests

```bash
pytest tests/
```
