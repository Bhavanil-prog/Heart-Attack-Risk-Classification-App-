import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))

RF_MODEL_PATH = os.environ.get("RF_MODEL_PATH", os.path.join(MODELS_DIR, "rf_model.pkl"))
LSTM_MODEL_PATH = os.environ.get("LSTM_MODEL_PATH", os.path.join(MODELS_DIR, "lstm_model.h5"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", os.path.join(MODELS_DIR, "tokenizer.pkl"))
CNN_MODEL_PATH = os.environ.get("CNN_MODEL_PATH", os.path.join(MODELS_DIR, "cnn_model.h5"))
