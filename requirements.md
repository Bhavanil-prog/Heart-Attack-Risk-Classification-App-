# Requirements Document

## Introduction

This feature covers the packaging and deployment of multiple ML projects as interactive web applications and/or API services. The projects include:

1. A Heart Attack Risk Classifier using a pre-trained Random Forest model, already implemented as a Streamlit app (`app.py` + `rf_model.pkl`).
2. A text-based model using a tokenizer (pickle) and an LSTM model (`.h5`, trained for 20 epochs).
3. An image or sequence-based model using a CNN (`.h5`, trained for 30 epochs).

The goal is to make all three models accessible to end users through deployed, production-ready interfaces.

## Glossary

- **Deployment_System**: The overall system responsible for packaging and serving ML models as web apps or APIs.
- **Streamlit_App**: A Streamlit-based web application that provides a UI for interacting with a model.
- **RF_Model**: The pre-trained Random Forest model stored as `rf_model.pkl`, used for heart attack risk classification.
- **LSTM_Model**: The LSTM neural network model saved as an `.h5` file after 20 training epochs.
- **CNN_Model**: The CNN neural network model saved as an `.h5` file after 30 training epochs.
- **Tokenizer**: The text tokenizer saved as a pickle file, used for preprocessing input to the LSTM_Model.
- **Prediction_Service**: A deployed endpoint (Streamlit page or API route) that accepts user input and returns model predictions.
- **Artifact**: A serialized model or preprocessing object (`.pkl`, `.h5`) required at inference time.

---

## Requirements

### Requirement 1: Heart Attack Risk Classification Deployment

**User Story:** As a user, I want to interact with the Heart Attack Risk Classifier through a web interface, so that I can get a risk prediction based on my health data.

#### Acceptance Criteria

1. THE Streamlit_App SHALL load the RF_Model from `rf_model.pkl` at startup.
2. WHEN the RF_Model file is missing or corrupt, THE Streamlit_App SHALL display a descriptive error message and halt prediction.
3. THE Streamlit_App SHALL accept the following inputs: Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, Gender, ChestPainType, RestingECG, ExerciseAngina, and ST_Slope.
4. WHEN the user submits the prediction form, THE Streamlit_App SHALL preprocess inputs using the same encoding and scaling logic used during training.
5. WHEN the RF_Model predicts class 1, THE Streamlit_App SHALL display a high-risk warning message.
6. WHEN the RF_Model predicts class 0, THE Streamlit_App SHALL display a low-risk success message.
7. IF any required input field contains an out-of-range value, THEN THE Streamlit_App SHALL prevent prediction and display a validation error.

---

### Requirement 2: LSTM Model Deployment

**User Story:** As a user, I want to submit text input and receive a prediction from the LSTM model, so that I can use the trained model for inference.

#### Acceptance Criteria

1. THE Deployment_System SHALL load the LSTM_Model from its `.h5` file at startup.
2. THE Deployment_System SHALL load the Tokenizer from its `.pkl` file at startup.
3. WHEN the LSTM_Model or Tokenizer artifact is missing, THE Deployment_System SHALL display a descriptive error and halt inference.
4. WHEN a user submits text input, THE Deployment_System SHALL tokenize and pad the input using the Tokenizer before passing it to the LSTM_Model.
5. WHEN the LSTM_Model produces a prediction, THE Deployment_System SHALL display the predicted class or value to the user.
6. IF the text input is empty, THEN THE Deployment_System SHALL display a validation error and not invoke the LSTM_Model.

---

### Requirement 3: CNN Model Deployment

**User Story:** As a user, I want to submit input data and receive a prediction from the CNN model, so that I can use the trained model for inference.

#### Acceptance Criteria

1. THE Deployment_System SHALL load the CNN_Model from its `.h5` file at startup.
2. WHEN the CNN_Model artifact is missing or corrupt, THE Deployment_System SHALL display a descriptive error and halt inference.
3. WHEN a user submits input, THE Deployment_System SHALL preprocess the input to match the shape and format expected by the CNN_Model.
4. WHEN the CNN_Model produces a prediction, THE Deployment_System SHALL display the predicted class or value to the user.
5. IF the submitted input does not match the expected format, THEN THE Deployment_System SHALL display a descriptive validation error.

---

### Requirement 4: Artifact Management

**User Story:** As a developer, I want all model artifacts to be versioned and consistently referenced, so that deployments are reproducible and maintainable.

#### Acceptance Criteria

1. THE Deployment_System SHALL store all Artifacts (`.pkl`, `.h5`) in a dedicated `models/` directory.
2. THE Deployment_System SHALL load each Artifact using a configurable file path, not a hardcoded absolute path.
3. IF an Artifact file path is not found at the configured location, THEN THE Deployment_System SHALL raise a descriptive error identifying which Artifact is missing.
4. THE Deployment_System SHALL document the required Artifact filenames and expected directory structure in a `README.md`.

---

### Requirement 5: Dependency and Environment Management

**User Story:** As a developer, I want a reproducible environment specification, so that the deployment can be set up consistently across machines.

#### Acceptance Criteria

1. THE Deployment_System SHALL provide a `requirements.txt` listing all Python dependencies with pinned versions.
2. WHEN a new environment is created using the `requirements.txt`, THE Deployment_System SHALL install all dependencies needed to run all three Prediction_Services.
3. THE Deployment_System SHALL specify the Python version required in the environment documentation.

---

### Requirement 6: Multi-App Navigation

**User Story:** As a user, I want a single entry point to navigate between the three ML apps, so that I don't need to run separate commands for each model.

#### Acceptance Criteria

1. THE Streamlit_App SHALL provide a navigation mechanism (sidebar or page selector) allowing the user to switch between the Heart Attack Classifier, LSTM Prediction_Service, and CNN Prediction_Service.
2. WHEN a user selects a Prediction_Service, THE Streamlit_App SHALL display only the inputs and outputs relevant to that service.
3. WHILE a Prediction_Service is loading its Artifact, THE Streamlit_App SHALL display a loading indicator to the user.
