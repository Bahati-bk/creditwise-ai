import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import joblib

MODEL_PATH = "models/classifier_v1.pkl"
DATA_PATH = "data/processed/microfinance_data.csv"

def train_model():
    """Train a new model, evaluate it, and save it."""
    print("Training a new model...")

    # Load preprocessed data
    data = pd.read_csv(DATA_PATH)
    X = data[["age", "income", "loan_amount", "previous_defaults"]]
    y = data["loan_approved"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost classifier
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")
    return model

# Load model if exists, else train new one
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded existing model from {MODEL_PATH}")
except (EOFError, FileNotFoundError):
    model = train_model()

# Example functions for your FastAPI app
def predict_approval(X):
    return model.predict(X)

def predict_amount(X):
    return model.predict_proba(X)[:, 1]

# Optional: if you run this script directly, retrain model
if __name__ == "__main__":
    print("Running training script...")
    train_model()
