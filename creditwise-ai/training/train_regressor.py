import os
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

MODEL_PATH = "models/regressor_v1.pkl"
DATA_PATH = "data/processed/microfinance_data.csv"

def train_regressor():
    print("Training new regressor model...")

    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df[["age", "income", "previous_defaults"]]
    y = df["loan_amount"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    reg = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    reg.fit(X_train, y_train)

    # Evaluate
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Root Mean Squared Error: {mse:.4f}")

    # Ensure models folder exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save the trained model
    joblib.dump(reg, MODEL_PATH)
    print(f"Regressor saved to {MODEL_PATH}")

    return reg

if __name__ == "__main__":
    train_regressor()
