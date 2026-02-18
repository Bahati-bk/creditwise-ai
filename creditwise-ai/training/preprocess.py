import pandas as pd
import numpy as np

def generate_data(n_samples=1000):
    # Generate synthetic data
    np.random.seed(42)
    data = pd.DataFrame({
        "age": np.random.randint(18, 60, size=n_samples),
        "income": np.random.randint(50000, 500000, size=n_samples),
        "loan_amount": np.random.randint(1000, 50000, size=n_samples),
        "previous_defaults": np.random.choice([0, 5], size=n_samples),
        "loan_approved": np.random.choice([0, 1], size=n_samples)
    })
    return data

def save_data(data, path="data/processed/microfinance_data.csv"):
    data.to_csv(path, index=False)
    
if __name__ == "__main__":
    data = generate_data()
    save_data(data)
    print("Data generated and saved to data/processed/microfinance_data.csv")