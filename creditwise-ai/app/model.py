import joblib

model = joblib.load("models/classifier_v1.pkl")
reg = joblib.load("models/regressor_v1.pkl")

def predict_approval(features):
    return model.predict([features])[0], model

def predict_amount(features):
    return reg.predict([features])[0]