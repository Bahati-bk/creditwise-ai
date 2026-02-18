from fastapi import FastAPI
from flask import app
from app.schemas import LoanRequest
from app.model import predict_approval, predict_amount
from app.explain import explain_classification, explain_regression

app = FastAPI(title="CreditWise AI - Loan Approval Prediction API")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(loan_request: LoanRequest):
    features = [loan_request.age, loan_request.income, loan_request.requested_amount, loan_request.previous_defaults]
    approval, confidence = predict_approval(features)
    suggested_amount = predict_amount([loan_request.age, loan_request.income, loan_request.previous_defaults])
    return {
        "approval": approval,
        "confidence": confidence,
        "suggested_amount": suggested_amount
    }
    
@app.post("/explain")
def explain(loan_request: LoanRequest):
    features = [loan_request.age, loan_request.income, loan_request.requested_amount, loan_request.previous_defaults]
    classification_explanation = explain_classification(features)
    regression_explanation = explain_regression([loan_request.age, loan_request.income, loan_request.previous_defaults])
    return {
        "classification_explanation": classification_explanation,
        "regression_explanation": regression_explanation
    }