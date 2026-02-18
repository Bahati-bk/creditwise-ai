import shap
import pandas as pd
from app.model import model, reg

explainer_model = shap.TreeExplainer(model)
explainer_reg = shap.TreeExplainer(reg)

def explain_classification(features):
    shap_values = explainer_model.shap_values(pd.DataFrame([features]))
    return shap_values.tolist()

def explain_regression(features):
    shap_values = explainer_reg.shap_values(pd.DataFrame([features]))
    return shap_values.tolist()