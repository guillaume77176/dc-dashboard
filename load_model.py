import joblib
import pandas as pd
from load_data import get_X_cop1, get_X_cop2


def pred_cop1(end_date, steps):
    X_cop1 = get_X_cop1(end_date=end_date, steps=steps)
    cop1 = joblib.load("models_cop/xgboost_cop_model1.pkl")
    pred_cop1 = cop1.predict(X_cop1)

    return pred_cop1   


def pred_cop2(end_date, steps):
    X_cop2 = get_X_cop2(end_date=end_date, steps=steps)
    cop2 = joblib.load("models_cop/xgboost_cop_model2.pkl")
    pred_cop2 = cop2.predict(X_cop2)

    return pred_cop2

def interpret_cop1(end_date, steps = 1):
    input = get_X_cop1(end_date=end_date, steps=steps)
    gam = joblib.load("models_cop/gam_cop_model1.pkl")
    contributions = []
    for i in range(input.shape[1]):
        c = gam.partial_dependence(term=i, X=input)[0]
        contributions.append(c)
    df_contrib = pd.DataFrame({
    "variable": input.columns,
    "contribution": contributions
    })

    df_contrib["abs"] = df_contrib["contribution"].abs()
    df_contrib = df_contrib.sort_values("abs", ascending=False)

    return df_contrib

def interpret_cop2(end_date, steps = 1):
    input = get_X_cop2(end_date=end_date, steps=steps)
    gam = joblib.load("models_cop/gam_cop_model2.pkl")
    contributions = []
    for i in range(input.shape[1]):
        c = gam.partial_dependence(term=i, X=input)[0]
        contributions.append(c)
    df_contrib = pd.DataFrame({
    "variable": input.columns,
    "contribution": contributions
    })

    df_contrib["abs"] = df_contrib["contribution"].abs()
    df_contrib = df_contrib.sort_values("abs", ascending=False)

    return df_contrib

