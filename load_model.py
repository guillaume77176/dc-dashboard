import joblib
import pandas as pd
import numpy as np
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

def interpret_cop1(end_date, steps):
    X = get_X_cop1(end_date=end_date, steps=steps)

    gam = joblib.load("models_cop/gam_cop_model1.pkl")

    contrib_matrix = []

    for i in range(X.shape[1]):
        # contribution de la variable i POUR TOUTES les lignes
        f_i = gam.partial_dependence(term=i, X=X)
    # shape = (n_samples, n_features)
        contrib_matrix.append(f_i)
    contrib_matrix = np.column_stack(contrib_matrix)

    
    df_contrib = pd.DataFrame({ "variable": input.columns, "contribution": contributions })
    df_contrib["abs"] = df_contrib["contribution"].abs()
    df_contrib = df_contrib.sort_values("abs", ascending=False)
    df_contrib = df_contrib.drop(["abs"], axis =1)
    return df_contrib


def interpret_cop2(end_date, steps):
    X = get_X_cop2(end_date=end_date, steps=steps)

    gam = joblib.load("models_cop/gam_cop_model2.pkl")

    contrib_matrix = []

    for i in range(X.shape[1]):
        # contribution de la variable i POUR TOUTES les lignes
        f_i = gam.partial_dependence(term=i, X=X)
        contrib_matrix.append(f_i)

    # shape = (n_samples, n_features)
    contrib_matrix = np.column_stack(contrib_matrix)

    df_contrib = pd.DataFrame({ "variable": input.columns, "contribution": contributions })
    df_contrib["abs"] = df_contrib["contribution"].abs()
    df_contrib = df_contrib.sort_values("abs", ascending=False)
    df_contrib = df_contrib.drop(["abs"], axis =1)
    return df_contrib

