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
        f_i = gam.partial_dependence(term=i, X=X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
    # shape = (n_samples, n_features)
    contrib_matrix = np.column_stack(contrib_matrix)

    # dataframe lisible
    df_contrib = pd.DataFrame(
        contrib_matrix,
        columns=X.columns
    )

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

    # dataframe lisible
    df_contrib = pd.DataFrame(
        contrib_matrix,
        columns=X.columns
    )
    return df_contrib

