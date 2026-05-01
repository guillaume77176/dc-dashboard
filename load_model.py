import joblib
import pandas as pd
import numpy as np
from load_data import get_X_cop1, get_X_cop2


def pred_cop1(data, cop1, end_date, steps):
    X_cop1 = get_X_cop1(data, end_date=end_date, steps=steps)
    pred_cop1 = cop1.predict(X_cop1)

    return pred_cop1   


def pred_cop2(data, cop2, end_date, steps):
    X_cop2 = get_X_cop2(data, end_date=end_date, steps=steps)
    pred_cop2 = cop2.predict(X_cop2)

    return pred_cop2

def interpret_cop1(data, gam, end_date, steps):
    X = get_X_cop1(data, end_date=end_date, steps=steps)

    contrib_matrix = []
    for i in range(X.shape[1]):
        # contribution de la variable i POUR TOUTES les lignes
        f_i = gam.partial_dependence(term=i, X=X)
    # shape = (n_samples, n_features)
        contrib_matrix.append(f_i)
    contrib_matrix = np.column_stack(contrib_matrix)
    # dataframe lisible
    df_contrib = pd.DataFrame(
        contrib_matrix,
        columns=X.columns
    )

    return df_contrib

def interpret_cop2(data, gam, end_date, steps):
    X = get_X_cop2(data, end_date=end_date, steps=steps)


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

