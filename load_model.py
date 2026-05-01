import joblib
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from load_data import get_X_cop1, get_X_cop2


def sample_action_det(
        actor : tf.keras.Model,
        state : np.array
)-> tf.Tensor:

        """
        Fonction qui tire un action d'une politique pi(.|s) avec paramétrisation pour gaussienne 
        provenant du réseau NN actor. Conçu pour renvoyer une action dans [0,1].
        Version déterministe où l'action tiré correspond à la moyenne des actions sous pi(a|s).

        actor : réseau NN actor
        state : vecteur d'état
        """
        state = tf.cast(state, tf.float32)

        mu, _ = actor(state)

        # Approximation avec sigmoid de l'action tirée
        a = tf.sigmoid(mu) 

        return a

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

def actor(end_date, steps):
    actor = keras.models.load_model("actor.keras")
    state = get_control(end_date, steps)

    good_traj = np.ones((state.shape[0], 4))

    for i in tqdm(range(0,state.shape[0])):
        S_it = state[i,:]
        actions = sample_action_det(actor, S_it.reshape(1,-1)).numpy()
        good_traj[i,0] = actions[0][0]
        good_traj[i,1] = actions[0][1]
        good_traj[i,2] = actions[0][2]
        good_traj[i,3] = actions[0][3]
    
    return good_traj