# -*- coding: utf-8 -*-
# Copyright 2024-2025 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas as pd
from load_data import get_data_viz, get_control
from load_model import pred_cop1, pred_cop2, pred_ppue, interpret_cop1, interpret_cop2, interpret_ppue
import plotly.graph_objects as go
import datetime as dt
import joblib


st.set_page_config(
    page_title="Dashboard Data center id X",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

"""
# :material/query_stats: Dashboard Simulation - Data center X (first containment) | 
"""

""  
st.text("AUTHORS : Guillaume Roustan, Guy Angoula, Emile Guillaume | Statapp ENSAE/KAPSDATA, Mai 2026.")
st.text("Predict metrics of interest, retrieve suspicious variables by an interpretation of contributions, and get the best decisions to make for modifying free-cooling and chiller valves")
cols = st.columns([1, 3])


#####################################################################

st.divider()

@st.cache_data
def get_data_viz_ch():
    data = pd.read_csv("https://minio.lab.sspcloud.fr/guillaume176/diffusion/data.csv")
    return data

@st.cache_data
def get_data_cop1():
    data = pd.read_csv("https://minio.lab.sspcloud.fr/guillaume176/diffusion/dataset_xgboost_ready_COP_v1_clim_1.csv")
    return data

@st.cache_data
def get_data_cop2():
    data = pd.read_csv("https://minio.lab.sspcloud.fr/guillaume176/diffusion/dataset_xgboost_ready_COP_v1_clim_2.csv")
    return data

@st.cache_data
def get_data_control():
    data = pd.read_csv("https://minio.lab.sspcloud.fr/guillaume176/diffusion/opt_actions.csv")
    return data


@st.cache_data
def get_data_ppue():
    data = pd.read_csv("https://minio.lab.sspcloud.fr/guillaume176/diffusion/dataset_xgboost_ready_ppue_clim1.csv")
    return data



@st.cache_resource
def model_cop1():
    cop1 = joblib.load("models_cop/xgboost_cop_model1.pkl")
    return cop1

@st.cache_resource
def model_cop2():
    cop2 = joblib.load("models_cop/xgboost_cop_model2.pkl")
    return cop2

@st.cache_resource
def model_ppue():
    ppue = joblib.load("models_ppue/xgboost_ppue_model.pkl")
    return ppue

@st.cache_resource
def model_interpretppue():
    gam = joblib.load("models_ppue/gam_ppue_model.pkl")
    return gam

@st.cache_resource
def model_interpretc1():
    gam = joblib.load("models_cop/gam_cop_model1.pkl")
    return gam

@st.cache_resource
def model_interpretc2():
    gam = joblib.load("models_cop/gam_cop_model2.pkl")
    return gam

data_global = get_data_viz_ch()
data_cop1 = get_data_cop1()
data_cop2 = get_data_cop2()
data_ppue = get_data_ppue()
data_control = get_data_control()

cop1 = model_cop1()
cop2 = model_cop2()
gam1 = model_interpretc1()
gam2 = model_interpretc2()
gam3 = model_interpretppue()
ppue = model_ppue()


st.markdown("### 🔮 Vizualise the predictions")

# Time horizon selector
metrics = {
    "Cop clim 1" : ["COP_v1_clim_1"],
    "Cop clim 2" : ["COP_v1_clim_2"],
    "Ppue" : ["ConfinementPPUE_1","OnduleurPuissance_1"]
    
}

top_left_cell1 = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

top_left_cell2 = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

top_left_cell3 = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)


with top_left_cell1:
    # Buttons for picking time horizon
    met = st.pills(
        "Select your metric of interest",
        options=list(metrics.keys()),
        default="Cop clim 1",
    )
    if met == "Cop clim 1":
        model = cop1
        interpret = gam1
        data_it = data_cop1
        func_c = pred_cop1
        func_gam = interpret_cop1
    elif met == "Cop clim 2":
        model = cop2
        interpret = gam2
        data_it = data_cop2
        func_c = pred_cop2
        func_gam = interpret_cop2
    else:
        model = ppue
        interpret = gam3
        data_it = data_ppue
        func_c = pred_ppue
        func_gam = interpret_ppue


with top_left_cell2:
    nb_step_predict = st.slider(
    "Select the number of prediction steps (1 step = 10 min | max step = 50)",
    min_value=1,
    max_value=50,
    value=50
    )

with top_left_cell3:
    try:
        if met != "Ppue":
            start_date, end_date = st.date_input(
            "Select the past data to be graphed before starting the predictions (1 day interval is recommended)",
            value=(dt.date(2026, 2, 1), dt.date(2026, 2, 2)),
            min_value=dt.date(2025, 10, 13),
            max_value=dt.date(2026, 2, 27)
            )
        else:
            start_date, end_date = st.date_input(
            "Select the past data to be graphed before starting the predictions (1 day interval is recommended)",
            value=(dt.date(2026, 2, 1), dt.date(2026, 2, 2)),
            min_value=dt.date(2025, 12, 4),
            max_value=dt.date(2026, 2, 27)
            )
    except ValueError:
        pass

#####################################################################

if met != "Ppue":
    selected_metric = metrics[met]
    columns_to_plot = ["index_time"] + selected_metric
    df = get_data_viz(data_global, columns_to_plot)
    df_hist = get_data_viz(data_global, columns_to_plot)
    df["index_time"] = pd.to_datetime(df["index_time"])
    df_hist["index_time"] = pd.to_datetime(df_hist["index_time"])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
else:
    selected_metric = metrics[met]
    columns_to_plot = ["_time"] + selected_metric
    df = get_data_viz(data_ppue, columns_to_plot)
    df_hist = get_data_viz(data_ppue, columns_to_plot)
    df_form = get_data_viz(data_ppue, columns_to_plot)
    df = df.rename(columns={"_time": "index_time"})
    df_hist = df_hist.rename(columns={"_time": "index_time"})
    df_form = df_form.rename(columns={"_time": "index_time"})
    df["index_time"] = pd.to_datetime(df["index_time"])
    df_hist["index_time"] = pd.to_datetime(df_hist["index_time"])
    df_form["index_time"] = pd.to_datetime(df_form["index_time"])
    df = df.drop(["OnduleurPuissance_1"], axis = 1)
    df_hist = df_hist.drop(["OnduleurPuissance_1"], axis = 1)



try:

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
        
    mask1 = (df["index_time"] >= start_date) & (df["index_time"] <= end_date)
    mask2 = (df_hist["index_time"] >= end_date)
    df_hist = df_hist.loc[mask2].iloc[:nb_step_predict,:]
    df = df.loc[mask1]
    if met == "Ppue":
        df_form = df_form.loc[mask2]

    pred = func_c(data_it, model, end_date, steps=nb_step_predict)

    actions_opt = get_control(data_control, end_date, steps=nb_step_predict+1)

    df_interpret_10min = func_gam(data_it, interpret, end_date, steps=nb_step_predict)
except NameError:
    pass



#####################################################################
                    ##### mise à jour viz predictions ######  

hist_x = df["index_time"]
hist_y = df[selected_metric[0]]
df_hist_x = df_hist["index_time"]
df_hist_y = df_hist[selected_metric[0]]

if "pred_x" not in st.session_state:
    st.session_state.pred_x = []

if "pred_y" not in st.session_state:
    st.session_state.pred_y = []

if "historical_y" not in st.session_state:
    st.session_state.historical_y = []

if "historical_x" not in st.session_state:
    st.session_state.historical_x = []

if "act_t" not in st.session_state:
    st.session_state.act_t = []

if "contrib_t" not in st.session_state:
     st.session_state.contrib_t = []

if "idx_y" not in st.session_state:
    st.session_state.idx_y = 0


if st.button("🔮 Next prediction step (from xgb)"):

    # dernier point historique ou prédiction
    if len(st.session_state.pred_y) == 0:
        last_y = hist_y.iloc[-1]
    else:
        last_y = st.session_state.pred_y[-1]

    # sécurité index
    if st.session_state.idx_y < len(pred):
        if met == "Ppue":
            delta = 1 + pred[st.session_state.idx_y]/df_form["OnduleurPuissance_1"].iloc[st.session_state.idx_y]
        else:
            delta = pred[st.session_state.idx_y]
        act = actions_opt.iloc[st.session_state.idx_y,:]
        contrib = df_interpret_10min.iloc[st.session_state.idx_y,:]

        historical_y = df_hist_y.iloc[st.session_state.idx_y]
        historical_x = df_hist_x.iloc[st.session_state.idx_y]

    else:
        if met == "Ppue":
            delta = 1 + pred[-1]/df_form["OnduleurPuissance_1"].iloc[-1]
        else:
            delta = pred[-1]

        act = actions_opt.iloc[-1,:]
        contrib = df_interpret_10min.iloc[-1,:]

        historical_y = df_hist_y.iloc[-1]
        historical_x = df_hist_x.iloc[-1]


    new_pred = delta
    new_action = act
    new_contrib = contrib
    new_hist = historical_y

    # temps
    if len(st.session_state.pred_x) == 0:
        last_time = hist_x.iloc[-1]
    else:
        last_time = st.session_state.pred_x[-1]

    new_time = last_time + pd.Timedelta(minutes=10)
    new_time_hist = historical_x

    # update state
    st.session_state.pred_y.append(new_pred)
    st.session_state.pred_x.append(new_time)
    st.session_state.act_t.append(new_action)
    st.session_state.contrib_t.append(new_contrib)
    st.session_state.historical_y.append(new_hist)
    st.session_state.historical_x.append(new_time_hist)
    st.session_state.idx_y += 1

if st.button("🔄 Refresh"):
    st.session_state.clear()
    st.rerun()

fig = go.Figure()

with cols[1]:
  
    fig.add_trace(go.Scatter(
        x=hist_x,
        y=hist_y,
        name=f"historical {met} ",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=st.session_state.historical_x,
        y=st.session_state.historical_y,
        name=f"current {met}",
        line=dict(color="purple", dash = "dash")
    ))

    fig.add_trace(go.Scatter(
        x=st.session_state.pred_x,
        y=st.session_state.pred_y,
        name=f"forecasts {met}",
        line=dict(color="red", dash="dash")
    ))


    fig.update_layout(
        title=f"Historical / Forecasts for {met}",
        xaxis_title="time",
        yaxis_title=f"{met}",
        legend_title="Legend"
    )

    st.plotly_chart(fig, use_container_width=True)

#####################################################################
            ##### interprétations pour augmentation à 10 min ######  



st.divider()

st.markdown("### 🧠 Interpretation for the next 10 min : get the ranking of contributions (from gam model)")

if len(st.session_state.contrib_t) != 0:

    st.text(f"Chosen Metric : {met} | Current Date : {st.session_state.historical_x[-1]} | Forecast : {st.session_state.pred_x[-1]}")
    st.write(st.session_state.contrib_t[-1])

else:
    st.text("Make predictions to get the ranking of contributions")


#####################################################################

    #### affichae actions à prendre pour les 10 prochaines minutes ######  

st.divider()

st.markdown("### 🎛️ Optimal control (free-cooling and chiller valves) for the next 10 min (from CQL(H) actor-critic)")

if len(st.session_state.act_t) != 0:
    st.write(st.session_state.act_t[-1][["index_time","free-cooling valves clim 1","chiller valves clim 1","free-cooling valves clim 2","chiller valves clim 2"]])
else:
    st.text("Make predictions to get optimal actions")

