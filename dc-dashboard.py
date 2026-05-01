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
from load_model import pred_cop1, pred_cop2, interpret_cop1, interpret_cop2
import plotly.graph_objects as go
import datetime as dt


st.set_page_config(
    page_title="Dashboard Data center id X",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

"""
# :material/query_stats: Dashboard Simulation - Data center X (first containment)
"""

""  
st.text("Predict metrics of interest, retrieve suspicious commands, and predict the best decisions to make for modifying free-cooling and chiller valves")
cols = st.columns([1, 3])


#####################################################################

st.divider()

st.markdown("### 🔮 Vizualise the predictions")

# Time horizon selector
metrics = {
    "Cop clim 1" : "COP_v1_clim_1",
    "Cop clim 2" : "COP_v1_clim_2",
    "Ppue clim 1" : "pp c1",
    "Ppue clim 2" : "pp c2",
    
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
        model = pred_cop1
        interpret = interpret_cop1
    elif met == "Cop clim 2":
        model = pred_cop2
        interpret = interpret_cop2


with top_left_cell2:
    nb_step_predict = st.slider(
    "Select the number of prediction steps (1 step = 10 min | max step = 50)",
    min_value=1,
    max_value=50,
    value=50
    )

with top_left_cell3:
    try:
        start_date, end_date = st.date_input(
        "Select the past data to be graphed before starting the predictions (1 day interval is recommended)",
        value=(dt.date(2026, 2, 1), dt.date(2026, 3, 1)),
        min_value=dt.date(2025, 10, 13),
        max_value=dt.date(2026, 3, 6)
        )
    except ValueError:
        pass

#####################################################################

selected_metric = metrics[met]
columns_to_plot = ["index_time"] + [selected_metric]

df = get_data_viz(columns_to_plot)
df["index_time"] = pd.to_datetime(df["index_time"])



try:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    mask = (df["index_time"] >= start_date) & (df["index_time"] <= end_date)
    df = df.loc[mask]

    pred = model(end_date, steps=nb_step_predict)

    actions_opt = get_control(end_date, steps=nb_step_predict)

except NameError:
    pass

if end_date:
    df_interpret_10min = interpret(end_date)

#####################################################################
                    ##### mise à jour viz predictions ######  

hist_x = df["index_time"]
hist_y = df[selected_metric]

if "pred_x" not in st.session_state:
    st.session_state.pred_x = []

if "pred_y" not in st.session_state:
    st.session_state.pred_y = []

if "act_t" not in st.session_state:
    st.session_state.act_t = []

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
        delta = pred[st.session_state.idx_y]
        act = actions_opt.iloc[st.session_state.idx_y,:]
    else:
        delta = pred[-1]  # fallback
        act = actions_opt.iloc[-1,:]

    new_pred = delta
    new_action = act

    # temps
    if len(st.session_state.pred_x) == 0:
        last_time = hist_x.iloc[-1]
    else:
        last_time = st.session_state.pred_x[-1]

    new_time = last_time + pd.Timedelta(minutes=10)

    # update state
    st.session_state.pred_y.append(new_pred)
    st.session_state.pred_x.append(new_time)
    st.session_state.act_t.append(new_action)
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

st.markdown("### 🧠 Interpretation for the next 10 min : get the ranking of contribution (from gam model)")

if df_interpret_10min is not None:
    st.write(df_interpret_10min[["variable","contribution"]])


#####################################################################

    #### affichae actions à prendre pour les 10 prochaines minutes ######  

st.divider()

st.markdown("### 🎛️ Optimal control (free-cooling and chiller valves) for the next 10 min (from CQL(H) actor-critic)")

st.write(st.session_state.act_t)

