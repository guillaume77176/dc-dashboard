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
import joblib


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

@st.cache_data
def get_data_viz_ch():
    data = pd.read_csv("https://minio.lab.sspcloud.fr/guillaume176/diffusion/data.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=YZGBQ1SDECI0HYL1PZIO%2F20260430%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260430T182446Z&X-Amz-Expires=604800&X-Amz-Security-Token=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJZWkdCUTFTREVDSTBIWUwxUFpJTyIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzc3NTcyMzczLCJhenAiOiJvbnl4aWEiLCJjbmYiOnsiamt0IjoiYjFoengtSjRKOUxJbjRuLTJ0WFlWUGxFeUZtWEFZTkdndEZIRHZMaDNLNCJ9LCJlbWFpbCI6Imd1aWxsYXVtZS5yb3VzdGFuQGVuc2FlLmZyIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImV4cCI6MTc3ODE3NzE4MiwiZmFtaWx5X25hbWUiOiJSb3VzdGFuIiwiZ2l2ZW5fbmFtZSI6Ikd1aWxsYXVtZSIsImdyb3VwcyI6WyJVU0VSX09OWVhJQSJdLCJpYXQiOjE3Nzc1NzIzODIsImlzcyI6Imh0dHBzOi8vYXV0aC5sYWIuc3NwY2xvdWQuZnIvYXV0aC9yZWFsbXMvc3NwY2xvdWQiLCJqdGkiOiJvbnJ0cnQ6NTYyYzgzMTctNTQ4MC04YTIxLTBjMDUtZTQxZDA0ZDVkZDAyIiwibG9jYWxlIjoiZnIiLCJuYW1lIjoiR3VpbGxhdW1lIFJvdXN0YW4iLCJwb2xpY3kiOiJzdHNvbmx5IiwicHJlZmVycmVkX3VzZXJuYW1lIjoiZ3VpbGxhdW1lMTc2IiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiI1MmRiODA2Yy0xY2M3LTIxNWEtNGIxNi00ZTYxNWFlYjYyNzMiLCJzdWIiOiI5Mjc4NzJjYi03NjgyLTRkNDAtYjliNy04M2IyYjk3YWRmMjgiLCJ0eXAiOiJEUG9QIn0.clL9Hed7aBw1UeEYCE06MNlh3rXwR3PCuRW0uVjxkFGP9h6JWp2KnblNj33jz1WuUyuCic3hF03FDhGuUP9GWQ&X-Amz-Signature=48a9623ffed4b959ea8502f5fd3318fcbfb03c9c05501308cfc717ef7bbcd152&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject")
    return data

@st.cache_data
def get_data_cop1():
    data = pd.read_csv("https://minio.lab.sspcloud.fr/guillaume176/diffusion/dataset_xgboost_ready_COP_v1_clim_1.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=YZGBQ1SDECI0HYL1PZIO%2F20260430%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260430T214341Z&X-Amz-Expires=604800&X-Amz-Security-Token=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJZWkdCUTFTREVDSTBIWUwxUFpJTyIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzc3NTcyMzczLCJhenAiOiJvbnl4aWEiLCJjbmYiOnsiamt0IjoiYjFoengtSjRKOUxJbjRuLTJ0WFlWUGxFeUZtWEFZTkdndEZIRHZMaDNLNCJ9LCJlbWFpbCI6Imd1aWxsYXVtZS5yb3VzdGFuQGVuc2FlLmZyIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImV4cCI6MTc3ODE3NzE4MiwiZmFtaWx5X25hbWUiOiJSb3VzdGFuIiwiZ2l2ZW5fbmFtZSI6Ikd1aWxsYXVtZSIsImdyb3VwcyI6WyJVU0VSX09OWVhJQSJdLCJpYXQiOjE3Nzc1NzIzODIsImlzcyI6Imh0dHBzOi8vYXV0aC5sYWIuc3NwY2xvdWQuZnIvYXV0aC9yZWFsbXMvc3NwY2xvdWQiLCJqdGkiOiJvbnJ0cnQ6NTYyYzgzMTctNTQ4MC04YTIxLTBjMDUtZTQxZDA0ZDVkZDAyIiwibG9jYWxlIjoiZnIiLCJuYW1lIjoiR3VpbGxhdW1lIFJvdXN0YW4iLCJwb2xpY3kiOiJzdHNvbmx5IiwicHJlZmVycmVkX3VzZXJuYW1lIjoiZ3VpbGxhdW1lMTc2IiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiI1MmRiODA2Yy0xY2M3LTIxNWEtNGIxNi00ZTYxNWFlYjYyNzMiLCJzdWIiOiI5Mjc4NzJjYi03NjgyLTRkNDAtYjliNy04M2IyYjk3YWRmMjgiLCJ0eXAiOiJEUG9QIn0.clL9Hed7aBw1UeEYCE06MNlh3rXwR3PCuRW0uVjxkFGP9h6JWp2KnblNj33jz1WuUyuCic3hF03FDhGuUP9GWQ&X-Amz-Signature=dc5db77d30a186a199bfc0cc34484ba746043c6cf267db0b3feeca57057e9615&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject")
    return data

@st.cache_data
def get_data_cop2():
    data = pd.read_csv("https://minio.lab.sspcloud.fr/guillaume176/diffusion/dataset_xgboost_ready_COP_v1_clim_2.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=YZGBQ1SDECI0HYL1PZIO%2F20260430%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260430T214922Z&X-Amz-Expires=604800&X-Amz-Security-Token=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJZWkdCUTFTREVDSTBIWUwxUFpJTyIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzc3NTcyMzczLCJhenAiOiJvbnl4aWEiLCJjbmYiOnsiamt0IjoiYjFoengtSjRKOUxJbjRuLTJ0WFlWUGxFeUZtWEFZTkdndEZIRHZMaDNLNCJ9LCJlbWFpbCI6Imd1aWxsYXVtZS5yb3VzdGFuQGVuc2FlLmZyIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImV4cCI6MTc3ODE3NzE4MiwiZmFtaWx5X25hbWUiOiJSb3VzdGFuIiwiZ2l2ZW5fbmFtZSI6Ikd1aWxsYXVtZSIsImdyb3VwcyI6WyJVU0VSX09OWVhJQSJdLCJpYXQiOjE3Nzc1NzIzODIsImlzcyI6Imh0dHBzOi8vYXV0aC5sYWIuc3NwY2xvdWQuZnIvYXV0aC9yZWFsbXMvc3NwY2xvdWQiLCJqdGkiOiJvbnJ0cnQ6NTYyYzgzMTctNTQ4MC04YTIxLTBjMDUtZTQxZDA0ZDVkZDAyIiwibG9jYWxlIjoiZnIiLCJuYW1lIjoiR3VpbGxhdW1lIFJvdXN0YW4iLCJwb2xpY3kiOiJzdHNvbmx5IiwicHJlZmVycmVkX3VzZXJuYW1lIjoiZ3VpbGxhdW1lMTc2IiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiI1MmRiODA2Yy0xY2M3LTIxNWEtNGIxNi00ZTYxNWFlYjYyNzMiLCJzdWIiOiI5Mjc4NzJjYi03NjgyLTRkNDAtYjliNy04M2IyYjk3YWRmMjgiLCJ0eXAiOiJEUG9QIn0.clL9Hed7aBw1UeEYCE06MNlh3rXwR3PCuRW0uVjxkFGP9h6JWp2KnblNj33jz1WuUyuCic3hF03FDhGuUP9GWQ&X-Amz-Signature=d541068f3c10f2c6bcaa0bfc04ba50d224f748226c4e4599edf8b6354dd8c1de&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject")
    return data

@st.cache_data
def get_data_control():
    data = pd.read_csv("https://minio.lab.sspcloud.fr/guillaume176/diffusion/opt_actons.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=YZGBQ1SDECI0HYL1PZIO%2F20260501%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260501T012228Z&X-Amz-Expires=604800&X-Amz-Security-Token=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJZWkdCUTFTREVDSTBIWUwxUFpJTyIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzc3NTcyMzczLCJhenAiOiJvbnl4aWEiLCJjbmYiOnsiamt0IjoiYjFoengtSjRKOUxJbjRuLTJ0WFlWUGxFeUZtWEFZTkdndEZIRHZMaDNLNCJ9LCJlbWFpbCI6Imd1aWxsYXVtZS5yb3VzdGFuQGVuc2FlLmZyIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImV4cCI6MTc3ODE3NzE4MiwiZmFtaWx5X25hbWUiOiJSb3VzdGFuIiwiZ2l2ZW5fbmFtZSI6Ikd1aWxsYXVtZSIsImdyb3VwcyI6WyJVU0VSX09OWVhJQSJdLCJpYXQiOjE3Nzc1NzIzODIsImlzcyI6Imh0dHBzOi8vYXV0aC5sYWIuc3NwY2xvdWQuZnIvYXV0aC9yZWFsbXMvc3NwY2xvdWQiLCJqdGkiOiJvbnJ0cnQ6NTYyYzgzMTctNTQ4MC04YTIxLTBjMDUtZTQxZDA0ZDVkZDAyIiwibG9jYWxlIjoiZnIiLCJuYW1lIjoiR3VpbGxhdW1lIFJvdXN0YW4iLCJwb2xpY3kiOiJzdHNvbmx5IiwicHJlZmVycmVkX3VzZXJuYW1lIjoiZ3VpbGxhdW1lMTc2IiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiI1MmRiODA2Yy0xY2M3LTIxNWEtNGIxNi00ZTYxNWFlYjYyNzMiLCJzdWIiOiI5Mjc4NzJjYi03NjgyLTRkNDAtYjliNy04M2IyYjk3YWRmMjgiLCJ0eXAiOiJEUG9QIn0.clL9Hed7aBw1UeEYCE06MNlh3rXwR3PCuRW0uVjxkFGP9h6JWp2KnblNj33jz1WuUyuCic3hF03FDhGuUP9GWQ&X-Amz-Signature=c500ed7fa1a1494ded1d3ba8d2b22849081b829efed55e61548d6903da95d669&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject")
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
data_control = get_data_control()

cop1 = model_cop1()
cop2 = model_cop2()
gam1 = model_interpretc1()
gam2 = model_interpretc2()


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


df = get_data_viz(data_global, columns_to_plot)
df["index_time"] = pd.to_datetime(df["index_time"])



try:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    mask1 = (df["index_time"] >= start_date) & (df["index_time"] <= end_date)
    mask2 = (df["index_time"] > end_date)
    df_hist = df.loc[mask2].iloc[:nb_step_predict,:]
    df = df.loc[mask1]

    pred = func_c(data_it, model, end_date, steps=nb_step_predict)

    actions_opt = get_control(data_control, end_date, steps=nb_step_predict)

    df_interpret_10min = func_gam(data_it, interpret, end_date, steps=nb_step_predict)
except NameError:
    pass



#####################################################################
                    ##### mise à jour viz predictions ######  

hist_x = df["index_time"]
hist_y = df[selected_metric]
df_hist = df_hist[selected_metric]

if "pred_x" not in st.session_state:
    st.session_state.pred_x = []

if "pred_y" not in st.session_state:
    st.session_state.pred_y = []

if "historical_y" not in st.session_state:
    st.session_state.historical_y = []

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
        delta = pred[st.session_state.idx_y]
        act = actions_opt.iloc[st.session_state.idx_y,:]
        contrib = df_interpret_10min.iloc[st.session_state.idx_y,:]
        historical_y = df_hist.iloc[st.session_state.idx_y,0]
    else:
        delta = pred[-1]  # fallback
        act = actions_opt.iloc[-1,:]
        contrib = df_interpret_10min.iloc[-1,:]
        historical_y = df_hist.iloc[-1,0]

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

    # update state
    st.session_state.pred_y.append(new_pred)
    st.session_state.pred_x.append(new_time)
    st.session_state.act_t.append(new_action)
    st.session_state.contrib_t.append(new_contrib)
    st.session_state.historical_y.append(new_hist)
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

    fig.add_trace(go.Scatter(
        x=st.session_state.pred_x,
        y=st.session_state.historical_y,
        line=dict(color="blue")
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

if len(st.session_state.contrib_t) != 0:
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

