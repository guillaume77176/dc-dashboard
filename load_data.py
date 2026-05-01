import pandas as pd 

def get_data_viz(data, col):
    return data[col]

def get_X_cop1(data, end_date, steps):
    data["_time"] = pd.to_datetime(data["_time"])
    end_date = pd.to_datetime(end_date)
    mask = (data["_time"] > end_date)
    data = data.loc[mask]
    data = data.iloc[0:steps,:]
    data = data.rename(columns={"_time": "index_time"})
    data = data.drop(["COP_v1_clim_1", "index_time"], axis = 1)
    return data

def get_X_cop2(data, end_date, steps):
    data["_time"] = pd.to_datetime(data["_time"])
    end_date = pd.to_datetime(end_date)
    mask = (data["_time"] > end_date)
    data = data.loc[mask]
    data = data.iloc[0:steps,:]
    data = data.rename(columns={"_time": "index_time"})
    data = data.drop(["COP_v1_clim_2", "index_time"], axis = 1)
    return data

def get_control(data, end_date, steps):
    data["index_time"] = pd.to_datetime(data["index_time"], utc=True)
    end_date = pd.to_datetime(end_date, utc=True)
    mask = (data["index_time"] >= end_date)
    data = data.loc[mask]
    data = data.iloc[:,:]
    return data
    

