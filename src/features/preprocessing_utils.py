import numpy as np
import pandas as pd
from arch import arch_model

def calculate_volatility(df):
    """Calculate underlying 90 days annualized moving average volatility from dataset of options"""
    df_vol = df[["Quote_date", "Underlying_last"]].drop_duplicates()
    df_vol["Volatility"] = np.log(df_vol["Underlying_last"] / df_vol["Underlying_last"].shift()).rolling(90).std()*(252**0.5)
    return df_vol[["Quote_date", "Volatility"]]


def calculate_volatility_GJR_GARCH(df):
    """Makes a time series of volatilities using GJR-GARCH."""
    df_vol = df[["Quote_date", "Underlying_last"]].drop_duplicates()
    df_vol["Log_returns"] = 100*(5 + np.log(df_vol["Underlying_last"] / df_vol["Underlying_last"].shift()))

    predicted_volatilities = {"Quote_date": [], "Volatility_GJR_GARCH": []}

    for i, date in enumerate(df_vol["Quote_date"].values):
        predicted_volatilities["Quote_date"].append(date)

        if i < 90:
            predicted_volatilities["Volatility_GJR_GARCH"].append(np.nan)
            continue

        prev_vol = df_vol.loc[df_vol.loc[:, "Quote_date"] < date, ["Log_returns"]].dropna()
        gjr_garch = arch_model(prev_vol, vol="GARCH", p=1, o=1, q=1, dist="studentst")
        gjr_garch_fit = gjr_garch.fit(disp=False)
        forecast = np.sqrt(252*(gjr_garch_fit.forecast(horizon=1).variance.values[0][0]))/100
        predicted_volatilities["Volatility_GJR_GARCH"].append(forecast)

    return pd.DataFrame(predicted_volatilities)

def process_options(df_opt, call = True):
    """Cleans up column names and add time to live (Ttl) and volatility column to the dataframe"""
    keys = {key: key[key.find("[")+1:key.find("]")][0] + key[key.find("[")+1:key.find("]")][1:].lower()  for key in df_opt.keys()}
    df_opt = df_opt.rename(columns=keys)

    if call:
        keys = {"C_ask": "Ask", "C_bid": "Bid"}
    else:
        keys = {"P_ask": "Ask", "P_bid": "Bid"}
    df_opt = df_opt.rename(columns=keys)

    df_opt["Quote_date"] = pd.to_datetime(df_opt["Quote_date"])
    df_opt["Expire_date"] = pd.to_datetime(df_opt["Expire_date"])
    df_opt["Ttl"] = df_opt.apply(lambda row: (row.Expire_date - row.Quote_date).days, axis = 1)
    df_opt["Price"] = (df_opt["Ask"] + df_opt["Bid"])/2
     
    df_vol = calculate_volatility(df_opt)
    df_opt = pd.merge(df_opt, df_vol, on ="Quote_date", how = "left")

    df_vol = calculate_volatility_GJR_GARCH(df_opt)
    df_opt = pd.merge(df_opt, df_vol, on ="Quote_date", how = "left")

    columns = ["Quote_date", "Expire_date", "Price", "Underlying_last", "Strike", "Ttl", "Volatility", "Volatility_GJR_GARCH"]
    df_opt = df_opt[columns]
    df_opt = df_opt[(df_opt["Ttl"] != 0) & (df_opt["Ttl"] <= 365*3)]
    return df_opt[columns]

def process_rates(df_r):
    """Renames date column and rate duration to days"""
    df_r["Date"] = pd.to_datetime(df_r["Date"])
    keys = {"Date" : "Quote_date",
            "1 Mo": 30,
            "3 Mo": 90,
            "6 Mo": 180,
            "1 Yr": 365,
            "2 Yr": 365*2,
            "3 Yr": 365*3,
            "5 Yr": 365*5,
            "7 Yr": 365*7,
            "10 Yr": 365*10}
    df_r = df_r.rename(columns = keys)
    return df_r[keys.values()]


def lag_features(df, features, seq_length):
    """Transforms a raw 2D dataframe of option data into 2D dataframe ofsequence data.
    Last 2 indexes per sequence are bid and ask price. The len(features)*seq_length
    features before are sequences of features"""
    df = df.sort_values(["Expire_date", "Strike", "Ttl"], ascending = [True, True, False])

    for step in range(seq_length)[::-1]:
        for feature in features:
            df[feature + "-" + str(step)] = df[feature].shift(step)
    
    df["Check_strike"] = df["Strike"] == df["Strike"].shift(seq_length-1)
    df["Check_expire"] = df["Expire_date"] == df["Expire_date"].shift(seq_length-1)
    df = df[(df["Check_strike"] == True) & (df["Check_expire"] == True)]
    df = df.drop(["Check_strike", "Check_expire"], axis=1)

    df["Price_last"] = df["Price"]
    df = df.sort_values(["Quote_date"], ascending = [True])
    return df