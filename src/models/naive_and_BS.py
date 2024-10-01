from model_utils import read_file, add_naive
import pandas as pd
import numpy as np
from scipy.stats import norm

def d1(S,K,T,r,sigma):
    x1 = S.apply(lambda x : np.log(x)) - K.apply(lambda x : np.log(x))
    x2 = (r + ((sigma.apply(lambda x : x**2)) / 2)) * T
    x3 = sigma * T.apply(lambda x: np.sqrt(x))
    return  (x1 + x2) / x3

def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma) - sigma * T.apply(lambda x : np.sqrt(x))  

def bs_call(S,K,T,r,sigma):
    T = T/365
    r = r/100
    return S * d1(S,K,T,r,sigma).apply(lambda x : norm.cdf(x)) - K * (-r*T).apply(lambda x : np.exp(x)) * d2(S,K,T,r,sigma).apply(lambda x : norm.cdf(x))

def main():
    file = f"model/predictions/2021_predictions_09-30_10-22.csv"
    df_options = read_file(file)
    df_options = df_options.rename(columns={"Prediction": "MLP"})
    df_options["Expire_date"] = pd.to_datetime(df_options["Quote_date"]) + pd.to_timedelta(df_options["Ttl"], unit="D")

    df_merge = read_file(f"models/predictions/2021_predictions_09-30_14-55_GARCH.csv")[["Quote_date", "Strike", "Ttl", "Prediction"]]
    df_options = pd.merge(df_options, df_merge, how="inner", on=["Quote_date", "Strike", "Ttl"])
    df_options = df_options.rename(columns={"Prediction": "MLP-GARCH"})

    df_merge = read_file("models/predictions/2021_predictions_09-26_21-44_LSTM.csv")[["Quote_date", "Strike", "Ttl", "Prediction"]]
    df_options = pd.merge(df_options, df_merge, how="inner", on=["Quote_date", "Strike", "Ttl"])
    df_options = df_options.rename(columns={"Prediction": "LSTM"})

    features = ["Underlying_last", "Strike", "Ttl", "Volatility", "R"]
    seq_length = 5
    num_features = len(features)
    num_outputs = 1

    df_options = add_naive(df_options, features, seq_length)

    df_options["BS"] = bs_call(df_options["Underlying_last"], df_options["Strike"], df_options["Ttl"], df_options["R"], df_options["Volatility"])
    df_options["BS-GJR-GARCH"] = bs_call(df_options["Underlying_last"], df_options["Strike"], df_options["Ttl"], df_options["R"], df_options["Volatility_GJR_GARCH"])

    df_options = df_options[["Quote_date", "Underlying_last", "Strike", "Ttl", "Volatility", "R", "Price", "Volatility_GJR_GARCH", "Naive", "MLP", "MLP-GARCH", "LSTM", "BS", "BS-GJR-GARCH"]]
    df_options['Quote_date'] = pd.to_datetime(df_options.Quote_date, format='%Y-%m-%d')
    df_options = df_options[df_options.Quote_date.dt.year == 2021]

    df_options.to_csv('models/predictions/all_predictions.csv', encoding='utf-8', index=False)
