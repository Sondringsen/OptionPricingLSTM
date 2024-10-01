import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta
from datetime import datetime
from pathlib import Path

def read_file(file):
    """Read a single file and return a dataframe"""
    return pd.read_csv(file, skipinitialspace=True)

def df_to_xy_MLP(df, model_type):
    """Transforms a dataframe into two arrays of explanatory variables x and explained variables y"""
    if model_type == "MLP":
        dx = df[["Underlying_last", "Strike", "Ttl", "Volatility", "R"]]
    elif model_type == "MLP-GARCH":
        dx = df[["Underlying_last", "Strike", "Ttl", "Volatility_GJR_GARCH", "R"]]
    dy = df["Price"]
    array_x, array_y = dx.to_numpy().astype(np.float32), dy.to_numpy().astype(np.float32)
    return array_x, array_y

def df_to_xy_LSTM(df, num_features, seq_length, num_outputs):
    """Transforms a dataframe into two arrays of explanatory variables x and explained variables y"""
    array = df.to_numpy()
    array_x, array_y = array[:, -num_features*seq_length - num_outputs:-num_outputs].astype(np.float32), array[:,-num_outputs:].astype(np.float32)
    return array_x, array_y

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


def make_train_val_test(model_type, return_first=False):
    """Makes the train, val, and test set for the rolling window approach.
    
    Args:
        model_type: one of LSTM, MLP, MLP-GARCH
        return_first: used when only the first set of train, val, and test should be returned.
        
    Note: return_first is only true when doing hyperparameter_sweep
    """
    first_year = 2019
    last_year = 2021

    if model_type in ["MLP", "LSTM"]:
        features = ["Underlying_last", "Strike", "Ttl", "Volatility", "R"]
    elif model_type == "MLP-GARCH":
        features = ["Underlying_last", "Strike", "Ttl", "Volatility_GJR_GARCH", "R"]

    seq_length = 5
    num_features = len(features)

    file = f"data/processed/{first_year}-{last_year}_underlying-strike_only-price.csv"
    df_read = read_file(file)

    num_models = 12
    seq_length = 5
    num_features = len(features)
    num_outputs = 1

    df_read_lags = lag_features(df_read, features, seq_length)

    train_val_test = []

    month = 4
    year = 0
    for i in range(num_models):
        if month == 13:
            year += 1
            month = 1
        train_start = datetime(2020 + year, month, 1)
        val_start = train_start + relativedelta(months=8)
        test_start = val_start + relativedelta(months=1)
        test_end = test_start + relativedelta(months=1)

        month += 1

        df_train_orginal = df_read_lags.loc[(df_read_lags.loc[:, "Quote_date"] >= str(train_start)) & (df_read_lags.loc[:, "Quote_date"] < str(val_start)), :]
        df_val_orginal = df_read_lags.loc[(df_read_lags.loc[:, "Quote_date"] >= str(val_start)) & (df_read_lags.loc[:, "Quote_date"] < str(test_start)), :]
        df_test_orginal = df_read_lags.loc[(df_read_lags.loc[:, "Quote_date"] >= str(test_start)) & (df_read_lags.loc[:, "Quote_date"] < str(test_end)), :]

        if model_type in ["MLP", "MLP-GARCH"]:
            train_x_org, train_y_org = df_to_xy_MLP(df_train_orginal, model_type)
            val_x_org, val_y_org = df_to_xy_MLP(df_val_orginal, model_type)
            test_x_org, test_y_org = df_to_xy_MLP(df_test_orginal, model_type)
        if model_type == "LSTM":
            train_x_org, train_y_org = df_to_xy_LSTM(df_train_orginal, num_features, seq_length, num_outputs)
            val_x_org, val_y_org = df_to_xy_LSTM(df_val_orginal, num_features, seq_length, num_outputs)
            test_x_org, test_y_org = df_to_xy_LSTM(df_test_orginal, num_features, seq_length, num_outputs)

        scaler = MinMaxScaler()
        train_x_scaled = scaler.fit_transform(train_x_org)
        val_x_scaled = scaler.transform(val_x_org)
        test_x_scaled = scaler.transform(test_x_org)

        if model_type in ["MLP", "MLP-GARCH"]:
            train_x_scaled = np.reshape(train_x_scaled, (len(train_x_scaled), num_features))
            val_x_scaled = np.reshape(val_x_scaled, (len(val_x_scaled), num_features))
            test_x_scaled = np.reshape(test_x_scaled, (len(test_x_scaled), num_features))
        elif model_type == "LSTM":
            train_x_scaled = np.reshape(train_x_scaled, (len(train_x_scaled), seq_length, num_features))
            val_x_scaled = np.reshape(val_x_scaled, (len(val_x_scaled), seq_length, num_features))
            test_x_scaled = np.reshape(test_x_scaled, (len(test_x_scaled), seq_length, num_features))

        if return_first:
            return train_x_scaled, train_y_org, val_x_scaled, val_y_org

        train_val_test.append(((train_x_scaled, train_y_org), (val_x_scaled, val_y_org), (test_x_scaled, test_y_org)))
    
    return train_val_test

def add_naive(df, features, seq_length):
    """Transforms a raw 2D dataframe of option data into 2D dataframe ofsequence data.
    Last 2 indexes per sequence are bid and ask price. The len(features)*seq_length
    features before are sequences of features"""
    df = df.sort_values(["Expire_date", "Strike", "Ttl"], ascending = [True, True, False])
    
    df["Naive"] = df["Price"].shift(1)

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

def save_predictions(predictions, model_type):
    first_year = 2019
    last_year = 2021

    file = f"data/processed/{first_year}-{last_year}_underlying-strike_only-price.csv"
    df_read = read_file(file)
    if model_type in ["LSTM", "MLP"]:
        df_read_lags = lag_features(df_read, ["Underlying_last", "Strike", "Ttl", "Volatility", "R"], 5)
    elif model_type == "MLP-GARCH":
        df_read_lags = lag_features(df_read, ["Underlying_last", "Strike", "Ttl", "Volatility_GJR_GARCH", "R"], 5)

    df_test_whole = df_read_lags.loc[df_read_lags.loc[:, "Quote_date"] >= "2021-01-01", :]
    df_test_whole["Prediction"] = predictions

    time = datetime.now()
    time = time.strftime("%m-%d_%H-%M")

    if model_type == "LSTM":
        filename = f"models/predictions/{last_year}_predictions_{time}_LSTM.csv"
    elif model_type == "MLP":
        filename = f"models/predictions/{last_year}_predictions_{time}.csv"
    elif model_type == "MLP-GARCH":
        filename = f"models/predictions/{last_year}_predictions_{time}_GARCH.csv"

    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok = True)
    df_test_whole.to_csv(filename)


