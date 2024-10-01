from models.lstm import create_model as create_model_lstm
from models.mlp import create_model as create_model_mlp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Hyperparameter tuning


import wandb
from wandb.integration.keras import WandbMetricsLogger
from keras.callbacks import EarlyStopping

from datetime import datetime
from dateutil.relativedelta import relativedelta

import os

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
    
    # Adding lag for naive benchmarking
    #df["Naive"] = df["Price"].shift(1)

    for step in range(seq_length)[::-1]:
        for feature in features:
            df[feature + "-" + str(step)] = df[feature].shift(step)
    
    df["Check_strike"] = df["Strike"] == df["Strike"].shift(seq_length-1)
    df["Check_expire"] = df["Expire_date"] == df["Expire_date"].shift(seq_length-1)
    df = df[(df["Check_strike"] == True) & (df["Check_expire"] == True)]
    df = df.drop(["Check_strike", "Check_expire"], axis=1)
    #df[["Bid_strike_last", "Ask_strike_last"]] = df[["Bid_strike", "Ask_strike"]]
    #df[["Bid_last", "Ask_last"]] = df[["Bid", "Ask"]]
    df["Price_last"] = df["Price"]
    df = df.sort_values(["Quote_date"], ascending = [True])
    return df

def hyperparam_sweep(sweep_configuration, model_type):
    os.environ["WANDB_NOTEBOOK_NAME"] = "/Users/sondrerogde/Dev/LSTM-for-option-pricing"
    wandb.login(key="6243a3eda53307963c7461f96bd7d2e9bad9a8b3")

    #Variables
    first_year = 2019
    last_year = 2021
    split_date ="2021-01-01"

    epochs = 300

    if model_type in ["MLP", "LSTM"]:
        features = ["Underlying_last", "Strike", "Ttl", "Volatility", "R"]
    elif model_type == "MLP-GARCH":
        features = ["Underlying_last", "Strike", "Ttl", "Volatility_GJR_GARCH", "R"]

    seq_length = 5
    num_features = len(features)
    model_config = {
        "seq_length": seq_length,
        "num_features": num_features,
        }

    file = f"../data/processed_data/{first_year}-{last_year}_underlying-strike_only-price.csv"
    df_read = read_file(file)

  
    seq_length = 5
    num_features = len(features)
    num_outputs = 1

    df_read_lags = lag_features(df_read, features, seq_length)

    train_val_test = []


    train_start = datetime(2020, 4, 1)
    val_start = train_start + relativedelta(months=8)
    test_start = val_start + relativedelta(months=1)
    test_end = test_start + relativedelta(months=1)

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

    train_val_test.append(((train_x_scaled, train_y_org), (val_x_scaled, val_y_org), (test_x_scaled, test_y_org)))


    def trainer(train_x = train_x_scaled, train_y = train_y_org, val_x = val_x_scaled, val_y = val_y_org, model_config = model_config, model_type = model_type):
    # Initialize a new wandb run
        with wandb.init(config=sweep_configuration):

            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config
            config["seq_length"] = model_config["seq_length"]
            config["num_features"] = model_config["num_features"]

            if model_type in ["MLP", "MLP-GARCH"]:
                model = create_model_mlp(config)
            elif model_type == "LSTM":
                model = create_model_lstm(config)

            early_stopping = EarlyStopping(
                monitor='val_loss',
                mode='min',
                min_delta=1e-6,
                patience=5,
            )
            
            wandb_callback = WandbMetricsLogger()

            # I removed these
            # shuffle = np.random.permutation(len(train_x))
            # train_x, train_y = train_x[shuffle], train_y[shuffle]

            model.fit(
                train_x,
                train_y,
                batch_size = config.minibatch_size,
                validation_data = (val_x, val_y),
                epochs = epochs,
                callbacks = [early_stopping, wandb_callback] 
            )
            if model_type in ["MLP", "MLP-GARCH"]:
                wandb.log({"validation_loss": model.evaluate(val_x, val_y)})
            elif model_type == "LSTM":
                wandb.log({"validation_loss": model.evaluate(val_x, val_y)[0]})
        
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="LSTM_OPTION_PRICING")
    wandb.agent(sweep_id=sweep_id, function=trainer, project="LSTM_OPTION_PRICING", count = 100)

if __name__ == "__main__":
    model_type = "MLP-GARCH"
    sweep_configuration = {
        'method': 'bayes',
        'name': 'MLP-GARCH',
        'metric': {
            'goal': 'minimize', 
            'name': 'validation_loss'
            },

        'parameters': {
            "units": {'values': [32, 64, 96, 128]},
            "learning_rate": {
                "distribution": "uniform",
                'max': 0.005, 'min': 0.0005},
            "layers": {'values': [4, 5, 6]},
            "minibatch_size": {'values': [1024, 2048, 4096]},
            "bn_momentum": {
                "distribution": "uniform",
                "max": 0.40,
                "min": 0.00
            },
            "weight_decay": {
                "distribution": "uniform",
                "max": 0.0005,
                "min": 0.00
            }
        }
    }
    

    hyperparam_sweep(sweep_configuration, model_type)



