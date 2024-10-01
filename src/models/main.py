# from preprocessing import get_model_dataset, create_train_test, min_max_scale, df_to_xy, read_file, lag_features
from models.lstm import create_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import date
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from keras.callbacks import EarlyStopping
import tensorflow as tf

def read_file(file):
    """Read a single file and return a dataframe"""
    return pd.read_csv(file, skipinitialspace=True)

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

def df_to_xy(df, num_features, seq_length, num_outputs):
    """Transforms a dataframe into two arrays of explanatory variables x and explained variables y"""
    array = df.to_numpy()
    array_x, array_y = array[:, -num_features*seq_length - num_outputs:-num_outputs].astype(np.float32), array[:,-num_outputs:].astype(np.float32)
    return array_x, array_y


def main(config):
    first_year = 2019
    last_year = 2021
    file = f"../data/processed_data/{first_year}-{last_year}_underlying-strike_only-price.csv"

    df_read = read_file(file)

    num_models = 12

    features = ["Underlying_last", "Strike", "Ttl", "Volatility", "R"]
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

        train_x_org, train_y_org = df_to_xy(df_train_orginal, num_features, seq_length, num_outputs)
        val_x_org, val_y_org = df_to_xy(df_val_orginal, num_features, seq_length, num_outputs)
        test_x_org, test_y_org = df_to_xy(df_test_orginal, num_features, seq_length, num_outputs)

        scaler = MinMaxScaler()
        train_x_scaled = scaler.fit_transform(train_x_org)
        val_x_scaled = scaler.transform(val_x_org)
        test_x_scaled = scaler.transform(test_x_org)

        print(month, test_x_scaled.shape)
        print(test_start, test_end)

        """shuffle = np.random.permutation(len(train_x_scaled))
        train_x_scaled, train_y_scaled = train_x_scaled[shuffle], train_y_scaled[shuffle]"""

        train_x_scaled = np.reshape(train_x_scaled, (len(train_x_scaled), seq_length, num_features))
        val_x_scaled = np.reshape(val_x_scaled, (len(val_x_scaled), seq_length, num_features))
        test_x_scaled = np.reshape(test_x_scaled, (len(test_x_scaled), seq_length, num_features))

        # print(f"Train_x shape: {train_x_scaled.shape}, train_y shape: {train_y_org.shape}")
        # print(f"Test_x shape: {test_x_scaled.shape}, test_y shape: {test_y_org.shape}")
        # print("------------------------------------------------")
        train_val_test.append(((train_x_scaled, train_y_org), (val_x_scaled, val_y_org), (test_x_scaled, test_y_org)))
    

    def trainer(train_x, train_y, model, val_x, val_y):
        epochs = 100
        minibatch_size = 4096

        tf.random.set_seed(42)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            min_delta = 0,
            patience = 20,
            # patience = 3,
        )

        model.fit(
            train_x,
            train_y,
            batch_size = minibatch_size,
            # validation_split = 0.3,
            validation_data = (val_x, val_y),
            epochs = epochs,
            callbacks = [early_stopping]
        )

    predictions = []
    for i, ((x_train, y_train), (x_val, y_val), (x_test, y_test)) in enumerate(train_val_test):
        model = create_model(config)
        model.summary()
        trainer(x_train, y_train, model, x_val, y_val)
        predictions.append(np.array(model(x_test)))

    # predictions = np.array(predictions)
    predictions = np.concatenate(predictions)

    def prediction(df_test, predictions):
        # df_test["Prediction"] = predictions.flatten()
        df_test["Prediction"] = predictions
        return df_test
    df_test_whole = df_read_lags.loc[df_read_lags.loc[:, "Quote_date"] >= "2021-01-01", :]
    df_test_whole = prediction(df_test_whole, predictions)

    time = datetime.now()
    time = time.strftime("%m-%d_%H-%M")

    filename = f"../data/Predictions/{last_year}_predictions_{time}_LSTM.csv"
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok = True)
    df_test_whole.to_csv(filename)



if __name__ == "__main__":
    config = {
        "units": 64,
        "learning_rate": 0.002594627161103502,
        "layers": 5,
        "bn_momentum" : 0.26212094315874734,
        "weight_decay": 0.0003327609151101109,
        "seq_length": 5,
        "num_features": 5,
    }
    main(config)