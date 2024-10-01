import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_files(path, filenames):
    """Reads all files and returns a dataframe"""
    return pd.concat((pd.read_csv(path + f, skipinitialspace=True) for f in filenames))

def read_file(file):
    """Read a single file and return a dataframe"""
    return pd.read_csv(file, skipinitialspace=True)

def create_train_test(df, split_date):
    """Splits data in training and test set, and transforms data to right 2D format"""
    return df[df["Quote_date"] < split_date], df[df["Quote_date"] >= split_date]

def df_to_xy(df, num_features, seq_length, num_outputs):
    """Transforms a dataframe into two arrays of explanatory variables x and explained variables y"""
    array = df.to_numpy()
    array_x, array_y = array[:, -num_features*seq_length - num_outputs:-num_outputs].astype(np.float32), array[:,-num_outputs:].astype(np.float32)
    return array_x, array_y

def min_max_scale(train, test):
    """Scales a training and test set using MinMaxScaler. The scaler is calibrated on the training set"""
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test