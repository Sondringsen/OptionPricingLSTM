import pandas as pd
import numpy as np
from pathlib import Path

from preprocessing_utils import process_rates, process_options
from data_utils import read_files


def combine_opt_rates(df_opt, df_r):
    """Combines dataframes for options and rates matching the Ttl of the option to the closest R"""
    df_opt = pd.merge(df_opt, df_r, on ="Quote_date", how = "left")
    rates = list(df_r.columns)
    rates.remove("Quote_date")
    df_opt["Ttl_diff"] = df_opt["Ttl"].apply(lambda x: (np.abs(np.array(rates) - x)).argmin())
    df_opt["R"] = df_opt[["Ttl_diff"] + rates].values.tolist()
    df_opt["R"] = df_opt["R"].apply(lambda x: x[int(x[0]+1)])
    df_opt = df_opt.drop(rates + ["Ttl_diff"], axis=1)
    return df_opt.dropna()

def get_model_dataset(path_opt, filenames_opt, path_r, filenames_r, call = True):
    """Wrapper function to extract option data and rates. Returns a combined dataframe"""
    df_opt = read_files(path_opt, filenames_opt)
    df_r = read_files(path_r, filenames_r)
    df_opt = process_options(df_opt, call)
    df_r = process_rates(df_r)
    df = combine_opt_rates(df_opt, df_r)
    return df


def create_csv(first_year, last_year):
    path_opt = "data/raw/"
    filenames_opt = ["spx_eod_" + str(year) + (str(month) if month >= 10 else "0"+str(month)) +".txt" for year in range(first_year-1, last_year+1) for month in range(1, 13)]
    path_r = "data/raw/"

    filenames_r = ["yield-curve-rates-1990-2023.csv"]
    
    call = True
    df = get_model_dataset(path_opt, filenames_opt, path_r, filenames_r, call)
    print("Data read")

    df = df[df["Quote_date"] >= f"{str(first_year)}-04-01"]
    df = df[df["Quote_date"] <= f"{str(last_year)}-12-31"]

    filename = f"data/processed/{first_year}-{last_year}_underlying-strike_only-price.csv"

    filepath = Path(filename)  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filename)
    print("Data written")


if __name__ == "__main__":
    first_year = 2019
    last_year = 2021
    create_csv(first_year, last_year)