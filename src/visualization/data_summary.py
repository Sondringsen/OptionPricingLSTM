import pandas as pd

def read_file(file):
    """Read a single file and return a dataframe"""
    return pd.read_csv(file, skipinitialspace=True)

def create_data_summary_table():
    df_read = read_file("data/processed/2019-2021_underlying-strike_only-price.csv")
    df_read = df_read.loc[df_read.loc[:, "Quote_date"] >= "2020-04-01", :]

    df_read["Moneyness"] = "0.97-1.03"
    df_read.loc[df_read.loc[:, "Underlying_last"]/df_read.loc[:, "Strike"] < 0.97, "Moneyness"] = "<0.97"
    df_read.loc[df_read.loc[:, "Underlying_last"]/df_read.loc[:, "Strike"] > 1.03, "Moneyness"] = ">1.03"

    df_group_moneyness = df_read.groupby(by="Moneyness")

    level_1 = ["Price", "Underlying_last", "Ttl", "Volatility", "R"] 
    level_2 = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]  
    index = pd.MultiIndex.from_product([level_1, level_2], names=["Feature", "Statistic"])

    level_3 = ["<0.97", "0.97-1.03", ">1.03"]
    columns = pd.Index(level_3, name="Moneyness")

    df_summary = pd.DataFrame(index=index, columns=columns)

    for moneyness_level, group_data in df_group_moneyness:
        description = group_data.describe()

        for feature in level_1:
            for stat in level_2:
                try:
                    df_summary.loc[(feature, stat), str(moneyness_level)] = description[feature][stat]
                except KeyError:
                    print(f"Feature {feature} or Statistic {stat} not found in description for moneyness {moneyness_level}")

    latex_table = df_summary.to_latex(multicolumn=True, multicolumn_format='c', float_format="%.2f")

    with open('reports/data_summary.tex', 'w') as f:
        f.write(latex_table)

def create_config_tables():
    config_lstm = {
        "Layers": 5,
        "Units per layer": 64,
        "Learning rate": 0.002594627161103502,
        "Weight decay": 0.0003327609151101109,
        "BN momentum" : 0.26212094315874734,
        "Minibatch size": 4096,
        "Epochs": "Early stopping",
        "Optimizer": "AdamW",
    }

    config_mlp = {
        "Layers": 4,
        "Units per layer": 32,
        "Learning rate": 0.004469423596275494,
        "Weight decay" : 0.00042470893538329376,
        "BN momentum" : 0.30057069329591907,
        "Minibatch size": 4096,
        "Epochs": "Early stopping",
        "Activation function": "ReLU",
        "Optimizer": "AdamW",
    }

    config_mlp_garch = {
        "Layers": 6,
        "Units per layer": 96,
        "Learning rate": 0.004102449498283615,
        "Weight decay" : 0.0002017422068564576,
        "BN momentum" : 0.32753376728017486,
        "Minibatch size": 2048,
        "Epochs": "Early stopping",
        "Activation function": "ReLU",
        "Optimizer": "AdamW",
    }

    lstm_hyper = pd.DataFrame.from_dict(config_lstm, orient="index")
    mlp_hyper = pd.DataFrame.from_dict(config_mlp, orient="index")
    mpl_garch_hyper = pd.DataFrame.from_dict(config_mlp_garch, orient="index")

    with open('reports/LSTM_hyperparameters.tex', 'w') as f:
        f.write(lstm_hyper.to_latex())

    with open('reports/MLP_hyperparameters.tex', 'w') as f:
        f.write(mlp_hyper.to_latex())

    with open('reports/MLP_GARCH_hyperparameters.tex', 'w') as f:
        f.write(mpl_garch_hyper.to_latex())

def main():
    create_data_summary_table()
    create_config_tables()


if __name__ == "__main__":
    main()
    


