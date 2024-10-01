# OptionPricingLSTM
This repo is used to conduct research on option pricing with deep learning models.

How to run the code:

Step 1: Download the data

Download data from OptionDX for the options data and from FRED for the risk free rate.
- https://www.optionsdx.com/product/spx-option-chain/
- https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202409

Step 2: Preprocess the data

Create features and preprocess the data by running src/features/make_dataset.py from root.

Step 3: Hyperparameter search

Do a hyperparameter search by running src/models/hyperparam_sweep.py from root with the right configuration.

Step 4: Training models

Train the different models with the src/models/main.py file. Make sure to change the config as specified.

Step 5: Adding benchmarks

After training the models. Run src/models/naive_and_BS.py to add naive and black-scholes benchmark. This has to be done after training the MLP.

Step 6: Report results 

The models are evaluated and compared in notebooks/compare.ipynb and the data is summarized in notebooks/tables.ipynb


# Other

- To get a key to perform the hyperparameter sweep you have to create a user at https://wandb.ai/site.
- The notebooks contains much of the same code as the scripts in the src-folder.
- While all scripts should be run from root, run the notebooks from the folder they are in.
- The notebooks does not contain any extra functionality than what is in the src-folder, so they do not have to be run.