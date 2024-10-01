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

Train the different models in the src/models folder

Step 5: Report results 

The models are evaluated and compared in notebooks/compare.ipynb and the data is summarized in notebooks/tables.ipynb
