from create_models import create_model_lstm, create_model_mlp
from secret import wandb_key
from model_utils import make_train_val_test
import wandb
from wandb.integration.keras import WandbMetricsLogger
from keras.callbacks import EarlyStopping


def hyperparam_sweep(sweep_configuration, model_type):
    # os.environ["WANDB_NOTEBOOK_NAME"] = "/Users/sondrerogde/Dev/LSTM-for-option-pricing"
    wandb.login(key=wandb_key)
    epochs = 300

    model_config = {
        "seq_length": 5,
        "num_features": 5,
    }

    train_x_scaled, train_y_org, val_x_scaled, val_y_org = make_train_val_test(model_type, True)

    def trainer(train_x = train_x_scaled, train_y = train_y_org, val_x = val_x_scaled, val_y = val_y_org, model_config = model_config, model_type = model_type):
        with wandb.init(config=sweep_configuration):

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
    # Change model_type to tune a different model
    model_type = "MLP-GARCH"
    sweep_configuration = {
        'method': 'bayes',
        'name': model_type,
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



