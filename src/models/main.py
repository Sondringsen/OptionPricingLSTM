from create_models import create_model_lstm, create_model_mlp
import numpy as np
from keras.callbacks import EarlyStopping
import tensorflow as tf
from model_utils import make_train_val_test, save_predictions

def main(config, model_type):

    def trainer(train_x, train_y, model, val_x, val_y):
        epochs = 100
        minibatch_size = config["minibatch_size"]

        tf.random.set_seed(42)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            min_delta = 0,
            patience = 1,
        )

        model.fit(
            train_x,
            train_y,
            batch_size = minibatch_size,
            validation_data = (val_x, val_y),
            epochs = epochs,
            callbacks = [early_stopping]
        )

    train_val_test = make_train_val_test(model_type)
    predictions = []
    for i, ((x_train, y_train), (x_val, y_val), (x_test, y_test)) in enumerate(train_val_test):
        if model_type == "LSTM":
            model = create_model_lstm(config)
        elif model_type in ["MLP", "MLP-GARCH"]:
            model = create_model_mlp(config)
        model.summary()
        trainer(x_train, y_train, model, x_val, y_val)
        predictions.append(np.array(model(x_test)))

    predictions = np.concatenate(predictions)

    save_predictions(predictions, model_type)



if __name__ == "__main__":
    model_type = "MLP"
    config = {
        "units": 64,
        "learning_rate": 0.002594627161103502,
        "layers": 5,
        "bn_momentum" : 0.26212094315874734,
        "weight_decay": 0.0003327609151101109,
        "minibatch_size": 4096,
        "seq_length": 5,
        "num_features": 5,
    }
    main(config, model_type)