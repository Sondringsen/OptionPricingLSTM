from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from keras.activations import relu


def create_model_lstm(config):
  """Builds an LSTM model of minimum 2 layers sequentially from a given config dictionary"""
  model = Sequential()

  model.add(LSTM(
    units = config["units"],
    activation = relu,
    input_shape = (config["seq_length"], config["num_features"]),
    return_sequences = True
  ))

  model.add(BatchNormalization(
    momentum = config["bn_momentum"]
  ))


  for i in range(config["layers"]-2):
    model.add(LSTM(
      units = config["units"],
      activation = relu,
      return_sequences = True
    ))
    model.add(BatchNormalization(
      momentum = config["bn_momentum"]
    ))

  model.add(LSTM(
    units = config["units"],
    activation = relu,
    return_sequences = False
  ))

  model.add(BatchNormalization(
    momentum = config["bn_momentum"]
  ))

  model.add(Dense(
    units = 1,
    activation = relu
  ))  

  model.compile(
    optimizer = AdamW(
      learning_rate = config["learning_rate"],
      weight_decay = config["weight_decay"]
    ),
    loss = "mse",
    metrics = ["mae"]
  )

  return model


def create_model_mlp(config):
  """Builds a model of minimum 2 layers sequentially from a given config dictionary"""
  model = Sequential()

  model.add(Dense(
    units = config["units"],
    activation = relu,
    input_shape = (config["num_features"],)
  ))

  model.add(BatchNormalization(
    momentum = config["bn_momentum"]
  ))


  for i in range(config["layers"]-2):
    model.add(Dense(
      units = config["units"],
      activation = relu
    ))
    model.add(BatchNormalization(
      momentum = config["bn_momentum"]
    ))

  model.add(Dense(
    units = config["units"],
    activation = relu
  ))

  model.add(BatchNormalization(
    momentum = config["bn_momentum"]
  ))

  model.add(Dense(
    units = 1,
    activation = relu
  ))  

  model.compile(
    optimizer = AdamW(
      learning_rate = config["learning_rate"],
      weight_decay = config["weight_decay"]
    ),
    loss = "mse",
  )

  return model