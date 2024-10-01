from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.activations import tanh, relu
from tensorflow.keras.optimizers import AdamW

def create_model(config):
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