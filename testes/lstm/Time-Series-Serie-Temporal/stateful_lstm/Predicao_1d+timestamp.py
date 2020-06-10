from keras.callbacks import TensorBoard
from keras.engine.saving import model_from_json
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib.pyplot as plt
import numpy
from numpy import concatenate, array
import numpy as np
from tqdm import tqdm


def fake_position(x):
    return 1 * x ** 5 + 4 * x ** 4 + 3 * x ** 3 + 14 * x ** 2 + 2 * x - 3


def fake_acceleration(x):
    return 20 * x ** 3 + 48 * x ** 2 + 18 * x + 28


# # date-time parsing function for loading the dataset
# def parser(x):
#     return datetime.strptime('190' + x, '%Y-%m')
#
#
# # frame a sequence as a supervised learning problem
# def timeseries_to_supervised(data, lag=1):
#     df = DataFrame(data)
#     columns = [df.shift(i) for i in range(1, lag + 1)]
#     columns.append(df)
#     df = concat(columns, axis=1)
#     return df


# create a differenced series

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    """
If you are working with an output composed by differences of this time in
relation to the previous time, this function is useful beacause it takes the
array with the history and sums the given difference (yhat) to the last value in
history (or the value "-interval" in history).

It's more reasonable to use interval as 1, but I implemented it this way because
it gets more general.

    :param history: Array of historical values in the output series.
    :param yhat: A difference to be summed with the previous value.
    :param interval: Which past value is this difference (yhat) related to (usually, interval = 1).
    :return: An output representing the current value of the measuring being tracked, no more the difference of this value.
    """
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    """
Simples rescale train and test data (separately) and returns scaled data. Note
that the scaling process is done for EACH FEATURE. Even if you have different
range for each feature, the scaling procces is done separately.
This scaler SAVES data_min, data_max and feature range for later unscaling, or
even rescaling newer data.
Equation details: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

Also returns the scaler object used to further inverse scaling.

    :param train: Train data to be rescaled (array or matrix).
    :param test: Test data to be rescaled (array or matrix).
    :return: Scaler, train scaled and test scaled.
    """
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    """
Since you "scale" object saves data_min and data_max, as well as feature_range,
you just need to pass it to this function, togheter with the object you want to
inverse scale (yhat) too.

    :param scaler: Scaler object from scipy, previously calculated in the same dataset being worked here.
    :param X: Matrix line used as input to generate this output prediction (yhat).
    :param yhat: Predicted output for your network.
    :return: Only the "yhat" data rescaled.
    """
    # We have input features in X information. We need to add "y" info
    # (prediction, not ground truth or reference) before unscaling (because scaler was made with whole dataset, X and y).
    # We call it "row" because it's a array (there's no columns). We'll reshape
    # it into a matrix line ahead in this function.
    new_row = [x for x in X] + [yhat]
    array = numpy.array(new_row)
    # Array is vertical, but scaler expects a matrix or matrix line, and that's
    # why we reshape it.
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons, time_steps=1, model_file_name="model", validation_data=None):
    """
This function assembles your network and train it with provided data (train).
It returns your fitted model.

    :param train: Training data in shape (samples x features)
    :param batch_size: Size of each batch. For ONLINE training, batch == 1.
    :param nb_epoch: How many epochs to train.
    :param neurons: How many neurons in the LSTM cell.
    :param time_steps: For a stateful LSTM, usually we have 1 time step and it's hard to think in an alternative case, so we leave it optional.
    :param model_file_name: File name to save model after training.
    :param validation_data: Same pattern that training data, but the portion of validation to check overfitting, etc.
    :return: Your trained model.
    """
    X, y = train[:, 0:-1], train[:, -1]

    X_validation, y_validation = validation_data[:, 0:-1], validation_data[:, -1]

    # (Samples, Time STEPS, Fetures)
    # Ex: 266 amostras com 1 time step (NAO tem a ver com online training) e 1 feature (1 entrada)
    X = X.reshape(X.shape[0], time_steps, X.shape[1])

    X_validation = X_validation.reshape(X_validation.shape[0], time_steps, X_validation.shape[1])
    model = Sequential()
    # A quantidade de amostras (ex: 266) nao eh levada em conta aqui, o que
    # importa eh o shape de entrada, por isso nao usa X.shape[0].
    # O batch_size indica quantas SAMPLES sao usadas para avaliar o gradiente
    # DE UMA VEZ. No caso, ajustamos a rede a cada 1 sample (onilne training),
    # que eh o tamanho do batch
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='nadam')

    keras_callback = TensorBoard(log_dir="./logs",
                                 histogram_freq=1,
                                 batch_size=batch_size,
                                 write_grads=True,
                                 write_graph=True,
                                 write_images=True,
                                 update_freq="epoch",
                                 embeddings_freq=0,
                                 embeddings_metadata=None)

    for i in tqdm(range(nb_epoch)):
        # I believe we need to train each epoch and then reset states, beacause
        # this is a stateful lstm, and it would "remember" previous inputs if we
        # didn't reset it.
        training_history = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, validation_data=(X_validation, y_validation), callbacks=[keras_callback])
        model.reset_states()

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_file_name + ".h5")
    return model, training_history


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def load_model(file_name="model"):
    # load json and create model
    json_file = open(file_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file_name + '.h5')
    print("Loaded model from disk")

    return loaded_model


# run a repeated experiment
def experiment(repeats):
    # transform data to be stationary
    raw_pos = [fake_position(i / 100) for i in range(-100, 300)]
    diff_pos = difference(raw_pos, 1)
    diff_pos = numpy.array(raw_pos)
    raw_accel = [fake_acceleration(i / 100) for i in range(-100, 300)]
    diff_accel = difference(raw_accel, 1)
    diff_accel = numpy.array(raw_accel)

    raw_timestamp = range(len(raw_pos))
    # diff_timestamp = difference(numpy.array(raw_timestamp) + numpy.random.rand(len(raw_pos)), 1)
    diff_timestamp = array(raw_timestamp)

    supervised_values = numpy.transpose(numpy.vstack((diff_timestamp, diff_accel, diff_pos)))
    # split data into train and test-sets
    train, test = supervised_values[0:int(len(diff_pos) * 2 / 3), :], supervised_values[int(-len(diff_pos) * 1 / 3):, :]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # run experiment
    error_scores = list()
    for r in range(repeats):
        # fit the base model
        # lstm_model, training_history = fit_lstm(train=train_scaled, batch_size=1, nb_epoch=1000, neurons=10, validation_data=test_scaled)
        lstm_model = load_model()
        # forecast test dataset
        predictions = list()
        for i in range(len(train_scaled)):
            # predict
            # X, y = (np.random.rand(133, 3) * 2 - 1)[i, 0:-1], test_scaled[i, -1]
            X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            # yhat = inverse_difference(np.hstack((raw_pos[-(len(test_scaled) + 1)], array(predictions))), yhat,  1)
            # store forecast
            predictions.append(yhat)
        # report performance
        plt.plot(range(len(predictions)), predictions, range(len(raw_pos[:len(train_scaled)])), raw_pos[:len(train_scaled)])
        plt.show()
        rmse = sqrt(mean_squared_error(raw_pos[:len(train_scaled)], predictions))
        print('%d) Test RMSE: %.3f' % (r + 1, rmse))
        error_scores.append(rmse)
    return error_scores


# entry point
experiment(1)
