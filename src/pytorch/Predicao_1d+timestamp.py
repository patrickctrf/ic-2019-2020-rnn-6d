import glob
import os
import pickle
import shutil

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, ProgbarLogger
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from math import sqrt, sin, cos
import matplotlib.pyplot as plt
import numpy
from numpy import concatenate, array
import numpy as np
from tensorflow.python.keras.models import model_from_json
from tensorflow_core.python.keras.optimizer_v2.nadam import Nadam
from tqdm import tqdm


def fake_position(x):
    """
Generate a simulated one dimensional position for training on "1d dataset" an
test if the neural network is able to generalize well.

Given "x" values, calculate f(x) for this.

It is supposed to be the output for the neural network.

    :param x: Array of values to calculate f(x) = y.
    :return: Array os respective position in "y" axis.
    """

    if x == 0:
        return 1

    return sin(x) / x


def fake_acceleration(x):
    """
Generate a simulated one dimensional acceleration for training on "1d dataset" an test if the neural network is able to generalize well.

Given "x" values, calculate f''(x) for this.

It is supposed to be the input for the neural network.

    :param x: Array of values to calculate f''(x) = y''.
    :return: Array os respective acceleration in "y" axis.
    """

    if x == 0:
        return -1 / 3

    return -((x ** 2 - 2) * sin(x) + 2 * x * cos(x)) / x ** 3


# create a differenced series

def difference(dataset, interval=1):
    """
For a given dataset, calculates the difference between each sample and the
previous one, for both input and output values.

    :param dataset: Dataset to difference from.
    :param interval: Which sample to compare (usually 1).
    :return: New dataset composed of differences.
    """
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


def tensorboard_and_callbacks(batch_size, log_dir="./logs", model_checkpoint_file="best_weights.{val_loss:.4f}-{epoch:05d}.hdf5", csv_file_path="loss_log.csv"):
    """
Utility function to generate tensorboard and others callback, deal with directory needs and keep the code clean.

    :param batch_size: batch size in training data (needed for compatibility).
    :param log_dir: Where to save the logs files.
    :param model_checkpoint_file: File to save weights that resulted best (smallest) validation loss.
    :param csv_file_path: CSV file path to save loss and validation loss values along the training process.
    :return: tesnorboard_callback for keras callbacks.
    """
    # We need to exclude previous tensorboard and callbacks logs, or it is gone
    # produce errors when trying to visualize it.
    try:
        shutil.rmtree(log_dir)
    except OSError as e:
        print("Aviso: %s : %s" % (log_dir, e.strerror))

    try:
        # Get a list of all the file paths with first 8 letters from model_checkpoint_file.
        file_list = glob.glob(model_checkpoint_file[:8] + "*")

        # Iterate over the list of filepaths & remove each file.
        for file_path in file_list:
            os.remove(file_path)
    except OSError as e:
        print("Error: %s : %s" % (model_checkpoint_file, e.strerror))

    # try:
    #     os.remove(csv_file_path)
    # except OSError as e:
    #     print("Error: %s : %s" % (csv_file_path, e.strerror))

    tensorboard_callback = TensorBoard(log_dir=log_dir,
                                       histogram_freq=1,
                                       batch_size=batch_size,
                                       write_grads=True,
                                       write_graph=True,
                                       write_images=True,
                                       update_freq="epoch",
                                       embeddings_freq=0,
                                       embeddings_metadata=None)

    model_checkpoint_callback = ModelCheckpoint(filepath=model_checkpoint_file,
                                                save_weights_only=True,
                                                monitor='val_loss',
                                                mode='min',
                                                save_best_only=True)

    csv_logger_callback = CSVLogger(csv_file_path, separator=",", append=True)

    return [model_checkpoint_callback, csv_logger_callback]


def get_model_sequential(neurons, batch_size, X):
    model = Sequential()
    # A quantidade de amostras (ex: 266) nao eh levada em conta aqui, o que
    # importa eh o shape de entrada, por isso nao usa X.shape[0].
    # O batch_size indica quantas SAMPLES sao usadas para avaliar o gradiente
    # DE UMA VEZ. No caso, ajustamos a rede a cada 1 sample (onilne training),
    # que eh o tamanho do batch
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Nadam(lr=10 ** -5))

    return model


def get_model_functional(neurons, batch_size, X):
    input = Input(shape=(X.shape[1], X.shape[2]))
    # A quantidade de amostras (ex: 266) nao eh levada em conta aqui, o que
    # importa eh o shape de entrada, por isso nao usa X.shape[0].
    # O batch_size indica quantas SAMPLES sao usadas para avaliar o gradiente
    # DE UMA VEZ. No caso, ajustamos a rede a cada 1 sample (onilne training),
    # que eh o tamanho do batch
    output = LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True)(input)
    output = Dense(1)(output)

    model = Model(inputs=[input], outputs=[output])

    model.compile(loss='mean_squared_error', optimizer='nadam')

    return model


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons, time_steps=1, model_file_name="model", validation_data=None, previous_model=None):
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
    :param previous_model: If we are going to resume a previous training process, the trained model is passed here. If None, new model is generated.
    :return: Your trained model.
    """
    X, y = train[:, 0:-1], train[:, -1]

    X_validation, y_validation = validation_data[:, 0:-1], validation_data[:, -1]

    # (Samples, Time STEPS, Fetures)
    # Ex: 266 amostras com 1 time step (NAO tem a ver com online training) e 1 feature (1 entrada)
    X = X.reshape(X.shape[0], time_steps, X.shape[1])

    X_validation = X_validation.reshape(X_validation.shape[0], time_steps, X_validation.shape[1])

    # ========RESUME-OLD-TRAINING===============================================
    # Este IF serve apenas para resgatar um modelo que ja tenha sido
    # parcialmente treinado anteriormente. Util para o google colab, por exemplo.
    if previous_model is None:
        model = get_model_sequential(neurons=neurons, batch_size=batch_size, X=X)
    else:
        model = get_model_sequential(neurons=neurons, batch_size=batch_size, X=X)
        model.compile(loss='mean_squared_error', optimizer=Nadam(lr=10 ** -5))
        # Atualiza os pesos do modelo que faz predicao online.
        old_weights = previous_model.get_weights()
        model.set_weights(old_weights)
    # ========END-RESUME-OLD-TRAINING===========================================

    # =================CHANGE-BATCH-INPUT-SIZE==================================
    # Very important
    # Our new model must be able to do 1 sample PREDICTIONS. We would need
    # "batch_size" predictions to avoid an error, and we want 1 sample
    # predictions. So, we load the weights into a new model with input shape
    # of 1. Solution 3: Copy Weights do site abaixo:
    # https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
    n_batch = 1
    # re-define model
    new_model = get_model_sequential(neurons=neurons, batch_size=n_batch, X=X)
    # copy weights
    old_weights = model.get_weights()
    new_model.set_weights(old_weights)
    # =================END-CHANGE-INPUT-BATCH-SIZE==============================

    # Saving model structure to file
    # serialize model to JSON
    model_json = new_model.to_json()
    with open(model_file_name + ".json", "w") as json_file:
        json_file.write(model_json)

    keras_callbacks = tensorboard_and_callbacks(batch_size, log_dir="./logs")

    for i in tqdm(range(nb_epoch)):
        # I believe we need to train each epoch and then reset states, beacause
        # this is a stateful lstm, and it would "remember" previous inputs if we
        # didn't reset it.
        # Lembra que ele faz RESET depois de passar pelo dataset "X" INTEIRO,
        # nao a cada sample
        training_history = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, validation_data=(X_validation, y_validation), callbacks=keras_callbacks)
        model.reset_states()

    # serialize (and save) WEIGHTS to HDF5 (equals to ".h5" file format)
    model.save_weights(model_file_name + ".hdf5")

    # Atualiza os pesos do modelo que faz predicao online.
    old_weights = model.get_weights()
    new_model.set_weights(old_weights)

    return new_model, training_history


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    """
Performs a one-step prediction for a given LSTM model.

    :param model: Trained model that forecasts the data.
    :param batch_size: Batch size used (usually 1).
    :param X: Input data (single sample).
    :return: Single step prediction.
    """
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def load_model(file_name="model", best_in_ranking=0):
    """
Retrives a Keras model from a file.

    :param file_name: Prefix for JSON (model) and h5 (weights) files.
    :param best_in_ranking: From all best validation losses saved, which one to retrive (Zero means smaller one).
    :return: Model obeject from Keras.
    """
    # load json and create model
    json_file = open(file_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Get a list of all the file paths with first 8 letters from model_checkpoint_file.
    file_list = glob.glob("best_wei*")

    # load weights into new model
    loaded_model.load_weights(file_list[best_in_ranking])
    print("Loaded model from disk")

    return loaded_model


def load_training(file_name="model", best_in_ranking=0):
    """
Retrives a Keras model from a file.

    :param file_name: Prefix for JSON (model) and h5 (weights) files.
    :param best_in_ranking: From all best validation losses saved, which one to retrive (Zero means smaller one).
    :return: Model obeject from Keras.
    """
    # load json and create model
    json_file = open(file_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(file_name + '.hdf5')
    print("Loaded model from disk")

    return loaded_model


def save_training_history(history, training_history_file_path="training_history_serialized"):
    """
Serialize a history object (generated with Keras model.fit()) and saves it into
a binary file for later retrieval.

Warning: Overwrites previous history if the same name is given.

    :param history: Training history object generated with Keras model.fit().
    :param training_history_file_path: File where history object is going to be serialized.
    """
    with open(training_history_file_path, "wb") as history_file:
        training_history = pickle.dump(history, history_file)

    # summarize history for loss and save it to file
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("val_loss.png", dpi=800)
    # plt.show()

    return


def load_training_history(training_history_file_path="training_history_serialized"):
    """
Retrieves a file containing training history, unserialize it and returns the
history object (generated with model.fit()).

    :param training_history_file_path: File where history object was serialized.
    :return: Original history object (genrated with model.fit())
    """
    with open(training_history_file_path, "rb") as history_file:
        training_history = pickle.load(history_file)

    return training_history


# run a repeated experiment
def experiment(repeats):
    """
Runs the experiment itself.

    :param repeats: Number of times to repeat the experiment. When we are trying to create a good network, it is reccomended to use 1.
    :return: Error scores for each repeat.
    """
    # transform data to be stationary
    raw_pos = [fake_position(i / 30) for i in range(-300, 300)]
    diff_pos = difference(raw_pos, 1)
    diff_pos = numpy.array(raw_pos)
    raw_accel = [fake_acceleration(i / 30) for i in range(-300, 300)]
    diff_accel = difference(raw_accel, 1)
    diff_accel = numpy.array(raw_accel)

    raw_timestamp = range(len(raw_pos))
    # diff_timestamp = difference(numpy.array(raw_timestamp) + numpy.random.rand(len(raw_pos)), 1)
    diff_timestamp = array(raw_timestamp)

    supervised_values = numpy.transpose(numpy.vstack((diff_timestamp, diff_accel, diff_pos)))
    # split data into train and test-sets
    train, test = supervised_values[0:int(len(diff_pos) * 2 / 3), :], supervised_values[int(-len(diff_pos) * 1 / 3):, :]

    # Zero first position
    train[:, 2] = train[:, 2] - train[0, 2]
    test[:, 2] = test[:, 2] - test[0, 2]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # run experiment
    error_scores = list()
    for r in range(repeats):
        # Resume old training
        lstm_model = load_training(file_name="/content/outputs_dissertacao/model")
        # fit the base model
        lstm_model, training_history = fit_lstm(train=train_scaled, batch_size=100, nb_epoch=1, neurons=3, validation_data=test_scaled, previous_model=lstm_model);
        save_training_history(history=training_history, training_history_file_path="training_history_serialized")
        # lstm_model = load_model(best_in_ranking=0)
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
        plt.close()
        plt.plot(range(len(predictions)), predictions, range(len(raw_pos[:len(train_scaled)])), raw_pos[:len(train_scaled)])
        plt.savefig("output_train.png", dpi=800)
        plt.show()
        rmse = mean_squared_error(raw_pos[:len(train_scaled)], predictions)
        print('%d) Test MSE: %.6f' % (r + 1, rmse))
        error_scores.append(rmse)

        lstm_model.reset_states()

        predictions = list()
        for i in range(len(test_scaled)):
            # predict
            # X, y = (np.random.rand(133, 3) * 2 - 1)[i, 0:-1], test_scaled[i, -1]
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            # yhat = inverse_difference(np.hstack((raw_pos[-(len(test_scaled) + 1)], array(predictions))), yhat,  1)
            # store forecast
            predictions.append(yhat)
        # report performance
        plt.close()
        plt.plot(range(len(predictions)), predictions, range(len(raw_pos[-len(test_scaled):])), raw_pos[-len(test_scaled):])
        plt.savefig("output_test.png", dpi=800)
        plt.show()
        rmse = mean_squared_error(raw_pos[-len(test_scaled):], predictions)
        print('%d) Test MSE: %.6f' % (r + 1, rmse))
        error_scores.append(rmse)

    return error_scores


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        # entry point
        experiment(1)
