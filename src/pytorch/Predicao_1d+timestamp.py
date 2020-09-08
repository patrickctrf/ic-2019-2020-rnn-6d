import glob
import os
import pickle
import shutil

import tensorflow as tf
import torch
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
from numpy import concatenate, array, absolute, arange
import numpy as np
from tensorflow.python.keras.models import model_from_json
from tensorflow_core.python.keras.optimizer_v2.nadam import Nadam
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from tqdm import tqdm
from ptk.timeseries import *
from ptk.utils import *


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

    try:
        os.remove(csv_file_path)
    except OSError as e:
        print("Error: %s : %s" % (csv_file_path, e.strerror))

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


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, training_batch_size=64, epochs=150):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.training_batch_size = training_batch_size
        self.epochs = epochs

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        # We train using multiple inputs (mini_batch), so we let this cell ready
        # to be called.
        self.hidden_cell_training = (torch.zeros(1, self.training_batch_size, self.hidden_layer_size),
                                     torch.zeros(1, self.training_batch_size, self.hidden_layer_size))

        # We predict using single input per time, so we let single batch cell
        # ready here.
        self.hidden_cell_prediction = (torch.zeros(1, 1, self.hidden_layer_size),
                                       torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        # (seq_len, batch, input_size), mas pode inverter o
        # batch com o seq_len se fizer batch_first==1 na criacao do LSTM
        lstm_out, self.hidden_cell_training = self.lstm(input_seq, self.hidden_cell_training)

        inverse = pad_packed_sequence(lstm_out, batch_first=True)
        
        lstm_out = inverse[0][arange(inverse[0].shape[0]), inverse[1] - 1]

        # O "-1" aqui eh para a quantidade de FEATURES saindo do LSTM. Batch
        # size acho que eh interpretado sozinho, tenho que testar.
        predictions = self.linear(lstm_out)

        return predictions

    def fit(self, X, y):
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        epochs = self.epochs

        # Invertendo para a sequencia ser DEcrescente.
        X = X[::-1]
        y = y[::-1]

        # y numpy array values into torch tensors
        y = torch.from_numpy(y).float()
        # split into mini batches
        y_batches = torch.split(y, split_size_or_sections=self.training_batch_size)

        # Para criar
        lista = [torch.from_numpy(i).view(-1, self.output_size).float() for i in X]
        X = pack_sequence(lista)

        for i in range(epochs):
            optimizer.zero_grad()
            # self.hidden_cell_training = (torch.zeros(1, 1, self.hidden_layer_size),
            #                              torch.zeros(1, 1, self.hidden_layer_size))

            y_pred = self(X)

            single_loss = loss_function(y_pred, y)
            single_loss.backward()
            optimizer.step()

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    # def get_params(self, **kwargs):
    #
    # def predict(self, X):
    #
    # def score(self, X, y, **kwargs):
    #
    # def set_params(self, **params):


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
    # diff_accel = numpy.array(raw_accel)

    raw_timestamp = range(len(raw_pos))
    # diff_timestamp = difference(numpy.array(raw_timestamp) + numpy.random.rand(len(raw_pos)), 1)
    diff_timestamp = array(raw_timestamp)

    model = LSTM(input_size=1, hidden_layer_size=100,
                 output_size=1, training_batch_size=10)

    X, y = timeseries_dataloader(data_x=raw_accel, data_y=diff_pos, enable_asymetrical=True)

    model.fit(X, y)

    # report performance
    # plt.close()
    # plt.plot(range(len(predictions)), predictions, range(len(raw_pos[:len(train_scaled)])), raw_pos[:len(train_scaled)])
    # plt.savefig("output_train.png", dpi=800)
    # plt.show()
    # rmse = mean_squared_error(raw_pos[:len(train_scaled)], predictions)

    error_scores = []

    return error_scores


if __name__ == '__main__':
    experiment(1)

# tscv = TimeSeriesSplit(n_splits=len(raw_accel) - 1)
#
# # train_index, test_index = tscv.split(array(raw_accel))
#
# # [train_index, test_index for train_index, test_index in tscv.split(array(raw_accel))]
#
# iterator_into_list = list(tscv.split(array(raw_accel)))
# iterator_into_array = array(iterator_into_list)
# train_index, test_index = iterator_into_array[:, :-1], iterator_into_array[:, 1]
#
# X, y = array(raw_accel)[train_index], array(diff_pos)[test_index]
