import csv
import glob
from math import sin, cos
import os
from time import sleep, time

import numpy
import torch
from matplotlib import pyplot as plt
from numpy import arange, random, vstack, transpose, asarray, absolute, diff, savetxt, save, savez, memmap, copyto, concatenate, ones, where, array, load, zeros
from pandas import Series, DataFrame, read_csv
from ptk.timeseries import *
from ptk.utils import *
from skimage.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skopt import BayesSearchCV
from torch import nn, stack
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm


# def fake_position(x):
#     """
# Generate a simulated one dimensional position for training on "1d dataset" an
# test if the neural network is able to generalize well.
#
# Given "x" values, calculate f(x) for this.
#
# It is supposed to be the output for the neural network.
#
#     :param x: Array of values to calculate f(x) = y.
#     :return: Array os respective position in "y" axis.
#     """
#
#     return cos(x) + 2 * sin(2 * x)
#
#
# def fake_acceleration(x):
#     """
# Generate a simulated one dimensional acceleration for training on "1d dataset" an test if the neural network is able to generalize well.
#
# Given "x" values, calculate f''(x) for this.
#
# It is supposed to be the input for the neural network.
#
#     :param x: Array of values to calculate f''(x) = y''.
#     :return: Array os respective acceleration in "y" axis.
#     """
#
#     return 4 * cos(2 * x) - sin(x)


# create a differenced series

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


def plot_csv(csv_path="dataset-room2_512_16/mav0/mocap0/data.csv"):
    output_data = read_csv(csv_path)

    for key in output_data.columns[1:]:
        plt.close()
        output_data.plot(kind='scatter', x=output_data.columns[0], y=key, color='red')
        plt.savefig(key + ".png", dpi=200)
        # plt.show()

    return


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
    return array(diff)


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


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, n_lstm_units=1, epochs=150, training_batch_size=64, validation_percent=0.2, bidirectional=False, device=torch.device("cpu")):
        """
This class implements the classical LSTM with 1 or more cells (stacked LSTM). It
receives sequences and returns the predcition at the end of each one.

There is a fit() method to train this model according to the parameters given in
the class initialization. It follows the sklearn header pattern.

This is also an sklearn-like estimator and may be used with any sklearn method
designed for classical estimators. But, when using GPU as PyTorch device, you
CAN'T use multiple sklearn workers (n_jobs), beacuse it raises an serializtion
error within CUDA.

        :param input_size: Input dimension size (how many features).
        :param hidden_layer_size: How many features there will be inside each LSTM.
        :param output_size: Output dimension size (how many features).
        :param n_lstm_units: How many stacked LSTM cells (or units).
        :param epochs: The number of epochs to train. The final model after
        train will be the one with best VALIDATION loss, not necessarily the
        model found after whole "epochs" number.
        :param training_batch_size: Size of each mini-batch during training
        process. If number os samples is not a multiple of
        "training_batch_size", the final batch will just be smaller than the
        others.
        :param validation_percent: The percentage of samples reserved for
        validation (cross validation) during training inside fit() method.
        :param bidirectional: If the LSTM units will be bidirectional.
        :param device: PyTorch device, such as torch.device("cpu") or
        torch.device("cuda:0").
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.training_batch_size = training_batch_size
        self.epochs = epochs
        self.validation_percent = validation_percent
        self.n_lstm_units = n_lstm_units
        if bidirectional:
            self.bidirectional = 1
            self.num_directions = 2
        else:
            self.bidirectional = 0
            self.num_directions = 1
        self.device = device

        # Proporcao entre dados de treino e de validacao
        self.train_percentage = 1 - self.validation_percent
        # Se usaremos packed_sequences no LSTM ou nao
        self.packing_sequence = False

        self.loss_function = None
        self.optimizer = None

        self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size, batch_first=True, num_layers=self.n_lstm_units, bidirectional=bool(self.bidirectional))

        self.linear = nn.Linear(self.num_directions * self.hidden_layer_size, self.output_size)

        # We train using multiple inputs (mini_batch), so we let this cell ready
        # to be called.
        self.hidden_cell = (torch.zeros(self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size).to(self.device),
                            torch.zeros(self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size).to(self.device))

        return

    def forward(self, input_seq):
        """
Classic forward method of every PyTorch model, as fast as possible. Receives an
input sequence and returns the prediction for the final step.

        :param input_seq: Input seuqnece of the time series.
        :return: The prediction in the end of the series.
        """
        # (seq_len, batch, input_size), mas pode inverter o
        # batch com o seq_len se fizer batch_first==1 na criacao do LSTM
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)

        if self.packing_sequence is True:
            # Para desempacotarmos a packed sequence, usamos esta funcao, que
            # preenche com zeros as sequencia para possuirem todas o mesmo tamanho
            # na matriz de saida, mas informa onde cada uma delas ACABA no segundo
            # elemento de sua tupla (inverse[1]).
            inverse = pad_packed_sequence(lstm_out, batch_first=True)

            # A saida do LSTM eh uma packed sequence, pois cada sequencia tem um
            # tamanho diferente. So nos interessa o ultimo elemento de cada
            # sequencia, que esta informado no segundo elemento (inverse[1]).
            # Fazemos o slicing do numpy selecionando todas as colunas com 'arange'
            # e o elemento que desejamos'inverse[1] -1'.
            lstm_out = inverse[0][arange(inverse[0].shape[0]), inverse[1] - 1]
        else:
            # All batch size, whatever sequence length, forward direction and
            # lstm output size (hidden size).
            # We only want the last output of lstm (end of sequence), that is
            # the reason of '[:,-1,:]'.
            lstm_out = lstm_out.view(input_seq.shape[0], -1, self.num_directions * self.hidden_layer_size)[:, -1, :]

        predictions = self.linear(lstm_out)

        return predictions

    def fit(self, X, y):
        """
This method contains the customized script for training this estimator. It must
be adjusted whenever the network structure changes.

        :param X: Input X data as numpy array. Each sample may have different length.
        :param y: Respective output for each input sequence. Also numpy array
        :return: Trained model with best validation loss found (it uses checkpoint).
        """
        # =====DATA-PREPARATION=================================================
        # y numpy array values into torch tensors
        self.train()
        self.packing_sequence = True
        self.to(self.device)
        if not isinstance(y, torch.Tensor): y = torch.from_numpy(y.astype("float32"))
        y = y.to(self.device).view(-1, self.output_size)
        # split into mini batches
        y_batches = torch.split(y, split_size_or_sections=self.training_batch_size)

        # Como cada tensor tem um tamanho Diferente, colocamos eles em uma
        # lista (que nao reclama de tamanhos diferentes em seus elementos).
        if not isinstance(X, torch.Tensor):
            lista_X = [torch.from_numpy(i.astype("float32")).view(-1, self.input_size).to(self.device) for i in X]
        else:
            lista_X = [i.view(-1, self.input_size) for i in X]
        X_batches = split_into_chunks(lista_X, self.training_batch_size)

        # pytorch only accepts different sizes tensors inside packed_sequences.
        # Then we need to convert it.
        aux_list = []
        for i in X_batches:
            aux_list.append(pack_sequence(i, enforce_sorted=False))
        X_batches = aux_list
        # =====fim-DATA-PREPARATION=============================================

        epochs = self.epochs
        best_validation_loss = 999999
        if self.loss_function is None: self.loss_function = nn.MSELoss()
        if self.optimizer is None: self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        f = open("loss_log.csv", "w")
        w = csv.writer(f)
        w.writerow(["epoch", "training_loss", "val_loss"])

        tqdm_bar = tqdm(range(epochs))
        for i in tqdm_bar:
            training_loss = 0
            validation_loss = 0
            for j, (X, y) in enumerate(zip(X_batches[:int(len(X_batches) * (1.0 - self.validation_percent))], y_batches[:int(len(y_batches) * (1.0 - self.validation_percent))])):
                self.optimizer.zero_grad()
                # Precisamos resetar o hidden state do LSTM a cada batch, ou
                # ocorre erro no backward(). O tamanho do batch para a cell eh
                # simplesmente o tamanho do batch em y ou X (tanto faz).
                self.hidden_cell = (torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device),
                                    torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device))

                y_pred = self(X)

                single_loss = self.loss_function(y_pred, y)
                single_loss.backward()
                self.optimizer.step()

                training_loss += single_loss
            # Tira a media das losses.
            training_loss = training_loss / (j + 1)

            for j, (X, y) in enumerate(zip(X_batches[int(len(X_batches) * (1.0 - self.validation_percent)):], y_batches[int(len(y_batches) * (1.0 - self.validation_percent)):])):
                self.hidden_cell = (torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device),
                                    torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device))
                y_pred = self(X)

                single_loss = self.loss_function(y_pred, y)

                validation_loss += single_loss
            # Tira a media das losses.
            validation_loss = validation_loss / (j + 1)

            # Checkpoint to best models found.
            if best_validation_loss > validation_loss:
                # Update the new best loss.
                best_validation_loss = validation_loss
                # torch.save(self, "{:.15f}".format(best_validation_loss) + "_checkpoint.pth")
                torch.save(self, "best_model.pth")
                torch.save(self.state_dict(), "best_model_state_dict.pth")

            tqdm_bar.set_description(f'epoch: {i:1} train_loss: {training_loss.item():10.10f}' + f' val_loss: {validation_loss.item():10.10f}')
            w.writerow([i, training_loss.item(), validation_loss.item()])
            f.flush()
        f.close()

        self.eval()
        self.packing_sequence = False

        # At the end of training, save the final model.
        torch.save(self, "last_training_model.pth")

        # Update itself with BEST weights foundfor each layer.
        self.load_state_dict(torch.load("best_model_state_dict.pth"))

        self.eval()

        # Returns the best model found so far.
        return torch.load("best_model.pth")

    def fit_dataloading(self):
        """
This method contains the customized script for training this estimator. Data is
obtained with PyTorch's dataset and dataloader classes for memory efficiency
when dealing with big datasets. Otherwise loading the whole dataset would
overflow the memory.

        :return: Trained model with best validation loss found (it uses checkpoint).
        """
        self.train()
        self.packing_sequence = True
        self.to(self.device)
        # =====DATA-PREPARATION=================================================
        room2_tum_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-room2_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room2_512_16/mav0/mocap0/data.csv", convert_first=True, device=self.device, min_window_size=100, max_window_size=350)
        train_dataset = Subset(room2_tum_dataset, arange(int(len(room2_tum_dataset) * self.train_percentage)))
        val_dataset = Subset(room2_tum_dataset, arange(int(len(room2_tum_dataset) * self.train_percentage), len(room2_tum_dataset)))

        # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        train_loader = PackingSequenceDataloader(train_dataset, batch_size=128, shuffle=True)
        val_loader = PackingSequenceDataloader(val_dataset, batch_size=128, shuffle=True)
        # =====fim-DATA-PREPARATION=============================================

        epochs = self.epochs
        best_validation_loss = 999999
        if self.loss_function is None: self.loss_function = nn.MSELoss()
        if self.optimizer is None: self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        f = open("loss_log.csv", "w")
        w = csv.writer(f)
        w.writerow(["epoch", "training_loss", "val_loss"])

        tqdm_bar = tqdm(range(epochs))
        for i in tqdm_bar:
            train_manager = DataManager(train_loader, device=self.device, buffer_size=3)
            val_manager = DataManager(val_loader, device=self.device, buffer_size=3)
            training_loss = 0
            validation_loss = 0
            self.optimizer.zero_grad()
            for j, (X, y) in enumerate(train_manager):

                # Fazemos a otimizacao a cada MINI BATCH size
                if (j + 1) % self.training_batch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Precisamos resetar o hidden state do LSTM a cada batch, ou
                # ocorre erro no backward(). O tamanho do batch para a cell eh
                # simplesmente o tamanho do batch em y ou X (tanto faz).
                self.hidden_cell = (torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device),
                                    torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device))

                y_pred = self(X)

                # Repare que NAO ESTAMOS acumulando a LOSS.
                single_loss = self.loss_function(y_pred, y)
                # Cada chamada ao backprop eh ACUMULADA no gradiente (optimizer)
                single_loss.backward()

                # .item() converts to numpy and therefore detach pytorch gradient.
                # Otherwise, it would try backpropagate whole dataset and may crash vRAM memory
                training_loss += single_loss.item()

            # O ultimo batch pode nao ter o mesmo tamanho que os demais e nao entrar no "if"
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Tira a media das losses.
            training_loss = training_loss / (j + 1)

            for j, (X, y) in enumerate(val_manager):
                self.hidden_cell = (torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device),
                                    torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device))
                y_pred = self(X)

                single_loss = self.loss_function(y_pred, y)

                # .item() converts to numpy and therefore detach pytorch gradient.
                # Otherwise, it would try backpropagate whole dataset and may crash vRAM memory
                validation_loss += single_loss.item()
            # Tira a media das losses.
            validation_loss = validation_loss / (j + 1)

            # Checkpoint to best models found.
            if best_validation_loss > validation_loss:
                # Update the new best loss.
                best_validation_loss = validation_loss
                # torch.save(self, "{:.15f}".format(best_validation_loss) + "_checkpoint.pth")
                torch.save(self, "best_model.pth")
                torch.save(self.state_dict(), "best_model_state_dict.pth")

            tqdm_bar.set_description(f'epoch: {i:1} train_loss: {training_loss:10.10f}' + f' val_loss: {validation_loss:10.10f}')
            w.writerow([i, training_loss, validation_loss])
            f.flush()
        f.close()

        self.eval()

        # At the end of training, save the final model.
        torch.save(self, "last_training_model.pth")

        # Update itself with BEST weights foundfor each layer.
        self.load_state_dict(torch.load("best_model_state_dict.pth"))

        self.eval()

        # Returns the best model found so far.
        return torch.load("best_model.pth")

    def get_params(self, *args, **kwargs):
        """
Get parameters for this estimator.

        :param args: Always ignored, exists for compatibility.
        :param kwargs: Always ignored, exists for compatibility.
        :return: Dict containing all parameters for this estimator.
        """
        return {"input_size": self.input_size,
                "hidden_layer_size": self.hidden_layer_size,
                "output_size": self.output_size,
                "n_lstm_units": self.n_lstm_units,
                "epochs": self.epochs,
                "training_batch_size": self.training_batch_size,
                "validation_percent": self.validation_percent,
                "bidirectional": self.bidirectional,
                "device": self.device}

    def predict(self, X):
        """
Predict using this pytorch model. Useful for sklearn search and/or mini-batch prediction.

        :param X: Input data of shape (n_samples, n_features).
        :return: The y predicted values.
        """
        # This method (predict) is intended to be used within training procces.
        self.eval()
        self.packing_sequence = False

        # Como cada tensor tem um tamanho Diferente, colocamos eles em uma
        # lista (que nao reclama de tamanhos diferentes em seus elementos).
        if not isinstance(X, torch.Tensor):
            lista_X = [torch.from_numpy(i.astype("float32")).view(-1, self.input_size).to(self.device) for i in X]
        else:
            lista_X = [i.view(-1, self.input_size) for i in X]

        y = []

        for X in lista_X:
            X = X.view(1, -1, self.input_size)
            self.hidden_cell = (torch.zeros(self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size).to(self.device),
                                torch.zeros(self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size).to(self.device))
            y.append(self(X))

        return torch.as_tensor(y).view(-1, self.output_size).detach().cpu().numpy()

    def score(self, X, y, **kwargs):
        """
Return the RMSE error score of the prediction.

        :param X: Input data os shape (n_samples, n_features).
        :param y: Predicted y values of shape (n_samples, n_outputs).)
        :param kwargs: Always ignored, exists for compatibility.
        :return: RMSE score.
        """
        # Se for um tensor, devemos converter antes para array numpy, ou o
        # sklearn retorna erro na RMSE.
        if isinstance(y, torch.Tensor):
            y = y.numpy()

        return make_scorer((mean_squared_error(self.predict(X).cpu().detach().numpy(), y)) ** 1 / 2, greater_is_better=False)

    def set_params(self, **params):
        """
Set the parameters of this estimator.

        :param params: (Dict) Estimator parameters.
        :return: Estimator instance.
        """
        input_size = params.get('input_size')
        hidden_layer_size = params.get('hidden_layer_size')
        output_size = params.get('output_size')
        n_lstm_units = params.get('n_lstm_units')
        epochs = params.get('epochs')
        training_batch_size = params.get('training_batch_size')
        validation_percent = params.get('validation_percent')
        bidirectional = params.get('bidirectional')
        device = params.get('device')

        if input_size:
            self.input_size = input_size
            self.__reinit_params__()
        if hidden_layer_size:
            self.hidden_layer_size = hidden_layer_size
            self.__reinit_params__()
        if output_size:
            self.output_size = output_size
            self.__reinit_params__()
        if n_lstm_units:
            self.n_lstm_units = n_lstm_units
            self.__reinit_params__()
        if epochs:
            self.epochs = epochs
            self.__reinit_params__()
        if training_batch_size:
            self.training_batch_size = training_batch_size
            self.__reinit_params__()
        if validation_percent:
            self.validation_percent = validation_percent
            self.__reinit_params__()
        if bidirectional is not None:
            if bidirectional:
                self.bidirectional = 1
                self.num_directions = 2
            else:
                self.bidirectional = 0
                self.num_directions = 1
        if device:
            self.device = device
            self.__reinit_params__()

        return self

    def __reinit_params__(self):
        """
Useful for updating params when 'set_params' is called.
        """

        self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size, batch_first=True, num_layers=self.n_lstm_units, bidirectional=bool(self.bidirectional))

        self.linear = nn.Linear(self.num_directions * self.hidden_layer_size, self.output_size)

        # We train using multiple inputs (mini_batch), so we let this cell ready
        # to be called.
        self.hidden_cell = (torch.zeros(self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size).to(self.device),
                            torch.zeros(self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size).to(self.device))

    def __assemble_packed_seq__(self):
        pass


def alinha_dataset_tum(imu_data, ground_truth, impossible_value=-444444):
    """
Funcao especifica para abrir e alinhar IMU e ground truth no formato de dataset
da TUM. Mto chata mds

    :param imu_data: Array-like (2D) with samples from IMU.
    :param ground_truth: Array-like (2D) with samples from ground truth.
    :param impossible_value: We define an impossible value for we to distinguish where we left unfilled elements.
    :return: IMU original input and ground truth with unfilled values.
    """
    # Initialize aux matrix with impossible value.
    aux_matrix = ones((imu_data.shape[0], ground_truth.shape[1])) * impossible_value

    # Now we try to align ground truth with IMU timestamp (undersampling).
    for sample in ground_truth:
        _, index = find_nearest(imu_data[:, 0], sample[0])
        aux_matrix[index] = sample

    return imu_data, aux_matrix


def format_dataset(dataset_directory="dataset-room2_512_16", enable_asymetrical=True, sampling_window_size=10, file_format="npy"):
    """
A utilidade desta função eh fazer o split do dataset no formato de serie
temporal e convete-lo para arquivos numpy (NPZ), de forma que a classe que criamos
para dataset no PyTorch possa opera-lo sem estouro de memoria RAM. Pois se
abrissemos ele inteiro de uma vez, aconteceria overflow de memoria.

O formato de dataset esperado eh o dataset visual-inercial da TUM.

    :param dataset_directory: Diretorio onde se encontra o dataset (inclui o nome da sequencia).
    :param sampling_window_size: (ignorado) O tamanho da janela de entrada na serie temporal (Se for simetrica. Do contrario, ignorado).
    :param enable_asymetrical: (ignorado) Se a serie eh assimetrica.
    :return: Arrays X e y completos.
    :param file_format: NPY (deafult): 1 Arquivo para X e outro para Y. NPZ: Cada linha das matrizes X e Y gera 1 arquivo dentro dos NPZ de saida.
    """
    counter = 0
    for interval in tqdm(range(1, 201)[::-1], desc="Split do dataset"):
        # Opening dataset.
        input_data = read_csv(dataset_directory + "/mav0/imu0/data.csv").to_numpy()
        output_data = read_csv(dataset_directory + "/mav0/mocap0/data.csv").to_numpy()

        # ===============DIFF=======================================================
        # Precisamos restaurar o time para alinhar os dados depois do "diff"
        original_ground_truth_timestamp = output_data[:, 0]

        # inutil agora, mas deixarei aqui pra nao ter que refazer depois
        original_imu_timestamp = input_data[:, 0]

        # Queremos apenas a VARIACAO de posicao a cada instante.
        output_data = difference(output_data, interval=interval)
        # Restauramos a referencia de time original.
        output_data[:, 0] = original_ground_truth_timestamp[interval:]
        # ===============fim-de-DIFF================================================

        # features without timestamp (we do not scale timestamp)
        input_features = input_data[:, 1:]
        output_features = output_data[:, 1:]

        # Scaling data
        input_scaler = StandardScaler()
        input_features = input_scaler.fit_transform(input_features)
        output_scaler = MinMaxScaler()
        output_features = output_scaler.fit_transform(output_features)

        # Replacing scaled data (we kept the original TIMESTAMP)
        input_data[:, 1:] = input_features
        output_data[:, 1:] = output_features

        # A IMU e o ground truth nao sao coletados ao mesmo tempo, precisamos alinhar.
        x, y = alinha_dataset_tum(input_data, output_data)

        # Depois de alinhado, timestamp nao nos importa mais. Vamos descartar
        x = x[:, 1:]
        y = y[:, 1:]

        # Divido x e y em diferentes conjuntos de mesmo tamanho e alinhados
        x_chunks = split_into_chunks(x, 10 ** 9)
        y_chunks = split_into_chunks(y, 10 ** 9)

        # Vou concatenando os dados ja "splittados" nestes arrays
        X_array = None
        y_array = None

        # A partir daqui, o bagulho fica louco. Vou dividir a entrada em pequenos
        # datasets e fazer o split deles individualente para ter mais granularidade
        # no tamanho das sequencias e nao resultar tambem em senquencias muito
        # longas (com o tamanho do dataset inteiro).
        for x, y in list(zip(x_chunks, y_chunks)):
            # Fazemos o carregamento correto no formato de serie temporal
            X, y = timeseries_dataloader(data_x=x, data_y=y, enable_asymetrical=False, sampling_window_size=interval)
            # Um ajuste na dimensao do y pois prevemos so o proximo passo.
            y = y.reshape(-1, 7)

            X = X.astype("float32")
            y = y.astype("float32")

            # Agora jogamos fora os valores onde nao ha ground truth e serviram apenas
            # para fazermos o alinhamento e o dataloader correto.
            samples_validas = where(y > -44000, True, False)
            X = X[samples_validas[:, 0]]
            y = y[samples_validas[:, 0]]

            if X_array is None and y_array is None:
                X_array = X.copy()
                y_array = y.copy()
            else:
                X_array = concatenate((X_array, X.copy()))
                y_array = concatenate((y_array, y.copy()))

        # Apena para manter o padrao de nomenclatura
        X = X_array
        y = y_array

        keys_to_concatenate = ["arr_" + str(i) for i in range(counter, X.shape[0] + counter)]
        counter += X.shape[0]

        if file_format.lower() == "npy":
            with open("x_data.npy", "wb") as x_file, open("y_data.npy", "wb") as y_file:
                save(x_file, X)
                save(y_file, y)
        else:
            with open("tmp_x/x_data" + str(counter) + ".npz", "wb") as x_file, open("tmp_y/y_data" + str(counter) + ".npz", "wb") as y_file:
                # Asterisco serve pra abrir a LISTA como se fosse *args.
                # Dois asteriscos serviriam pra abrir um DICIONARIO como se fosse **kwargs.
                savez(x_file, **dict(zip(keys_to_concatenate, X)))
                savez(y_file, **dict(zip(keys_to_concatenate, y)))

    return X, y


def join_npz_files(files_origin_path="./", output_file="./x_data.npz"):
    with open(output_file, "wb") as file:
        npfiles = glob.glob(os.path.normpath(files_origin_path) + "/" + "*.npz")
        npfiles.sort()
        all_arrays = []
        for i, npfile in enumerate(npfiles):
            npz_file = load(npfile)
            files_names = npz_file.files
            all_arrays.extend([npz_file[file_name] for file_name in files_names])  # , mmap_mode="r"
        savez(file, *all_arrays)
    return


def experiment(repeats):
    """
Runs the experiment itself.

    :param repeats: Number of times to repeat the experiment. When we are trying to create a good network, it is reccomended to use 1.
    :return: Error scores for each repeat.
    """

    # Recebe os arquivos do dataset e o aloca de no formato (numpy npz) adequado.
    # X, y = format_dataset(dataset_directory="dataset-room2_512_16", file_format="NPZ")
    # join_npz_files(files_origin_path="./tmp_x", output_file="./x_data.npz")
    # join_npz_files(files_origin_path="./tmp_y", output_file="./y_data.npz")
    # return

    model = LSTM(input_size=6, hidden_layer_size=300, n_lstm_units=1, bidirectional=False,
                 output_size=7, training_batch_size=10, epochs=50, device=device)
    model.to(device)

    # Gera os parametros de entrada aleatoriamente. Alguns sao uniformes nos
    # EXPOENTES.
    hidden_layer_size = random.uniform(40, 80, 20).astype("int")
    n_lstm_units = arange(1, 4)

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'hidden_layer_size': hidden_layer_size, 'n_lstm_units': n_lstm_units}

    splitter = TimeSeriesSplitCV(n_splits=2,
                                 training_percent=0.8,
                                 blocking_split=False)
    regressor = model
    cv_search = \
        BayesSearchCV(estimator=regressor, cv=splitter,
                      search_spaces=parametros,
                      refit=True,
                      n_iter=4,
                      verbose=1,
                      # n_jobs=4,
                      scoring=make_scorer(mean_squared_error,
                                          greater_is_better=False,
                                          needs_proba=False))

    # Let's go fit! Comment if only loading pretrained model.
    # model.fit(X, y)
    model.fit_dataloading()

    # Realizamos a busca atraves do treinamento
    # cv_search.fit(X, y.reshape(-1, 1))
    # print(cv_search.cv_results_)
    # cv_dataframe_results = DataFrame.from_dict(cv_search.cv_results_)
    # cv_dataframe_results.to_csv("cv_results.csv")

    # =====================PREDICTION-TEST======================================
    dataset_directory = "dataset-room2_512_16"

    # Opening dataset.
    input_data = read_csv(dataset_directory + "/mav0/imu0/data.csv").to_numpy()
    output_data = read_csv(dataset_directory + "/mav0/mocap0/data.csv").to_numpy()

    # Precisamos restaurar o time para alinhar os dados depois do "diff"
    original_ground_truth_timestamp = output_data[:, 0]

    # inutil agora, mas deixarei aqui pra nao ter que refazer depois
    original_imu_timestamp = input_data[:, 0]

    plt.close()
    plt.plot(output_data[:, 1])
    plt.show()

    plt.close()
    plt.plot(output_data[1:, 1] - output_data[:-1, 1])
    plt.show()

    # Queremos apenas a VARIACAO de posicao a cada instante.
    output_data = diff(output_data, axis=0)

    plt.close()
    plt.plot(output_data[:, 1])
    plt.show()

    # Restauramos a referencia de time original.
    output_data[:, 0] = original_ground_truth_timestamp[1:]

    # features without timestamp (we do not scale timestamp)
    input_features = input_data[:, 1:]
    output_features = output_data[:, 1:]

    # Scaling data
    input_scaler = StandardScaler()
    input_features = input_scaler.fit_transform(input_features)
    output_scaler = MinMaxScaler()
    output_features = output_scaler.fit_transform(output_features)

    # These arrays/tensors are only helpful for plotting the prediction.
    X_graphic = torch.from_numpy(input_features.astype("float32")).to(device)
    y_graphic = output_features.astype("float32")

    # model = cv_search.best_estimator_
    model = torch.load("best_model.pth")
    # model.load_state_dict(torch.load("best_model_state_dict.pth"))
    model.to(device)
    model.packing_sequence = False
    yhat = []
    model.hidden_cell = (torch.zeros(model.num_directions * model.n_lstm_units, 1, model.hidden_layer_size).to(model.device),
                         torch.zeros(model.num_directions * model.n_lstm_units, 1, model.hidden_layer_size).to(model.device))
    model.eval()
    for X in X_graphic:
        yhat.append(model(X.view(1, -1, 6)).detach().cpu().numpy())
    # from list to numpy array
    yhat = array(yhat).reshape(-1, 7)

    # ======================PLOT================================================
    dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    for i, dim_name in enumerate(dimensoes):
        plt.close()
        plt.plot(original_imu_timestamp, yhat[:, i], original_ground_truth_timestamp[1:], y_graphic[:, i])
        plt.legend(['predict', 'reference'], loc='upper right')
        plt.savefig(dim_name + ".png", dpi=200)
        plt.show()
    # rmse = mean_squared_error(yhat, y_graphic) ** 1 / 2
    # print("RMSE trajetoria inteira: ", rmse)

    error_scores = []

    return error_scores


if __name__ == '__main__':

    # plot_csv()

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Usando GPU")
    else:
        dev = "cpu"
        print("Usando CPU")
    device = torch.device(dev)

    # dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-room2_512_16/mav0/imu0/data.csv",
    #                                        y_csv_path="dataset-room2_512_16/mav0/mocap0/data.csv",
    #                                        convert_first=True)

    # print("abrindo dataset")
    # t0 = time()
    # room2_tum_dataset = GenericDatasetFromFiles(data_path="../drive/My Drive/PODE APAGAR/dataset-tum-room2/", convert_first=True, device=device)
    # print("tempo abrindo dataset:", time() - t0)
    #
    # print("abrindo loader")
    # train_loader = PackingSequenceDataloader(room2_tum_dataset, batch_size=128, shuffle=True)
    # loader_iterable = iter(train_loader)
    #
    # print("abrindo manager")
    # train_manager = iter(DataManager(train_loader, device=device, buffer_size=3))
    #
    # while True:
    #     print("esperando")
    #     sleep(1)
    #     t0 = time()
    #     x = next(train_manager)
    #     print("tempo de resposta manager:", time() - t0)

    # aux = zeros((1000,))
    # for i in tqdm(dataset):
    #     aux[i[0].shape[0]] += 1
    #
    # plt.plot(aux)
    # plt.show()

    experiment(1)

