import csv
from math import sin, cos

import numpy
import torch
from numpy import arange
from pandas import Series
from ptk.timeseries import *
from ptk.utils import *
from skimage.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import LSTM
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
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

    return cos(x) + 2 * sin(2 * x)


def fake_acceleration(x):
    """
Generate a simulated one dimensional acceleration for training on "1d dataset" an test if the neural network is able to generalize well.

Given "x" values, calculate f''(x) for this.

It is supposed to be the input for the neural network.

    :param x: Array of values to calculate f''(x) = y''.
    :return: Array os respective acceleration in "y" axis.
    """

    return 4 * cos(2 * x) - sin(x)


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


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, n_lstm_units=1, epochs=150, training_batch_size=64, validation_percent=0.2, bidirectional=False, device=torch.device("cpu")):
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

        self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size, batch_first=True, num_layers=self.n_lstm_units, bidirectional=bool(self.bidirectional))

        self.linear = nn.Linear(self.num_directions * self.hidden_layer_size, self.output_size)

        # We train using multiple inputs (mini_batch), so we let this cell ready
        # to be called.
        self.hidden_cell_training = (torch.zeros(self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size).to(self.device),
                                     torch.zeros(self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size).to(self.device))

        # We predict using single input per time, so we let single batch cell
        # ready here.
        self.hidden_cell_prediction = (torch.zeros(self.num_directions * self.n_lstm_units, 1, self.hidden_layer_size).to(self.device),
                                       torch.zeros(self.num_directions * self.n_lstm_units, 1, self.hidden_layer_size).to(self.device))

        return

    def forward(self, input_seq):
        # (seq_len, batch, input_size), mas pode inverter o
        # batch com o seq_len se fizer batch_first==1 na criacao do LSTM
        lstm_out, self.hidden_cell_training = self.lstm(input_seq, self.hidden_cell_training)

        if self.training is True:
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
        # =====DATA-PREPARATION=================================================
        # y numpy array values into torch tensors
        self.train()
        if not isinstance(y, torch.Tensor): y = torch.from_numpy(y.astype("float32"))
        y = y.to(self.device)
        # split into mini batches
        y_batches = torch.split(y, split_size_or_sections=self.training_batch_size)

        # Como cada tensor tem um tamanho Diferente, colocamos eles em uma
        # lista (que nao reclama de tamanhos diferentes em seus elementos).
        if not isinstance(X, torch.Tensor):
            lista_X = [torch.from_numpy(i.astype("float32")).view(-1, self.output_size).to(self.device) for i in X]
        else:
            lista_X = [i.view(-1, self.output_size) for i in X]
        X_batches = split_into_chunks(lista_X, self.training_batch_size)

        # pytorch only accepts different sizes tensors inside packed_sequences.
        # Then we need to convert it.
        aux_list = []
        for i in X_batches:
            aux_list.append(pack_sequence(i, enforce_sorted=False))
        X_batches = aux_list
        # =====fim-DATA-PREPARATION=============================================

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        epochs = self.epochs
        best_validation_loss = 999999

        f = open("loss_log.csv", "a")
        w = csv.writer(f)
        w.writerow(["epoch", "training_loss", "val_loss"])

        for i in tqdm(range(epochs)):
            training_loss = 0
            validation_loss = 0
            for j, (X, y) in enumerate(zip(X_batches[:int(len(X_batches) * self.validation_percent)], y_batches[:int(len(y_batches) * self.validation_percent)])):
                optimizer.zero_grad()
                # Precisamos resetar o hidden state do LSTM a cada batch, ou
                # ocorre erro no backward(). O tamanho do batch para a cell eh
                # simplesmente o tamanho do batch em y ou X (tanto faz).
                self.hidden_cell_training = (torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device),
                                             torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device))

                y_pred = self(X)

                single_loss = loss_function(y_pred, y)
                single_loss.backward()
                optimizer.step()

                training_loss += single_loss
            # Tira a media das losses.
            training_loss = training_loss / (j + 1)

            for j, (X, y) in enumerate(zip(X_batches[int(len(X_batches) * self.validation_percent):], y_batches[int(len(y_batches) * self.validation_percent):])):
                self.hidden_cell_training = (torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device),
                                             torch.zeros(self.num_directions * self.n_lstm_units, y.shape[0], self.hidden_layer_size).to(self.device))
                y_pred = self(X)

                single_loss = loss_function(y_pred, y)

                validation_loss += single_loss
            # Tira a media das losses.
            validation_loss = validation_loss / (j + 1)

            # Checkpoint to best models found.
            if best_validation_loss > validation_loss:
                # Update the new best loss.
                best_validation_loss = validation_loss
                # torch.save(self, "{:.15f}".format(best_validation_loss) + "_checkpoint.pth")
                torch.save(self, "best_model_epoch_" + str(i) + ".pth")

            print(f'\nepoch: {i:1} train_loss: {training_loss.item():10.10f}', f'val_loss: {validation_loss.item():10.10f}')
            w.writerow([i, training_loss.item(), validation_loss.item()])
            f.flush()
        f.close()

        # At the end of training, save the final model.
        torch.save(self, "last_training_model.pth")

        self.eval()

        return self

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
Predict using this pytorch model. Useful for sklearn and/or mini-batch prediction.

        :param X: Input data of shape (n_samples, n_features).
        :return: The y predicted values.
        """
        # This method (predict) is intended to be used within training procces.
        self.train()

        # Como cada tensor tem um tamanho Diferente, colocamos eles em uma
        # lista (que nao reclama de tamanhos diferentes em seus elementos).
        if not isinstance(X, torch.Tensor):
            lista_X = [torch.from_numpy(i.astype("float32")).view(-1, self.output_size).to(self.device) for i in X]
        else:
            lista_X = [i.view(-1, self.output_size) for i in X]

        self.hidden_cell_training = (torch.zeros(self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size).to(self.device),
                                     torch.zeros(self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size).to(self.device))

        X = pack_sequence(lista_X, enforce_sorted=False)

        predictions = self(X)

        self.eval()

        return predictions

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
        self.hidden_cell_training = (torch.zeros(self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size).to(self.device),
                                     torch.zeros(self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size).to(self.device))

        # We predict using single input per time, so we let single batch cell
        # ready here.
        self.hidden_cell_prediction = (torch.zeros(self.num_directions * self.n_lstm_units, 1, self.hidden_layer_size).to(self.device),
                                       torch.zeros(self.num_directions * self.n_lstm_units, 1, self.hidden_layer_size).to(self.device))


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
    raw_accel = [fake_acceleration(i / 30) for i in range(-300, 300)]
    diff_accel = difference(raw_accel, 1)
    # diff_accel = numpy.array(raw_accel)

    # raw_timestamp = range(len(raw_pos))
    # # diff_timestamp = difference(numpy.array(raw_timestamp) + numpy.random.rand(len(raw_pos)), 1)
    # diff_timestamp = array(raw_timestamp)

    # testar com lstm BIDIRECIONAL
    model = LSTM(input_size=1, hidden_layer_size=100, n_lstm_units=2, bidirectional=False,
                 output_size=1, training_batch_size=60, epochs=4, device=device)

    raw_pos = raw_pos[1:]
    raw_accel = raw_accel[1:]

    raw_accel = array(raw_accel)
    raw_pos = array(raw_pos)
    diff_accel = array(diff_accel)
    diff_pos = array(diff_pos)

    X_scaler = StandardScaler()
    raw_accel = X_scaler.fit_transform(raw_accel.reshape(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    diff_pos = y_scaler.fit_transform(diff_pos.reshape(-1, 1))

    raw_accel = raw_accel.reshape(-1)
    diff_pos = diff_pos.reshape(-1)

    X, y = timeseries_dataloader(data_x=raw_accel, data_y=diff_pos, enable_asymetrical=True)

    # Invertendo para a sequencia ser DEcrescente.
    X = X[::-1]
    y = y[::-1]

    # enabling CUDA
    model.to(device)
    # Let's go fit
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
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Usando GPU")
    else:
        dev = "cpu"
        print("Usando CPU")
    device = torch.device(dev)

    experiment(1)
