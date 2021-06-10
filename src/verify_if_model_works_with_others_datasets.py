from matplotlib import pyplot as plt
import csv

import torch
from matplotlib import pyplot as plt
from numpy import arange, array
from numpy import random
from numpy.linalg import norm
from skimage.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from torch import nn, tensor, movedim
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Sequential, Conv1d
from torch.utils.data import Subset, ConcatDataset
from tqdm import tqdm

from mydatasets import *
from ptk.timeseries import *
from ptk.utils import *


class ResBlock(nn.Module):
    def __init__(self, n_input_channels=6, n_output_channels=7,
                 kernel_size=(7,), stride=(1,), padding=0, dilation=(1,),
                 groups=1, bias=True, padding_mode='replicate'):
        """
    ResNet-like block, receives as arguments the same that PyTorch's Conv1D
    module.
        """
        super(ResBlock, self).__init__()

        self.feature_extractor = \
            Sequential(
                Conv1d(n_input_channels, n_output_channels, kernel_size,
                       stride, kernel_size[0] // 2 * dilation, dilation,
                       groups, bias, padding_mode),
                nn.PReLU(), nn.BatchNorm1d(n_output_channels),
                Conv1d(n_output_channels, n_output_channels, kernel_size,
                       stride, kernel_size[0] // 2 * dilation + padding,
                       dilation, groups, bias, padding_mode),
                nn.PReLU(), nn.BatchNorm1d(n_output_channels)
            )

        self.skip_connection = \
            Sequential(
                Conv1d(n_input_channels, n_output_channels, (1,),
                       stride, padding, dilation, groups, bias, padding_mode)
            )

    def forward(self, input_seq):
        return self.feature_extractor(input_seq) + self.skip_connection(input_seq)


class SumLayer(nn.Module):
    def __init__(self, n_input_channels):
        """
    This layer aims to sum the last dimension of a tensor and compute the batch
    norm.

        :param n_input_channels: How many channels in the entrying tensor.
        """
        super(SumLayer, self).__init__()

        self.bn1 = nn.BatchNorm1d(n_input_channels)

    def forward(self, input_seq):
        return torch.sum(input_seq, dim=-1, keepdim=True)


class InertialModule(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, n_lstm_units=1, epochs=150, training_batch_size=64, validation_percent=0.2, bidirectional=False, device=torch.device("cpu"),
                 use_amp=True):
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
        self.use_amp = use_amp  # Automatic Mixed Precision (float16 and float32)

        # Proporcao entre dados de treino e de validacao
        self.train_percentage = 1 - self.validation_percent
        # Se usaremos packed_sequences no LSTM ou nao
        self.packing_sequence = False

        self.loss_function = None
        self.optimizer = None

        # ATTENTION: You cannot change this anymore, since w added a sum layer
        # and it casts conv outputs to 1 feature per channel
        pooling_output_size = 1

        n_base_filters = 64
        n_output_features = 512
        self.feature_extractor = \
            Sequential(
                # Conv1d(input_size, 1 * n_base_filters, (7,), dilation=(1,), stride=(1,)), nn.PReLU(), nn.BatchNorm1d(1 * n_base_filters, affine=True),
                # ResBlock(1 * n_base_filters, 2 * n_base_filters, (7,), dilation=1, stride=1),
                # ResBlock(2 * n_base_filters, 4 * n_base_filters, (7,), dilation=1, stride=1),
                # ResBlock(4 * n_base_filters, n_output_features, (7,), dilation=1, stride=1),
                Conv1d(input_size, 1 * n_base_filters, (7,)), nn.PReLU(), nn.Dropout2d(),  # nn.BatchNorm1d(1 * n_base_filters),
                Conv1d(1 * n_base_filters, 2 * n_base_filters, (7,)), nn.PReLU(), nn.Dropout2d(),  # nn.BatchNorm1d(2 * n_base_filters),
                Conv1d(2 * n_base_filters, 3 * n_base_filters, (7,)), nn.PReLU(), nn.Dropout2d(),  # nn.BatchNorm1d(3 * n_base_filters),
                Conv1d(3 * n_base_filters, 4 * n_base_filters, (7,)), nn.PReLU(),  # nn.BatchNorm1d(4 * n_base_filters),
                Conv1d(4 * n_base_filters, n_output_features, (7,)), nn.PReLU(),  # nn.BatchNorm1d(n_output_features)
            )

        self.sum_layer = SumLayer(n_output_features)
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(pooling_output_size)

        self.dense_network = Sequential(
            nn.Linear(2 * pooling_output_size * n_output_features, 32), nn.PReLU(),
            # nn.Dropout(p=0.5),  # nn.BatchNorm1d(128, affine=True),
            # nn.Linear(32, 32), nn.PReLU(),
            # nn.BatchNorm1d(32, affine=True),
            nn.Linear(32, self.output_size)
        )
        # self.lstm = nn.LSTM(input_size, self.hidden_layer_size, batch_first=True, num_layers=self.n_lstm_units, bidirectional=bool(self.bidirectional))
        #
        # self.linear = nn.Linear(self.num_directions * self.hidden_layer_size, self.output_size)
        #
        # # We train using multiple inputs (mini_batch), so we let this cell ready
        # # to be called.
        # self.hidden_cell = (torch.zeros((self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size), device=self.device),
        #                     torch.zeros((self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size), device=self.device))

        return

    def forward(self, input_seq):
        """
Classic forward method of every PyTorch model, as fast as possible. Receives an
input sequence and returns the prediction for the final step.

        :param input_seq: Input sequence of the time series.
        :return: The prediction in the end of the series.
        """

        # As features (px, py, pz, qw, qx, qy, qz) sao os "canais" da
        # convolucao e precisam vir no meio para o pytorch
        input_seq = movedim(input_seq, -2, -1)

        input_seq = self.feature_extractor(input_seq)

        # # input_seq = movedim(input_seq, -2, -1)
        # #
        # # (seq_len, batch, input_size), mas pode inverter o
        # # batch com o seq_len se fizer batch_first==1 na criacao do LSTM
        # lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        #
        # # All batch size, whatever sequence length, forward direction and
        # # lstm output size (hidden size).
        # # We only want the last output of lstm (end of sequence), that is
        # # the reason of '[:,-1,:]'.
        # lstm_out = lstm_out.view(input_seq.shape[0], -1, self.num_directions * self.hidden_layer_size)[:, -1, :]
        #
        # predictions = self.linear(lstm_out)

        output_seq = \
            torch.cat((
                self.sum_layer(input_seq),
                self.adaptive_pooling(input_seq),
            ), dim=1)

        predictions = self.dense_network(output_seq.view(output_seq.shape[0], -1))

        return predictions

    def fit(self):
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
        room1_tum_dataset = BatchTimeseriesDataset(x_csv_path="datasets-novos-para-verificacao/input_dataset.csv", y_csv_path="datasets-novos-para-verificacao/noisy_output_dataset.csv",
                                                   min_window_size=100, max_window_size=350, batch_size=self.training_batch_size, shuffle=False, noise=(0, 0.03),
                                                   reference_x_csv_path="datasets-novos-para-verificacao/input_dataset.csv", reference_y_csv_path="datasets-novos-para-verificacao/noisy_output_dataset.csv")

        # # Diminuir o dataset para verificar o funcionamento de scripts
        # room1_tum_dataset = Subset(room1_tum_dataset, arange(int(len(room1_tum_dataset) * 0.001)))

        train_dataset = Subset(room1_tum_dataset, arange(int(len(room1_tum_dataset) * self.train_percentage)))
        val_dataset = Subset(room1_tum_dataset, arange(int(len(room1_tum_dataset) * self.train_percentage), len(room1_tum_dataset)))

        train_loader = CustomDataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True)
        val_loader = CustomDataLoader(dataset=val_dataset, batch_size=1, shuffle=True, pin_memory=True)

        # Pesos das saídas: px py pz wq wx wy wz (translacoes e quaternios).
        weights = array([1000, 1000, 1000, 1, 1, 1, 1])
        weights = tensor(weights / norm(weights)).to(device=self.device)

        # train_loader = PackingSequenceDataloader(train_dataset, batch_size=128, shuffle=True)
        # val_loader = PackingSequenceDataloader(val_dataset, batch_size=128, shuffle=True)
        # =====fim-DATA-PREPARATION=============================================

        epochs = self.epochs
        best_validation_loss = 999999
        if self.loss_function is None: self.loss_function = nn.MSELoss()
        if self.optimizer is None: self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.0)
        scaler = GradScaler(enabled=self.use_amp)

        f = open("loss_log.csv", "w")
        w = csv.writer(f)
        w.writerow(["epoch", "training_loss", "val_loss"])

        tqdm_bar = tqdm(range(epochs))
        for i in tqdm_bar:
            train_manager = DataManager(train_loader, device=self.device, buffer_size=1)
            val_manager = DataManager(val_loader, device=self.device, buffer_size=2)
            training_loss = 0.0
            validation_loss = 0.0
            ponderar_losses = 0.0
            self.optimizer.zero_grad()
            # Voltamos ao modo treino
            self.train()
            for j, (X, y) in enumerate(train_manager):
                # Precisamos resetar o hidden state do LSTM a cada batch, ou
                # ocorre erro no backward(). O tamanho do batch para a cell eh
                # simplesmente o tamanho do batch em y ou X (tanto faz).
                self.hidden_cell = (torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device),
                                    torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device))

                with autocast(enabled=self.use_amp):
                    y_pred = self(X)
                    # O peso do batch no calculo da loss eh proporcional ao seu
                    # tamanho.
                    single_loss = self.loss_function(y_pred, y) * X.shape[0] / self.training_batch_size
                # Cada chamada ao backprop eh ACUMULADA no gradiente (optimizer)
                scaler.scale(single_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                # We need detach() to no accumulate gradient. Otherwise, memory
                # overflow will happen.
                # We divide by the size of batch because we dont need to
                # compensate batch size when estimating average training loss,
                # otherwise we woul get an explosive and incorrect loss value.
                training_loss += single_loss.detach()

                ponderar_losses += X.shape[0]

            # Tira a media das losses.
            training_loss = training_loss * self.training_batch_size / ponderar_losses

            ponderar_losses = 0.0

            # Nao precisamos perder tempo calculando gradientes da loss durante
            # a validacao
            with torch.no_grad():
                # validando o modelo no modo de evaluation
                self.eval()
                for j, (X, y) in enumerate(val_manager):
                    # Precisamos resetar o hidden state do LSTM a cada batch, ou
                    # ocorre erro no backward(). O tamanho do batch para a cell eh
                    # simplesmente o tamanho do batch em y ou X (tanto faz).
                    self.hidden_cell = (torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device),
                                        torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device))

                    with autocast(enabled=self.use_amp):
                        y_pred = self(X)
                        single_loss = self.loss_function(y_pred, y) * X.shape[0] / self.training_batch_size

                    validation_loss += single_loss.detach()

                    ponderar_losses += X.shape[0]

            # Tira a media ponderada das losses.
            validation_loss = validation_loss * self.training_batch_size / ponderar_losses

            # Checkpoint to best models found.
            if best_validation_loss > validation_loss:
                # Update the new best loss.
                best_validation_loss = validation_loss
                self.eval()
                # torch.save(self, "{:.15f}".format(best_validation_loss) + "_checkpoint.pth")
                torch.save(self, "best_model.pth")
                torch.save(self.state_dict(), "best_model_state_dict.pth")

            tqdm_bar.set_description(f'epoch: {i:1} train_loss: {training_loss.item():10.10f}' + f' val_loss: {validation_loss.item():10.10f}')
            w.writerow([i, training_loss.item(), validation_loss.item()])
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

    def load_feature_extractor(self, freeze_pretrained_model=True):
        """
Here you may load a pretrained InertialModule to make predictions. By default,
it freezes all InertialModule layers.

        :param freeze_pretrained_model: Whenever to freeze pretrained InertialModule layers. Default = True.
        :return: Void.
        """
        model = torch.load("best_model.pth")
        # model.load_state_dict(torch.load("best_model_state_dict.pth"))

        self.feature_extractor = model.feature_extractor
        # Aproveitamos para carregar tambem a camada densa de previsao de uma vez
        self.adaptive_pooling = model.adaptive_pooling
        self.dense_network = model.dense_network

        if freeze_pretrained_model is True:
            # Congela todas as camadas do extrator para treinar apenas as camadas
            # seguintes
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            # Estamos congelndo tambem a camada de previsao, apenas para aproveitar a chamada da funcao
            for param in self.dense_network.parameters():
                param.requires_grad = False

        return

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


def experiment():
    """
Runs the experiment itself.

    :return: Trained model.
    """

    # Recebe os arquivos do dataset e o aloca de no formato (numpy npz) adequado.
    # X, y = format_dataset(dataset_directory="dataset-room2_512_16", file_format="NPZ")
    # join_npz_files(files_origin_path="./tmp_x", output_file="./x_data.npz")
    # join_npz_files(files_origin_path="./tmp_y", output_file="./y_data.npz")
    # return

    model = InertialModule(input_size=4, hidden_layer_size=64, n_lstm_units=3, bidirectional=True,
                           output_size=7, training_batch_size=1024, epochs=50, device=device, validation_percent=0.2)

    # Carrega o extrator de features convolucional pretreinado e congela (grad)
    # model.load_feature_extractor()

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
    # cv_search = \
    #     BayesSearchCV(estimator=regressor, cv=splitter,
    #                   search_spaces=parametros,
    #                   refit=True,
    #                   n_iter=4,
    #                   verbose=1,
    #                   # n_jobs=4,
    #                   scoring=make_scorer(mean_squared_error,
    #                                       greater_is_better=False,
    #                                       needs_proba=False))

    # Let's go fit! Comment if only loading pretrained model.
    # model.fit(X, y)
    model.fit()

    model.eval()
    # ===========PREDICAO-["px", "py", "pz", "qw", "qx", "qy", "qz"]============
    room2_tum_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-room2_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room2_512_16/mav0/mocap0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, device=device, convert_first=True)

    predict = []
    reference = []
    for i, (x, y) in tqdm(enumerate(room2_tum_dataset), total=len(room2_tum_dataset)):
        y_hat = model(x.view(1, x.shape[0], x.shape[1])).view(-1)
        predict.append(y_hat.detach().cpu().numpy())
        reference.append(y.detach().cpu().numpy())

    predict = array(predict)
    reference = array(reference)

    dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    for i, dim_name in enumerate(dimensoes):
        plt.close()
        plt.plot(arange(predict.shape[0]), predict[:, i], arange(reference.shape[0]), reference[:, i])
        plt.legend(['predict', 'reference'], loc='upper right')
        plt.savefig(dim_name + ".png", dpi=200)
        # plt.show()

    # ===========FIM-DE-PREDICAO-["px", "py", "pz", "qw", "qx", "qy", "qz"]=====

    print(model)

    return model


if __name__ == '__main__':

    # plot_csv()

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Usando GPU")
    else:
        dev = "cpu"
        print("Usando CPU")
    device = torch.device(dev)

    experiment()
