import csv
import time
from threading import Thread, Event

import torch
import zmq
from numpy import arange, array
from numpy.linalg import norm
from skimage.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from torch import nn, movedim, absolute, tensor
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Sequential, Conv1d, GaussianNLLLoss, MSELoss
from torch.optim import lr_scheduler
from torch.utils.data import Subset, ConcatDataset
from tqdm import tqdm

from mydatasets import *
from ptk.utils import DataManager
from ptk.utils.torchtools import axis_angle_into_rotation_matrix, axis_angle_into_quaternion, rotation_matrix_into_axis_angle

# When importing every models from this module, make sure only models are
# imported
__all__ = ["InertialModule", "IMUHandler", "ResBlock", "SumLayer",
           "IMUHandlerWithPreintegration", "PreintegrationModule",
           "EachSamplePreintegrationModule", "SignalEnvelope", "SignalWavelet"]

from pytorch_wavelets.dwt.transform1d import DWT1DForward
from mydatasets import ParallelBatchTimeseriesDataset
from losses import *


class SignalWavelet(nn.Module):
    def __init__(self, wave='db6', j=3):
        """
    Discrete 1D Wavelet transform described in:
    https://github.com/fbcotter/pytorch_wavelets

        :param wave: Wavelet type.
        :param j: Number of levels of decomposition.
        """
        super().__init__()
        self.dwt = DWT1DForward(wave=wave, J=j)

    def forward(self, X):
        # conv1d-like: (N, C, L)
        # X = torch.randn(10, 5, 100)

        # yl, yh = dwt(X)
        return self.dwt(X)


class SignalEnvelope(nn.Module):
    def __init__(self, n_channels=2, kernel_size_envelope=9, kernel_size_avg=20, norm_order=9):
        """
    Returns original signal, moving average, upper and lower envelope, respective.
    For each channel, those metrics are calculated invidually.
        :param n_channels: Number of input (and output) channels.
        :param kernel_size_envelope: How many samples use to compute envelopes.
        :param kernel_size_avg: How many samples from original signal to compute average mean.
        :param norm_order: Number of norm order to apply when calculating envelope.
        """
        super(SignalEnvelope, self).__init__()

        self.norm_order = norm_order

        # Usaremos convolucoes 1D para calcular medias locais (media movel).
        # Nao eh propriamente uma convolucao e seus parametros ficam congelados,
        # mas eh uma layer que computa de forma rapida e direta o que queremos

        # garante que o kernel seja impar
        kernel_size = (kernel_size_envelope // 2) * 2 + 1
        self.envelope_layer = Conv1d(n_channels, n_channels, (kernel_size,), padding=(kernel_size // 2,), padding_mode="replicate", bias=False).requires_grad_(False)
        # Os unicas sinapses valendo 1 serao aquelas que ligam o sinal de entrada ao seu respectivo envelope
        self.envelope_layer.weight[:, :, :] = torch.zeros(n_channels, n_channels, kernel_size, )
        for i in range(n_channels):
            self.envelope_layer.weight[i, i, :] = torch.ones(kernel_size, )

        # garante que o kernel seja impar
        kernel_size = (kernel_size_avg // 2) * 2 + 1
        self.avg_layer = Conv1d(n_channels, n_channels, (kernel_size,), padding=(kernel_size // 2,), padding_mode="replicate", bias=False).requires_grad_(False)
        # Os unicas sinapses valendo 1 serao aquelas que ligam o sinal de entrada a sua media movel
        self.avg_layer.weight[:, :, :] = torch.zeros(n_channels, n_channels, kernel_size, )
        for i in range(n_channels):
            self.avg_layer.weight[i, i, :] = torch.ones(kernel_size, ) / kernel_size

    def forward(self, signal):
        # Calculamos a media movel do sinal. A media movel eh uma curva com
        # tendencia central em nossa distribuicao, enquanto que a media
        # aritmetica seria apenas 1 numero constante, nao atendendo ao nosso
        # objetivo.
        # Repare que a convolucaocalcula perfeitamente a media movel de cada
        # canal, com os valores fixos que colocamos em seus weights.
        media_movel_sinal = self.avg_layer(signal)

        # cat(original_signal, media_movel, upper_envelope, lower_envelope)
        return \
            torch.cat(
                (
                    signal, media_movel_sinal,
                    # Os envelopes sao basicamente uma media tirada sobre a
                    # soma do ABSOLUTO de CADA desvio (o ponto atual menos a
                    # media movel) com a propria curva de media movel.
                    # A diferenca aqui eh que, ao inves de tiramos apenas a
                    # media, elevamos cada desvio local a uma potencia
                    # (norm_order), somamos os desvios locais, e tiramos a raiz
                    # de MESMA ordem. Eh como se calculassemos o modulo ao inves
                    # da media, pois da melhores resultados para o envelope.
                    # Consideramos que da melhores resultados porque um modulo
                    # de ordem infinita basicamente retorna o valor de seu
                    # membro mais alto, ou seja, o retorno deste valor seria o
                    # do pico ou vale mais extremo naquele ponto da curva,
                    # garantindo que o envelope esteja sempre sobre a silhueta
                    # da curva.
                    (media_movel_sinal + self.envelope_layer((torch.abs(signal - media_movel_sinal)) ** self.norm_order) ** (1.0 / self.norm_order)),
                    (media_movel_sinal - self.envelope_layer((torch.abs(signal - media_movel_sinal)) ** self.norm_order) ** (1.0 / self.norm_order))
                ),
                dim=1)


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
                nn.LeakyReLU(),
                nn.BatchNorm1d(n_output_channels),
                Conv1d(n_output_channels, n_output_channels, kernel_size,
                       stride, kernel_size[0] // 2 * dilation + padding,
                       dilation, groups, bias, padding_mode),
                nn.PReLU(num_parameters=n_output_channels, init=0.01),
                nn.BatchNorm1d(n_output_channels)
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


class SqueezeAndExcitationBlock1D(nn.Module):
    def __init__(self, n_channels):
        """
    Classical implementation of Squeeze and Excitation Networks
    from https://arxiv.org/pdf/1709.01507.pdf .
    This block provides attention mechanism for Conv1D layers output.

        :param n_channels: Number of channels to receive from Conv1D.
        """
        super().__init__()

        self.n_channels = n_channels

        self.squeeze_and_excitation = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels, n_channels),
            nn.LeakyReLU(),
            nn.Linear(n_channels, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, input_seq):
        # We flattened the input, so we resize it back to shape after global
        # pooling and multiply the original input sequence.
        return input_seq * self.squeeze_and_excitation(input_seq).view(-1, self.n_channels, 1)


class EachSamplePreintegrationModule(nn.Module):
    def __init__(self, device=torch.device("cpu"), dtype=torch.float32, imu_freq=200):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.delta_t = 1 / imu_freq

        self.identity_matrix = torch.eye(n=3, m=3, device=device, dtype=dtype)

    def forward(self, input_seq):
        """
    This method computes delta R, v and p (orientation, velocity and position),
    according to https://arxiv.org/abs/2101.07061 and
    https://arxiv.org/abs/1512.02363 about inertial feature preintegration.

        :param a_samples: IMU accelerometer input samples to compute over.
        :param w_samples: IMU gyroscope input samples to compute over.
        :param initial_velocity: Whenever using a statefull approach, passing
        initial_velocity will bring up better preintegration results.
        :return: R (converted from matriz into quaternion), v and p (both 3D tensors).
        """
        # orientation matrix
        delta_r = self.identity_matrix.clone()
        # velocity tensor (3 element)
        delta_v = 0.0
        # position tensor (3 element)
        delta_p = 0.0

        deltas_p = torch.zeros((input_seq.shape[0], input_seq.shape[1], 3, 1), device=self.device, dtype=self.dtype)
        deltas_v = torch.zeros((input_seq.shape[0], input_seq.shape[1], 3, 1), device=self.device, dtype=self.dtype)
        deltas_q = torch.zeros((input_seq.shape[0], input_seq.shape[1], 4), device=self.device, dtype=self.dtype)

        # avoid dividing delta_t by 2 on every loop iteration
        delta_t_divided_by_2 = self.delta_t / 2
        square_delta_t_divided_by_2 = delta_t_divided_by_2 * self.delta_t

        w = input_seq.movedim(1, 0)[:, :, :3]
        a = input_seq.movedim(1, 0)[:, :, 3:].unsqueeze(3)
        # interactive productory and summation steps
        for i, (w_k, a_k) in enumerate(list(zip(w, a))):
            delta_r = torch.matmul(delta_r, axis_angle_into_rotation_matrix(w_k, self.delta_t, device=self.device, dtype=self.dtype))
            delta_v += torch.matmul(delta_r, a_k * self.delta_t)
            # Slightly different from original paper, now including
            # initial_velocity (if available) to compute CURRENT velocity, not
            # only delta_v (variation)
            delta_p += delta_v * self.delta_t + torch.matmul(delta_r, a_k * square_delta_t_divided_by_2)

            deltas_p[:, i, :, 0:] = delta_p
            deltas_v[:, i, :, 0:] = delta_v
            deltas_q[:, i, :] = \
                axis_angle_into_quaternion(
                    *rotation_matrix_into_axis_angle(
                        delta_r, device=self.device, dtype=self.dtype
                    ), device=self.device, dtype=self.dtype
                )

        # noinspection PyTypeChecker
        return torch.cat((input_seq, deltas_p.squeeze(3), deltas_q,
                          deltas_v.squeeze(3)), dim=2)


class LSTMLatentFeatures(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, n_lstm_units=1, bidirectional=False):
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
        self.n_lstm_units = n_lstm_units
        if bidirectional:
            self.bidirectional = 1
            self.num_directions = 2
        else:
            self.bidirectional = 0
            self.num_directions = 1

        self.lstm = nn.LSTM(input_size, self.hidden_layer_size,
                            batch_first=True, num_layers=self.n_lstm_units,
                            bidirectional=bool(self.bidirectional),
                            dropout=0.5)

        self.dense_network = Sequential(
            nn.Linear(self.num_directions * self.hidden_layer_size, 128),
            nn.PReLU(num_parameters=128, init=0.1),
            nn.Linear(128, self.output_size)
        )
        # We train using multiple inputs (mini_batch), so we let this cell ready
        # to be called.
        # self.hidden_cell_zeros = (torch.zeros((self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size), device=self.device),
        #                           torch.zeros((self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size), device=self.device))
        self.hidden_cell_zeros = None

        self.hidden_cell_output = None

        return

    def forward(self, input_seq):
        """
Classic forward method of every PyTorch model, as fast as possible. Receives an
input sequence and returns the prediction for the final step.

        :param input_seq: Input sequence of the time series.
        :return: The prediction in the end of the series.
        """

        # (seq_len, batch, input_size), mas pode inverter o
        # batch com o seq_len se fizer batch_first==1 na criacao do LSTM
        lstm_out, self.hidden_cell_output = self.lstm(input_seq, self.hidden_cell_zeros)

        # All batch size, whatever sequence length, forward direction and
        # lstm output size (hidden size).
        return lstm_out.view(input_seq.shape[0], -1, self.num_directions * self.hidden_layer_size)


class _MoveDimModule(nn.Module):
    def __init__(self, source=-2, destination=-1):
        super().__init__()
        self.source = source
        self.destination = destination

    def forward(self, input_seq):
        return movedim(input_seq, self.source, self.destination)


class ConvLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, n_lstm_units=1, bidirectional=False):
        """
Receives a multi channel signal, calculate its envelopes, then calculate its
wavelets and finally uses LSTMs to compress each Wavelet and original signal
channel into an individual state representation.

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
        self.n_lstm_units = n_lstm_units
        if bidirectional:
            self.bidirectional = 1
            self.num_directions = 2
        else:
            self.bidirectional = 0
            self.num_directions = 1

        n_base_filters = 8
        n_output_features = 32
        self.feature_extractor = Sequential(
            SignalEnvelope(n_channels=input_size),
            Conv1d(4 * input_size, 1 * n_base_filters, (3,), dilation=(2,), stride=(1,)), nn.PReLU(num_parameters=1 * n_base_filters), nn.BatchNorm1d(1 * n_base_filters),
            Conv1d(1 * n_base_filters, 2 * n_base_filters, (3,), dilation=(2,), stride=(1,)), nn.PReLU(num_parameters=2 * n_base_filters), nn.BatchNorm1d(2 * n_base_filters),
            Conv1d(2 * n_base_filters, n_output_features, (3,), dilation=(2,), stride=(1,)), nn.PReLU(num_parameters=n_output_features), nn.BatchNorm1d(n_output_features),
            _MoveDimModule(source=-2, destination=-1),
            nn.LSTM(n_output_features, self.hidden_layer_size,
                    batch_first=True, num_layers=self.n_lstm_units,
                    bidirectional=bool(self.bidirectional),
                    )
        )

        self.dense_network = Sequential(
            nn.Linear(self.num_directions * self.hidden_layer_size, 32),
            nn.PReLU(num_parameters=32),
            nn.Linear(32, 16),
            nn.PReLU(num_parameters=16),
            nn.Linear(16, self.output_size)
        )

        return

    def forward(self, input_seq):

        lstm_out, _ = self.feature_extractor(input_seq.movedim(-2, -1))

        # All batch size, whatever sequence length, forward direction and
        # lstm output size (hidden size).
        return lstm_out[:, -1, :].flatten(start_dim=1)


class Conv1DFeatureExtractor(nn.Module):
    def __init__(self, input_size=1, output_size=8):
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

        # Half of the outputs are summed and the other half is Avg pooling.
        self.output_size = output_size

        # ATTENTION: You cannot change this anymore, since we added a sum layer
        # and it casts conv outputs to 1 feature per channel
        pooling_output_size = 1

        n_base_filters = 128
        n_output_features = 6 * 128
        self.feature_extractor = \
            Sequential(
                #
                # Conv1d(input_size, 1 * n_base_filters, (7,), dilation=(1,), stride=(1,)), nn.PReLU(), nn.BatchNorm1d(1 * n_base_filters, affine=True),
                # ResBlock(1 * n_base_filters, 2 * n_base_filters, (7,), dilation=1, stride=1),
                # ResBlock(2 * n_base_filters, 4 * n_base_filters, (7,), dilation=1, stride=1),
                # ResBlock(4 * n_base_filters, n_output_features, (7,), dilation=1, stride=1),
                nn.BatchNorm1d(input_size, affine=False),
                # SignalEnvelope(n_channels=input_size),
                Conv1d(1 * input_size, 1 * n_base_filters, (3,), dilation=(2,), stride=(1,)), nn.LeakyReLU(), nn.BatchNorm1d(1 * n_base_filters),
                Conv1d(1 * n_base_filters, 2 * n_base_filters, (3,), dilation=(2,), stride=(1,)), nn.LeakyReLU(), nn.BatchNorm1d(2 * n_base_filters),
                # nn.Dropout2d(p=0.5),
                Conv1d(2 * n_base_filters, 3 * n_base_filters, (3,), dilation=(2,), stride=(1,)), nn.LeakyReLU(), nn.BatchNorm1d(3 * n_base_filters),
                # nn.Dropout2d(p=0.5),
                # Conv1d(3 * n_base_filters, 4 * n_base_filters, (3,), dilation=(2,), stride=(1,)), nn.LeakyReLU(),  # nn.BatchNorm1d(4 * n_base_filters),
                Conv1d(3 * n_base_filters, 4 * n_base_filters, (3,), dilation=(2,), stride=(1,)), nn.PReLU(num_parameters=4 * n_base_filters), nn.BatchNorm1d(4 * n_base_filters),
                # SqueezeAndExcitationBlock1D(3 * n_base_filters),
                Conv1d(4 * n_base_filters, 5 * n_base_filters, (3,), dilation=(2,), stride=(1,)), nn.PReLU(num_parameters=5 * n_base_filters), nn.BatchNorm1d(5 * n_base_filters),
                # SqueezeAndExcitationBlock1D(2 * n_base_filters),
                Conv1d(5 * n_base_filters, n_output_features, (3,), dilation=(2,), stride=(1,)), nn.PReLU(num_parameters=n_output_features), nn.BatchNorm1d(n_output_features),
                # SqueezeAndExcitationBlock1D(n_output_features)
            )

        self.sum_layer = SumLayer(n_output_features)
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(pooling_output_size)

        self.dense_network = Sequential(
            nn.Linear(2 * pooling_output_size * n_output_features, 4096), nn.PReLU(num_parameters=4096),
            nn.BatchNorm1d(4096, affine=True),
            # nn.BatchNorm1d(128, affine=True),  # nn.Dropout(p=0.5),
            # # nn.Linear(16, 16), nn.Tanh(),
            # # nn.BatchNorm1d(16, affine=True),
            nn.Linear(4096, self.output_size)
        )
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

        return torch.cat((
            self.sum_layer(input_seq),
            self.adaptive_pooling(input_seq),
        ), dim=1)


class LSTMReceivingWavelet(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, n_lstm_units=1, bidirectional=False):
        """
Receives a multi channel signal, calculate its envelopes, then calculate its
wavelets and finally uses LSTMs to compress each Wavelet and original signal
channel into an individual state representation.

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
        self.n_lstm_units = n_lstm_units
        if bidirectional:
            self.bidirectional = 1
            self.num_directions = 2
        else:
            self.bidirectional = 0
            self.num_directions = 1

        self.envelope = SignalEnvelope(n_channels=input_size, norm_order=5)
        self.wavelet = SignalWavelet()

        # Each LSTM will compress the sequence of different sizes sequences
        # into an individual state representation.
        self.lstm_imu = nn.LSTM(input_size, self.hidden_layer_size,
                                batch_first=True, num_layers=self.n_lstm_units,
                                bidirectional=bool(self.bidirectional),
                                dropout=0.5)

        self.lstm_yl = nn.LSTM(4 * input_size, self.hidden_layer_size,
                               batch_first=True, num_layers=self.n_lstm_units,
                               bidirectional=bool(self.bidirectional),
                               dropout=0.5)

        self.lstm_yh_0 = nn.LSTM(4 * input_size, self.hidden_layer_size,
                                 batch_first=True, num_layers=self.n_lstm_units,
                                 bidirectional=bool(self.bidirectional),
                                 dropout=0.5)

        self.lstm_yh_1 = nn.LSTM(4 * input_size, self.hidden_layer_size,
                                 batch_first=True, num_layers=self.n_lstm_units,
                                 bidirectional=bool(self.bidirectional),
                                 dropout=0.5)

        self.lstm_yh_2 = nn.LSTM(4 * input_size, self.hidden_layer_size,
                                 batch_first=True, num_layers=self.n_lstm_units,
                                 bidirectional=bool(self.bidirectional),
                                 dropout=0.5)

        self.dense_network = Sequential(
            nn.Linear(self.num_directions * self.hidden_layer_size * 5, 32),
            nn.PReLU(num_parameters=32, init=0.1),
            nn.Linear(32, 16),
            nn.PReLU(num_parameters=16, init=0.1),
            nn.Linear(16, self.output_size)
        )
        # We train using multiple inputs (mini_batch), so we let this cell ready
        # to be called.
        # self.hidden_cell_zeros = (torch.zeros((self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size), device=self.device),
        #                           torch.zeros((self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size), device=self.device))
        self.hidden_cell_zeros = None

        self.hidden_cell_output = None

        return

    def forward(self, input_seq):
        """
Classic forward method of every PyTorch model, as fast as possible. Receives an
input sequence and returns the prediction for the final step.

        :param input_seq: Input sequence of the time series.
        :return: The prediction in the end of the series.
        """

        yl, yh = self.wavelet(self.envelope(input_seq.movedim(1, 2)))

        # (seq_len, batch, input_size), mas pode inverter o
        # batch com o seq_len se fizer batch_first==1 na criacao do LSTM
        # lstm_imu_out, _ = self.lstm_imu(input_seq)
        # lstm_yl_out, _ = self.lstm_yl(yl)
        # lstm_yh_0_out, _ = self.lstm_yh_0(yh[0])
        # lstm_yh_1_out, _ = self.lstm_yh_1(yh[1])
        # lstm_yh_2_out, _ = self.lstm_yh_2(yh[2])

        return torch.cat((self.lstm_imu(input_seq)[0][:, -1, :],
                          self.lstm_yl(yl.movedim(1, 2))[0][:, -1, :],
                          self.lstm_yh_0(yh[0].movedim(1, 2))[0][:, -1, :],
                          self.lstm_yh_1(yh[1].movedim(1, 2))[0][:, -1, :],
                          self.lstm_yh_2(yh[2].movedim(1, 2))[0][:, -1, :]),
                         dim=1)


class PreintegrationModule(nn.Module):
    def __init__(self, device=torch.device("cpu"), dtype=torch.float32, imu_freq=200):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.delta_t = 1 / imu_freq

        self.identity_matrix = torch.eye(n=3, m=3, device=device, dtype=dtype)

    def forward(self, input_seq):
        """
    This method computes delta R, v and p (orientation, velocity and position),
    according to https://arxiv.org/abs/2101.07061 and
    https://arxiv.org/abs/1512.02363 about inertial feature preintegration.

        :param a_samples: IMU accelerometer input samples to compute over.
        :param w_samples: IMU gyroscope input samples to compute over.
        :param initial_velocity: Whenever using a statefull approach, passing
        initial_velocity will bring up better preintegration results.
        :return: R (converted from matriz into quaternion), v and p (both 3D tensors).
        """
        # orientation matrix
        delta_r = self.identity_matrix.clone()
        # velocity tensor (3 element)
        delta_v = 0.0
        # position tensor (3 element)
        delta_p = 0.0

        # avoid dividing delta_t by 2 on every loop iteration
        delta_t_divided_by_2 = self.delta_t / 2
        square_delta_t_divided_by_2 = delta_t_divided_by_2 * self.delta_t

        w = input_seq.movedim(1, 0)[:, :, :3]
        a = input_seq.movedim(1, 0)[:, :, 3:].unsqueeze(3)
        # interactive productory and summation steps
        for w_k, a_k in zip(w, a):
            delta_r = torch.matmul(delta_r, axis_angle_into_rotation_matrix(w_k, self.delta_t, device=self.device, dtype=self.dtype))
            delta_v += torch.matmul(delta_r, a_k * self.delta_t)
            # Slightly different from original paper, now including
            # initial_velocity (if available) to compute CURRENT velocity, not
            # only delta_v (variation)
            delta_p += delta_v * self.delta_t + torch.matmul(delta_r, a_k * square_delta_t_divided_by_2)

        # noinspection PyTypeChecker
        return torch.cat((delta_p.squeeze(2), axis_angle_into_quaternion(*rotation_matrix_into_axis_angle(delta_r, device=self.device, dtype=self.dtype), device=self.device, dtype=self.dtype)),
                         dim=1)  # , delta_v


class SingleSamplePreintegrationModule(nn.Module):
    def __init__(self, device=torch.device("cpu"), dtype=torch.float32, imu_freq=200):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.delta_t = 1 / imu_freq

        self.identity_matrix = torch.eye(n=3, m=3, device=device, dtype=dtype)

    def forward(self, input_seq):
        """
    This method computes delta R, v and p (orientation, velocity and position),
    according to https://arxiv.org/abs/2101.07061 and
    https://arxiv.org/abs/1512.02363 about inertial feature preintegration.

        :param a_samples: IMU accelerometer input samples to compute over.
        :param w_samples: IMU gyroscope input samples to compute over.
        :param initial_velocity: Whenever using a statefull approach, passing
        initial_velocity will bring up better preintegration results.
        :return: R (converted from matriz into quaternion), v and p (both 3D tensors).
        """

        # w, a
        # input_seq[0, :, :3], input_seq[0, :, 3:]

        # single sample productory and summation steps
        delta_r = axis_angle_into_rotation_matrix(input_seq[0, :, :3], self.delta_t, device=self.device, dtype=self.dtype)
        delta_v = torch.matmul(delta_r, input_seq[0, :, 3:] * self.delta_t)
        delta_p = delta_v * self.delta_t + torch.matmul(delta_r, input_seq[0, :, 3:] * (self.delta_t ** 2) / 2)

        # noinspection PyTypeChecker
        return torch.cat((delta_p, axis_angle_into_quaternion(*rotation_matrix_into_axis_angle(delta_r, device=self.device, dtype=self.dtype), device=self.device, dtype=self.dtype), delta_v),
                         dim=-1)


class InertialModule(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, n_lstm_units=1, epochs=150, training_batch_size=64, validation_percent=0.2, bidirectional=False,
                 device=torch.device("cpu"),
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
        self.creation_time = time.time()
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

        print("creation_time: ", self.creation_time)

        # self.preintegration_module = \
        #     EachSamplePreintegrationModule(device=self.device)
        #
        # # ATTENTION: You cannot change this anymore, since we added a sum layer
        # # and it casts conv outputs to 1 feature per channel
        # pooling_output_size = 1
        #
        # n_base_filters = 8
        # n_output_features = 8
        # self.feature_extractor = \
        #     LSTMLatentFeatures(input_size=input_size,
        #                        hidden_layer_size=hidden_layer_size,
        #                        output_size=output_size,
        #                        n_lstm_units=n_lstm_units,
        #                        bidirectional=bidirectional)

        # self.feature_extractor = \
        #     ConvLSTM(input_size=input_size,
        #              hidden_layer_size=hidden_layer_size,
        #              output_size=output_size,
        #              n_lstm_units=n_lstm_units,
        #              bidirectional=bidirectional)

        self.feature_extractor = Conv1DFeatureExtractor(input_size=input_size,
                                                        output_size=output_size)

        # Assim nao precisamos adaptar a rede densa a uma saida de CNN ou LSTM,
        # ja pegamos a rede adaptada do proprio extrator de
        # features (seja lstm ou CNN)
        self.dense_network = self.feature_extractor.dense_network

        return

    def forward(self, input_seq):
        """
Classic forward method of every PyTorch model, as fast as possible. Receives an
input sequence and returns the prediction for the final step.

        :param input_seq: Input sequence of the time series.
        :return: The prediction in the end of the series.
        """

        # input_seq = self.preintegration_module(input_seq)

        # # As features (px, py, pz, qw, qx, qy, qz) sao os "canais" da
        # # convolucao e precisam vir no meio para o pytorch
        # input_seq = movedim(input_seq, -2, -1)

        output_seq = self.feature_extractor(input_seq)

        predictions = self.dense_network(output_seq.flatten(start_dim=1))

        # output_seq = self.feature_extractor(input_seq)
        #
        # # All batch size, whatever sequence length, forward direction and
        # # lstm output size (hidden size).
        # # We only want the last output of lstm (end of sequence), that is
        # # the reason of '[:,-1,:]'.
        # output_seq = output_seq.view(output_seq.shape[0], -1, self.num_directions * self.hidden_layer_size)[:, -1, :]
        #
        # predictions = self.dense_network(output_seq)

        pos = predictions[:, 0: 3]
        quat = torch.nn.functional.normalize(predictions[:, 3:7])
        var = predictions[:, 7:]

        return torch.cat((pos, quat, var), 1)

    def fit(self):
        """
This method contains the customized script for training this estimator. Data is
obtained with PyTorch's dataset and dataloader classes for memory efficiency
when dealing with big datasets. Otherwise loading the whole dataset would
overflow the memory.

        :return: Trained model with best validation loss found (it uses checkpoint).
        """
        print("creation_time: ", self.creation_time)
        self.train()
        self.packing_sequence = True
        self.to(self.device)
        # =====DATA-PREPARATION=================================================
        euroc_v1_01_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V1_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                             n_threads=2,
                                                             min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)

        # Esse daqui gera NAN no treino e na validacao, melhor nao usar
        euroc_v2_01_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V2_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                             n_threads=2,
                                                             min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_v2_02_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V2_02_medium/mav0/imu0/data.csv",
                                                             y_csv_path="dataset-files/V2_02_medium/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                             min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_v2_03_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V2_03_difficult/mav0/imu0/data.csv",
                                                             y_csv_path="dataset-files/V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                             min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_v1_02_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V1_02_medium/mav0/imu0/data.csv",
                                                             y_csv_path="dataset-files/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                             min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_v1_03_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V1_03_difficult/mav0/imu0/data.csv",
                                                             y_csv_path="dataset-files/V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                             min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_mh1_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/MH_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                           n_threads=2,
                                                           min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_mh2_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/MH_02_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                           n_threads=2,
                                                           min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_mh3_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/MH_03_medium/mav0/imu0/data.csv",
                                                           y_csv_path="dataset-files/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                           min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_mh4_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/MH_04_difficult/mav0/imu0/data.csv",
                                                           y_csv_path="dataset-files/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                           min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_mh5_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/MH_05_difficult/mav0/imu0/data.csv",
                                                           y_csv_path="dataset-files/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                           min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)

        # room1_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room1_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room1_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=40, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)
        #
        # room2_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room2_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room2_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=40, max_window_size=100, batch_size=self.training_batch_size, shuffle=False)
        #
        # room3_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room3_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room3_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False, noise=None)
        #
        # room4_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room4_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room4_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=20, max_window_size=100, batch_size=self.training_batch_size, shuffle=False)
        #
        # room5_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room5_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room5_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=150, max_window_size=200, batch_size=self.training_batch_size, shuffle=False, noise=None)
        #
        # room6_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room6_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room6_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=150, max_window_size=200, batch_size=self.training_batch_size, shuffle=False)

        # # Diminuir o dataset para verificar o funcionamento de scripts
        # room1_tum_dataset = Subset(room1_tum_dataset, arange(int(len(room1_tum_dataset) * 0.001)))

        # train_dataset = Subset(room1_tum_dataset, arange(int(len(room1_tum_dataset) * self.train_percentage)))
        # val_dataset = Subset(room1_tum_dataset, arange(int(len(room1_tum_dataset) * self.train_percentage), len(room1_tum_dataset)))

        train_dataset = ConcatDataset([euroc_v1_01_dataset, euroc_v1_02_dataset,
                                       euroc_mh1_dataset, euroc_mh5_dataset,
                                       euroc_v2_03_dataset, euroc_mh4_dataset])
        val_dataset = ConcatDataset([euroc_v2_02_dataset, euroc_mh3_dataset])

        # train_dataset = ConcatDataset([euroc_v1_01_dataset, ])
        # val_dataset = ConcatDataset([euroc_mh3_dataset, ])

        train_loader = CustomDataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4, multiprocessing_context='spawn')
        val_loader = CustomDataLoader(dataset=val_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4, multiprocessing_context='spawn')

        # Pesos das saídas: px py pz wq wx wy wz (translacoes e quaternios).
        weights = array([1000, 1000, 1000, 1, 1, 1, 1])
        weights = tensor(weights / norm(weights)).to(device=self.device)

        # train_loader = PackingSequenceDataloader(train_dataset, batch_size=128, shuffle=True)
        # val_loader = PackingSequenceDataloader(val_dataset, batch_size=128, shuffle=True)
        # =====fim-DATA-PREPARATION=============================================

        epochs = self.epochs
        best_validation_loss = 999999
        if self.loss_function is None: self.loss_function = PosAndAngleLoss()  # MSELoss()
        if self.optimizer is None: self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1, )  # momentum=0.9, nesterov=True)
        scaler = GradScaler(enabled=self.use_amp)
        scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1, last_epoch=-1)
        # scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1)
        # scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=1.0, steps_per_epoch=len(train_loader), epochs=epochs, final_div_factor=1e3)

        f = open("loss_log.csv", "w")
        w = csv.writer(f)
        w.writerow(["epoch", "training_loss", "val_loss"])

        tqdm_bar = tqdm(range(epochs))
        for i in tqdm_bar:
            # Voltamos ao modo treino
            self.train()
            train_manager = DataManager(train_loader, device=self.device, buffer_size=2)
            val_manager = DataManager(val_loader, device=self.device, buffer_size=2)
            training_loss = 0.0
            validation_loss = 0.0
            ponderar_losses = 0.0
            self.optimizer.zero_grad()
            # t = time.time()
            for j, (X, y) in enumerate(train_manager):
                # # Precisamos resetar o hidden state do LSTM a cada batch, ou
                # # ocorre erro no backward(). O tamanho do batch para a cell eh
                # # simplesmente o tamanho do batch em y ou X (tanto faz).
                # self.feature_extractor.hidden_cell_zeros = (torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device),
                #                                             torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device))
                # print("t carregando dados: ", '{:f}'.format(time.time() - t))
                # t = time.time()
                with autocast(enabled=self.use_amp):
                    y_pred = self(X)
                    var = torch.exp(y_pred[:, self.output_size // 2:])
                    y_pred = y_pred[:, :self.output_size // 2]
                    # O peso do batch no calculo da loss eh proporcional ao seu
                    # tamanho.
                    single_loss = self.loss_function(y_pred, y) * X.shape[0] / 1e6
                # Cada chamada ao backprop eh ACUMULADA no gradiente (optimizer)
                scaler.scale(single_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                # print("t GPU: ", '{:f}'.format(time.time() - t))
                # t = time.time()
                # We need detach() to no accumulate gradient. Otherwise, memory
                # overflow will happen.
                # We divide by the size of batch because we dont need to
                # compensate batch size when estimating average training loss,
                # otherwise we woul get an explosive and incorrect loss value.
                training_loss += single_loss.detach()

                ponderar_losses += X.shape[0] / 1e6

                # # Run lr_scheduler OneCycleLr
                # scheduler.step()

                # print("t operando python: ", '{:f}'.format(time.time() - t))
                # t = time.time()

            # Tira a media ponderada das losses.
            training_loss = training_loss / ponderar_losses

            ponderar_losses = 0.0

            # Nao precisamos perder tempo calculando gradientes da loss durante
            # a validacao
            with torch.no_grad():
                # validando o modelo no modo de evaluation
                self.eval()
                for j, (X, y) in enumerate(val_manager):
                    # # Precisamos resetar o hidden state do LSTM a cada batch, ou
                    # # ocorre erro no backward(). O tamanho do batch para a cell eh
                    # # simplesmente o tamanho do batch em y ou X (tanto faz).
                    # self.feature_extractor.hidden_cell_zeros = (torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device),
                    #                                             torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device))

                    with autocast(enabled=self.use_amp):
                        y_pred = self(X)
                        var = torch.exp(y_pred[:, self.output_size // 2:])
                        y_pred = y_pred[:, :self.output_size // 2]
                        single_loss = self.loss_function(y_pred, y) * X.shape[0] / 1e6

                    validation_loss += single_loss.detach()

                    ponderar_losses += X.shape[0] / 1e6

            # Tira a media ponderada das losses.
            validation_loss = validation_loss / ponderar_losses

            # Run learning rate scheduler. ReduceOnPlateau or ExponentialLR
            # scheduler.step(validation_loss)
            scheduler.step()

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


class IMUHandler(nn.Module, Thread):

    def __init__(self, sampling_window_size=200, imu_input_size=6,
                 position_output_size=7, device=torch.device("cpu"), dtype=torch.float32):
        """
This module receives IMU samples and uses the InertialModule to predict our
current position at real time.

Our estimated position takes into account the estimated position
at sampling_window_size IMU samples before, summed with the displacement
predicted by InertialModule after all those IMU samples. It is an arithmetic
summation of each displacement and doesnt takes into account any estimated
trajectory for the object being tracked.

ATENTION: You need to call load_feature_extractor() after instantiating this
class, or your InertialModule predictor will just be a random output. You also
need to call this object.start() to begin updating positions in another thread.

        :param sampling_window_size: (Integer) How many IMU samples to calculate the displacement. Default = 200.
        :param imu_input_size: (Integer) How many features IMU give to us. Default = 6 (ax, ay, az, wx, wq, wz).
        :param position_output_size: (Integer) How many features we use to represent our position annotation. Default = 7 (px, py, pz, qw, qx, qy, qz).
        """
        nn.Module.__init__(self)
        Thread.__init__(self)
        self.sampling_window_size = sampling_window_size
        self._imu_input_size = imu_input_size
        self._position_output_size = position_output_size
        self.device = device
        self.dtype = dtype

        # Event synchronizer for position updates. We'll only estimate a new
        # position if new IMU samples have arrived.
        self._imu_samples_arrived = Event()

        # Indicate for clients that our position predictions have been updated.
        self.new_predictions_arrived = Event()

        # A control flag ordering this thread to stop.
        self.stop_flag = False

        # Here we are gonna store previous IMU samples and our estimated
        # positions at that time. Also, we need a timestamp info to synchronize
        # them. These buffers will grow as new samples and predictions arrive.
        self.predictions_buffer = torch.zeros((1, 1 + position_output_size,),
                                              dtype=self.dtype,
                                              device=self.device,
                                              requires_grad=False)
        # First dimension will always be 1, because of batch dimension.
        self.imu_buffer = torch.zeros((1, 1, 1 + imu_input_size,),
                                      dtype=self.dtype,
                                      device=self.device,
                                      requires_grad=False)

        # This avoids recalculating these thresholds.
        self._imu_buffer_size_limit = 2 * sampling_window_size
        self._imu_buffer_reduction_threshold = int(1.5 * sampling_window_size)

        # ========DEFAULT-PREDICTOR-FOR-COMPATIBILITY===========================
        # Here we define a default InertialModule component, but your are
        # strongly encouraged to load_feature_extractor(), since the modules
        # defined here are not trained yet.
        pooling_output_size = 100
        n_base_filters = 72
        n_output_features = 200
        self.feature_extractor = \
            Sequential(
                Conv1d(imu_input_size, 1 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(1 * n_base_filters),
                Conv1d(1 * n_base_filters, 2 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(2 * n_base_filters),
                Conv1d(2 * n_base_filters, 3 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(3 * n_base_filters),
                Conv1d(3 * n_base_filters, 4 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(4 * n_base_filters),
                Conv1d(4 * n_base_filters, n_output_features, (7,)), nn.PReLU(), nn.BatchNorm1d(n_output_features)
            )
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(pooling_output_size)
        self.dense_network = Sequential(
            nn.Linear(pooling_output_size * n_output_features, 72), nn.PReLU(), nn.BatchNorm1d(72), nn.Dropout(p=0.5),
            nn.Linear(72, 32), nn.PReLU(),
            nn.Linear(32, position_output_size)
        )
        # ========end-of-DEFAULT-PREDICTOR-FOR-COMPATIBILITY====================

        return

    def forward(self, input_seq):
        """
Classic forward method of every PyTorch model. However,
you are not expected to call this method in this model. We have a thread that
updates our position and returns it to with get_current_position().

        :param input_seq: Input sequence of the time series.
        :return: The prediction in the end of the series.
        """

        # As features (px, py, pz, qw, qx, qy, qz) sao os "canais" da
        # convolucao e precisam vir no meio para o pytorch
        input_seq = movedim(input_seq, -2, -1)

        input_seq = self.feature_extractor(input_seq)
        input_seq = self.adaptive_pooling(input_seq)

        predictions = self.dense_network(input_seq.view(input_seq.shape[0], -1))

        return predictions

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

    def get_current_position(self, ):
        """
Returns latest position update generated by this model. We also receive a
timestamp indicating the time that position was estimated.

        :return: Tuple with (Tensor (1x7) containing latest position update; Respective position time reference).
        """
        if self.predictions_buffer.shape[0] > 3 * self.sampling_window_size:
            self.predictions_buffer = \
                self.predictions_buffer[-2 * self.sampling_window_size:]

        self.new_predictions_arrived.clear()
        # Return the last position update and drop out the associated timestamp.
        return self.predictions_buffer[-1, 1:], self.predictions_buffer[-1, 0]

    def set_initial_position(self, position, timestamp=time.time()):
        """
    This Handler calculates position according to displacement from initial
    position. It gives you absolute position estimatives if the initial
    position is set according to your global reference.

        :param position: Starting position (px, py, pz, qw, qx, qy, qz).
        """

        position = torch.tensor(position, device=self.device,
                                dtype=self.dtype, requires_grad=False)

        timestamp = torch.tensor(timestamp, device=self.device,
                                 dtype=self.dtype, requires_grad=False)

        self.predictions_buffer[0, 1:] = position
        self.predictions_buffer[0, 0:1] = timestamp

        return

    def push_imu_sample_to_buffer(self, sample, timestamp):
        """
Stores a new IMU sample to the buffer, so it will be taken into account in next
predictions.

        :param sample: List or iterable containig 6 channels of IMU sample (ax, ay, az, wx, wq, wz).
        :return: Void.
        """

        if self.imu_buffer.shape[1] > self._imu_buffer_size_limit:
            self.imu_buffer = \
                self.imu_buffer[0:, -self._imu_buffer_reduction_threshold:, :]

        self.imu_buffer = \
            torch.cat((self.imu_buffer,
                       torch.tensor([timestamp] + list(sample),
                                    dtype=self.dtype, requires_grad=False,
                                    device=self.device).view(1, 1, -1)
                       ), dim=1)

        # Informs this thread that new samples have been generated.
        self._imu_samples_arrived.set()

        return

    # This class is not intended to be used as training for the InertialModule
    # first stage, so we disable grad for faster computation.
    @torch.no_grad()
    def run(self):

        # We only can start predicting positions after we have the
        # sampling_window_size IMU samples collected.
        while self.imu_buffer.shape[1] < self.sampling_window_size:
            time.sleep(1)

        # Put pytorch module into evaluation mode (batchnorm and dropout need).
        self.eval()

        # If this thread is not stopped, continue updating position.
        while self.stop_flag is False:
            # Wait for IMU new readings
            self._imu_samples_arrived.wait()

            # We check the timestamp of the last sampling_window_size reading and
            # search for the closest timestamp in prediction buffer.
            prediction_closest_timestamp, prediction_idx = \
                IMUHandler.find_nearest(
                    self.predictions_buffer[:, 0],
                    self.imu_buffer[0, -self.sampling_window_size, 0]
                )

            # We also need the prediction's closest reading, that may not be
            # the one at sampling_window_size before
            imu_closest_timestamp, imu_idx = \
                IMUHandler.find_nearest(
                    self.imu_buffer[0, :, 0],
                    prediction_closest_timestamp
                )

            # We need to now for what time we are calculating position.
            current_timestamp = self.imu_buffer[0:, -1, 0:1]

            # Registers that there are no more NEW imu readings to process.
            self._imu_samples_arrived.clear()

            self.predictions_buffer = \
                torch.cat((
                    self.predictions_buffer,
                    torch.cat((
                        current_timestamp,
                        self.predictions_buffer[prediction_idx, 1:] +
                        self(self.imu_buffer[0:, imu_idx:, 1:])
                    ), dim=1)
                ))

            self.new_predictions_arrived.set()

    @staticmethod
    def find_nearest(tensor_to_search, value):
        """
This method takes 1 tensor as first argument and a value to find the element
in array whose value is the closest. Returns the closest value element and its
index in the original array.

        :param tensor_to_search: Reference tensor.
        :param value: Value to find closest element.
        :return: Tuple (Element value, element index).
        """
        idx = (absolute(tensor_to_search - value)).argmin()
        return tensor_to_search[idx], idx


class IMUHandlerWithPreintegration(nn.Module, Thread):

    def __init__(self, sampling_window_size=200, imu_input_size=6,
                 position_output_size=7, device=torch.device("cpu"),
                 dtype=torch.float32, imu_frequency=200):
        """
This module receives IMU samples and uses the InertialModule to predict our
current position at real time.

Our estimated position takes into account the estimated position
at sampling_window_size IMU samples before, summed with the displacement
predicted by InertialModule after all those IMU samples. It is an arithmetic
summation of each displacement and doesnt takes into account any estimated
trajectory for the object being tracked.

ATENTION: You need to call load_feature_extractor() after instantiating this
class, or your InertialModule predictor will just be a random output. You also
need to call this object.start() to begin updating positions in another thread.

        :param sampling_window_size: (Integer) How many IMU samples to calculate the displacement. Default = 200.
        :param imu_input_size: (Integer) How many features IMU give to us. Default = 6 (ax, ay, az, wx, wq, wz).
        :param position_output_size: (Integer) How many features we use to represent our position annotation. Default = 7 (px, py, pz, qw, qx, qy, qz).
        """
        nn.Module.__init__(self)
        Thread.__init__(self)
        self.sampling_window_size = sampling_window_size
        self._imu_input_size = imu_input_size
        self._position_output_size = position_output_size
        self.device = device
        self.dtype = dtype

        # Event synchronizer for position updates. We'll only estimate a new
        # position if new IMU samples have arrived.
        self._imu_samples_arrived = Event()

        # sample period
        self.delta_t = 1.0 / imu_frequency

        # Auxiliary variable
        self.identity_matrix = torch.eye(n=3, m=3,
                                         dtype=self.dtype,
                                         device=self.device,
                                         requires_grad=False)

        # Indicate for clients that our position predictions have been updated.
        self.new_predictions_arrived = Event()

        # A control flag ordering this thread to stop.
        self.stop_flag = False

        # These buffers stores previous IMU samples and our estimated
        # positions at that time. Also, we need a timestamp info to synchronize
        # them. These buffers will grow as new samples and predictions arrive.
        # The first buffer store translational position history and timestamp
        # where each one was calculated.
        self.position_predictions_buffer = torch.zeros((1, 1 + 3,),
                                                       dtype=self.dtype,
                                                       device=self.device,
                                                       requires_grad=False)

        # Store rotation matrices representing synchronous orientations.
        # Since orientation info are "summed" by multiplying each other,
        # first rotation is an identity matrix (neutral element).
        self.orientation_predictions_buffer = \
            torch.eye(n=3, m=3,
                      dtype=self.dtype,
                      device=self.device,
                      requires_grad=False).view(1, 3, 3)

        # acceleration and angular velocity buffers
        self.a_imu_buffer = torch.zeros((1, 1 + 3,),
                                        dtype=self.dtype,
                                        device=self.device,
                                        requires_grad=False)
        # We save angular velocity already converted into skew-matrices,
        # timestamp sinchronization is done within acceleration buffer.
        self.w_imu_buffer = torch.zeros((1, 3, 3),
                                        dtype=self.dtype,
                                        device=self.device,
                                        requires_grad=False)

        # This avoids recalculating these thresholds.
        self._imu_buffer_size_limit = 2 * sampling_window_size
        self._imu_buffer_reduction_threshold = int(1.5 * sampling_window_size)

        # ========DEFAULT-PREDICTOR-FOR-COMPATIBILITY===========================
        # Here we define a default InertialModule component, but your are
        # strongly encouraged to load_feature_extractor(), since the modules
        # defined here are not trained yet.
        pooling_output_size = 100
        n_base_filters = 72
        n_output_features = 200
        self.feature_extractor = \
            Sequential(
                Conv1d(imu_input_size, 1 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(1 * n_base_filters),
                Conv1d(1 * n_base_filters, 2 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(2 * n_base_filters),
                Conv1d(2 * n_base_filters, 3 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(3 * n_base_filters),
                Conv1d(3 * n_base_filters, 4 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(4 * n_base_filters),
                Conv1d(4 * n_base_filters, n_output_features, (7,)), nn.PReLU(), nn.BatchNorm1d(n_output_features)
            )
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(pooling_output_size)
        self.dense_network = Sequential(
            nn.Linear(pooling_output_size * n_output_features, 72), nn.PReLU(), nn.BatchNorm1d(72), nn.Dropout(p=0.5),
            nn.Linear(72, 32), nn.PReLU(),
            nn.Linear(32, position_output_size)
        )
        # ========end-of-DEFAULT-PREDICTOR-FOR-COMPATIBILITY====================

        return

    def forward(self, a, w, v=0):
        """
Classic forward method of every PyTorch model. However,
you are not expected to call this method in this model. We have a thread that
updates our position and returns it to with get_current_position().

        :param a: Acceleration samples.
        :param w: Angular velocity samples.
        :param v: Initial velocity, if available.
        :return: The prediction in the end of the series.
        """

        return self.delta_r_v_p(a, w)

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

    def get_current_position(self, ):
        """
Returns latest position update generated by this model. We also receive a
timestamp indicating the time that position was estimated.

        :return: Tuple with (Tensor (1x7) containing latest position update; Respective position time reference).
        """
        if self.position_predictions_buffer.shape[0] > 3 * self.sampling_window_size:
            self.position_predictions_buffer = \
                self.position_predictions_buffer[-2 * self.sampling_window_size:]
            self.orientation_predictions_buffer = \
                self.orientation_predictions_buffer[-2 * self.sampling_window_size:]

        # Rotation/Orientation matrix into quaternion notation.
        # Converts R orientation matrix into equivalent axis-angle notation.
        normalized_axis_r, phi = self.rotation_matrix_into_axis_angle(self.orientation_predictions_buffer[-1])
        # Takes an axis-angle rotation into quaternion rotation
        quaternion_orientation_r = self.axis_angle_into_quaternion(normalized_axis_r, phi)

        self.new_predictions_arrived.clear()

        # Return the last position update and drop out the associated timestamp.
        return torch.cat((self.position_predictions_buffer[-1, 1:], quaternion_orientation_r)), self.position_predictions_buffer[-1, 0]

    def set_initial_position(self, position, timestamp=time.time()):
        """
    This Handler calculates position according to displacement from initial
    position. It gives you absolute position estimatives if the initial
    position is set according to your global reference.

        :param position: Starting position (px, py, pz, qw, qx, qy, qz).
        """

        position = torch.tensor(position, device=self.device,
                                dtype=self.dtype, requires_grad=False)

        timestamp = torch.tensor(timestamp, device=self.device,
                                 dtype=self.dtype, requires_grad=False)

        self.position_predictions_buffer[0, 1:] = position[:3]
        self.position_predictions_buffer[0, 0:1] = timestamp

        axis, angle = self.quaternion_into_axis_angle(position[3:])

        # Converting anxis-angle into skew, and skew into rotation matrix.
        self.orientation_predictions_buffer[0] = \
            torch.matrix_exp(
                self.skew_matrix_from_tensor(
                    axis * angle
                )
            )

        return

    def push_imu_sample_to_buffer(self, a_sample, w_sample, timestamp):
        """
Stores a new IMU sample to the buffer, so it will be taken into account in next
predictions.

        :param sample: List or iterable containig 6 channels of IMU sample (ax, ay, az, wx, wq, wz).
        :return: Void.
        """

        if self.a_imu_buffer.shape[0] > self._imu_buffer_size_limit:
            self.a_imu_buffer = \
                self.a_imu_buffer[-self._imu_buffer_reduction_threshold:, :]
            self.w_imu_buffer = \
                self.w_imu_buffer[-self._imu_buffer_reduction_threshold:]

        self.a_imu_buffer = \
            torch.cat((self.a_imu_buffer,
                       torch.tensor(
                           [[timestamp] +
                            [i * self.delta_t for i in list(a_sample)]],
                           dtype=self.dtype, requires_grad=False,
                           device=self.device)
                       ), dim=0)

        self.w_imu_buffer = \
            torch.cat(
                (self.w_imu_buffer,
                 torch.matrix_exp(self.delta_t * self.skew_matrix_from_tensor(w_sample)).view(1, 3, 3)
                 ), dim=0)

        # Informs this thread that new samples have been generated.
        self._imu_samples_arrived.set()

        return

    # This class is not intended to be used as training for the InertialModule
    # first stage, so we disable grad for faster computation.
    @torch.no_grad()
    def run(self):

        # We only can start predicting positions after we have the
        # sampling_window_size IMU samples collected.
        while self.a_imu_buffer.shape[0] < self.sampling_window_size:
            time.sleep(1)

        # Put pytorch module into evaluation mode (batchnorm and dropout need).
        self.eval()

        # If this thread is not stopped, continue updating position.
        while self.stop_flag is False:
            # Wait for IMU new readings
            self._imu_samples_arrived.wait()

            # We check the timestamp of the last sampling_window_size reading and
            # search for the closest timestamp in prediction buffer.
            prediction_closest_timestamp, prediction_idx = \
                IMUHandler.find_nearest(
                    self.position_predictions_buffer[:, 0],
                    self.a_imu_buffer[-self.sampling_window_size, 0]
                )

            # We also need the prediction's closest reading, that may not be
            # the one at sampling_window_size before
            imu_closest_timestamp, imu_idx = \
                IMUHandler.find_nearest(
                    self.a_imu_buffer[:, 0],
                    prediction_closest_timestamp
                )

            # We need to now for what time we are calculating position.
            current_timestamp = self.a_imu_buffer[-1, 0:1]

            # Registers that there are no more NEW imu readings to process.
            self._imu_samples_arrived.clear()

            # We discard the timestamp in a_imu_buffer
            predicao = self(self.a_imu_buffer[imu_idx:, 1:], self.w_imu_buffer[imu_idx:])

            self.position_predictions_buffer = \
                torch.cat((
                    self.position_predictions_buffer,
                    torch.cat((
                        current_timestamp,
                        self.position_predictions_buffer[prediction_idx, 1:] +
                        predicao[0]
                    ), dim=0).view(1, -1)
                ))

            self.orientation_predictions_buffer = \
                torch.cat((
                    self.orientation_predictions_buffer,
                    torch.matmul(self.orientation_predictions_buffer[prediction_idx], predicao[1]).view(1, 3, 3)
                ))

            self.new_predictions_arrived.set()

    @staticmethod
    def find_nearest(tensor_to_search, value):
        """
This method takes 1 tensor as first argument and a value to find the element
in array whose value is the closest. Returns the closest value element and its
index in the original array.

        :param tensor_to_search: Reference tensor.
        :param value: Value to find closest element.
        :return: Tuple (Element value, element index).
        """
        idx = (absolute(tensor_to_search - value)).argmin()
        return tensor_to_search[idx], idx

    def delta_r_v_p(self, a_samples, w_samples, initial_velocity=0):
        """
    This method computes delta R, v and p (orientation, velocity and position),
    according to https://arxiv.org/abs/2101.07061 and
    https://arxiv.org/abs/1512.02363 about inertial feature preintegration.

        :param a_samples: IMU accelerometer input samples to compute over.
        :param w_samples: IMU gyroscope input samples to compute over.
        :param initial_velocity: Whenever using a statefull approach, passing
        initial_velocity will bring up better preintegration results.
        :return: R (converted from matriz into quaternion), v and p (both 3D tensors).
        """
        # orientation matrix
        delta_r = self.identity_matrix.clone()
        # velocity tensor (3 element)
        delta_v = 0.0
        # position tensor (3 element)
        delta_p = 0.0

        # avoid dividing delta_t by 2 on every loop iteration
        delta_t_divided_by_2 = self.delta_t / 2

        # interactive productory and summation steps
        for w_k, a_k in zip(w_samples, a_samples):
            delta_r = torch.matmul(delta_r, w_k)
            delta_v += torch.matmul(delta_r, a_k)
            # Slightly different from original paper, now including
            # initial_velocity (if available) to compute CURRENT velocity, not
            # only delta_v (variation)
            delta_p += (initial_velocity + delta_v) * self.delta_t + torch.matmul(delta_r, a_k * delta_t_divided_by_2)

        return delta_p, delta_r, delta_v

    def rotation_matrix_into_axis_angle(self, r_matrix):
        """
    Converts a 3x3 rotation matrix into equivalente axis-angle rotation.

        :param r_matrix: 3x3 rotation matrix (tensor).
        :return: Tuple -> (normalized_axis (3-element tensor), rotation angle)
        """
        # Converts R orientation matrix into equivalent skew matrix. SO(3) -> so(3)
        # phi is a simple rotation angle (the value in radians of the angle of rotation)
        phi = torch.acos((torch.trace(r_matrix) - 1) / 2)

        # Skew "orientation" matrix into axis-angles tensor (3-element).
        # we do not multiply by phi, so we have a normalized rotation AXIS (in a SKEW matrix yet)
        # normalized because we didnt multiply the axis by the rotation angle (phi)
        return self.tensor_from_skew_matrix((r_matrix - r_matrix.T) / (2 * torch.sin(phi))), phi

    def axis_angle_into_quaternion(self, normalized_axis, angle):
        """
    Takes an axis-angle rotation and converts into quaternion rotation.
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

        :param normalized_axis: Axis of rotation (3-element tensor).
        :param angle: Simple rotation angle (float or 1-element tensor).
        :return: 4-element tensor, containig quaternion (q0,q1,q2,q3).
        """
        # From axis-angle notation into quaternion notation.
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        quaternion_orientation_r = torch.zeros((4,), dtype=self.dtype, device=self.device)
        quaternion_orientation_r[0] = torch.cos(angle / 2)
        quaternion_orientation_r[1] = torch.sin(angle / 2) * normalized_axis[0]
        quaternion_orientation_r[2] = torch.sin(angle / 2) * normalized_axis[1]
        quaternion_orientation_r[3] = torch.sin(angle / 2) * normalized_axis[2]

        return quaternion_orientation_r

    def quaternion_into_axis_angle(self, quaternion):
        """
    Takes an quaternion rotation and converts into axis-angle rotation.
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

        :param quaternion: 4-element tensor, containig quaternion (q0,q1,q2,q3).
        :return: (Axis of rotation (3-element tensor), Simple rotation angle (float or 1-element tensor))
        """
        # Simple rotation angle
        angle = torch.acos(quaternion[0]) * 2

        # Avoids recalculating this sin.
        sin_angle_2 = torch.sin(angle / 2)

        # Rotation axis
        normalized_axis = torch.zeros((3,))
        normalized_axis[0] = quaternion[1] / sin_angle_2
        normalized_axis[1] = quaternion[2] / sin_angle_2
        normalized_axis[2] = quaternion[3] / sin_angle_2

        return normalized_axis, angle

    def skew_matrix_from_tensor(self, x):
        """
    Receives a 3-element tensor and return its respective skew matrix.

        :param x: 3-element tensor.
        :return: Respective skew-matrix (3x3)
        """
        return torch.tensor([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0],
        ], dtype=self.dtype, device=self.device, requires_grad=False)

    def tensor_from_skew_matrix(self, x):
        """
    Receives a skew matrix and returns its associated 3-element vector (tensor).

        :param x: Skew matrix (3x3)
        :return: Associated tensor (3-element).
        """
        return torch.tensor([x[2][1], x[0][2], x[1][0]],
                            dtype=self.dtype,
                            device=self.device,
                            requires_grad=False)


class ORB_SLAM3(nn.Module):
    def __init__(self, input_port, output_port):
        """
        """
        super().__init__()

        # setup socket
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PAIR)
        zmq_socket.bind("tcp://127.0.0.1:6009")



        return

    def forward(self, input_seq):
        # Read file content
        img_number = 1403638128445096960 + i
        with open("/home/patrickctrf/Documentos/ORB_SLAM3/MH04/mav0/cam0/data/" +
                 img_files_names[i], 'rb') as f:
            bytes = bytearray(f.read())

            # Encode to send
            strng = base64.b64encode(bytes)
            print("Sending file over")
            print("\n\nEncoded message size: ", len(bytes))  # 4194328 in my case
            print("\n\nEncoded message size: ", len(strng))  # 4194328 in my case
            zmq_socket.send(strng)
            pose_dict = json.loads(zmq_socket.recv_string())
            print("pose_dict: ", pose_dict)

            return


