import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import hilbert, chirp
from torch import tensor
from torch.nn import Conv1d

from src.models import SignalEnvelope


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


n_channels = 2

kernel_size = 9
# garante que o kernel seja impar
kernel_size = (kernel_size // 2) * 2 + 1
envelope_layer = Conv1d(n_channels, n_channels, (kernel_size,), padding=(kernel_size // 2,), padding_mode="replicate", bias=False).requires_grad_(False)
envelope_layer.weight[:, :, :] = torch.zeros(n_channels, n_channels, kernel_size, )
for i in range(n_channels):
    envelope_layer.weight[i, i, :] = torch.ones(kernel_size, )

norm_order = 9
kernel_size = 20
# garante que o kernel seja impar
kernel_size = (kernel_size // 2) * 2 + 1
avg_layer = Conv1d(n_channels, n_channels, (kernel_size,), padding=(kernel_size // 2,), padding_mode="replicate", bias=False).requires_grad_(False)
avg_layer.weight[:, :, :] = torch.zeros(n_channels, n_channels, kernel_size, )
for i in range(n_channels):
    avg_layer.weight[i, i, :] = torch.ones(kernel_size, ) / kernel_size

duration = 1.0
fs = 400.0
samples = int(fs * duration)
t = np.arange(samples) / fs

signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t))
signal += 10 * t ** 2

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)

signal = torch.tensor(signal.reshape(1, 1, -1), dtype=torch.float32)
signal = torch.cat((signal, torch.ones(signal.shape)), dim=1)
media_movel_sinal = avg_layer(tensor(signal, dtype=torch.float32))
upper_envelope = (media_movel_sinal + envelope_layer(tensor((torch.abs(signal - media_movel_sinal)), dtype=torch.float32) ** norm_order) ** (1.0 / norm_order)).numpy()

lower_envelope = (media_movel_sinal - envelope_layer(tensor((torch.abs(signal - media_movel_sinal)), dtype=torch.float32) ** norm_order) ** (1.0 / norm_order)).numpy()

envelope_layer = SignalEnvelope()

envelope = envelope_layer(signal).numpy()

fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(t, signal[0][0], label='signal')
ax0.plot(t, envelope[0][4], label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 120.0)
fig.tight_layout()
plt.show()
