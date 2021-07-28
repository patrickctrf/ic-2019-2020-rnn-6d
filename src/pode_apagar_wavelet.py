import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
from src.models import SignalWavelet

dwt = DWT1DForward(wave='db6', J=3)
dwt = SignalWavelet()
X = torch.randn(10, 5, 100)
yl, yh = dwt(X)
print(yl.shape)
print(yh[0].shape)
print(yh[1].shape)
print(yh[2].shape)
idwt = DWT1DInverse(wave='db6')
x = idwt((yl, yh))

print(x)
