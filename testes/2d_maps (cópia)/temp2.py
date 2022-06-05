# Load the Pandas libraries with alias 'pd'
import numpy as np
import pandas as pd
import torch
from torch import nn

# precisa ter uma dimensao de batch e uma dimensao de CANAL, que nem convolucao.
df = torch.tensor([[pd.read_csv("orb.txt", sep=" ").to_numpy()]])

upsample_linear = nn.Upsample(size=(40000, 4), mode='bilinear', )

upsample_knearest = nn.Upsample(size=(40000, 4), mode='nearest', )

df_novo = np.zeros((40000,) + (df.shape[-1],))

# linear da coluna do tempo e das posicoes px py pz
df_novo[:, 0:4] = upsample_linear(df[:, :, :, 0:4])[0][0].detach().numpy() + np.random.normal(0, 0.05, (df_novo.shape[0], 4))
df_novo[:, 4:] = upsample_knearest(df[:, :, :, 4:])[0][0].detach().numpy()

df = pd.DataFrame(df_novo)

df.to_csv("NOVO.txt", index=False, header=False, sep=" ", float_format='%.9f')

print(df.head())
