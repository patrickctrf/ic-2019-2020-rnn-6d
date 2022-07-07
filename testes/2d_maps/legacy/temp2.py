# Load the Pandas libraries with alias 'pd'
import numpy as np
import pandas as pd
import torch
from torch import nn

# precisa ter uma dimensao de batch e uma dimensao de CANAL, que nem convolucao.
df = torch.tensor([[pd.read_csv("orb.txt", sep=" ", float_precision='round_trip').to_numpy()]])

upsample_linear = nn.Upsample(size=(19000, 4), mode='bilinear', align_corners=True)

upsample_knearest = nn.Upsample(size=(19000, 4), mode='nearest', )

df_novo = np.zeros((19000,) + (df.shape[-1],))

# linear da coluna do tempo e das posicoes px py pz
df_novo[:, 0:4] = upsample_linear(df[:, :, :, 0:4])[0][0].detach().numpy()  # + np.random.normal(0, 0.05, (df_novo.shape[0], 4))
df_novo[:, 4:] = upsample_knearest(df[:, :, :, 4:])[0][0].detach().numpy()

df_novo[:, 0] = np.linspace(start=df[0, 0, 0, 0].item(), stop=df[0, 0, -1, 0].item(), num=df_novo.shape[0])

df = pd.DataFrame(df_novo)

df.to_csv("NOVO.txt", index=False, header=False, sep=" ", float_format='%.9f')

print(df.head())
