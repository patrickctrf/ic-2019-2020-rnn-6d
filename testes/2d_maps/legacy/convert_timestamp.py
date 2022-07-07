# Load the Pandas libraries with alias 'pd'
import numpy as np
import pandas as pd
import torch
from torch import nn

# precisa ter uma dimensao de batch e uma dimensao de CANAL, que nem convolucao.
df = pd.read_csv("BASALT.csv", sep=" ", float_precision='round_trip').to_numpy()

df[:, 0] = df[:, 0] / 1e9

df = pd.DataFrame(df)

df.to_csv("BASALT.csv", index=False, header=False, sep=" ", float_format='%.9f')

print(df.head())
