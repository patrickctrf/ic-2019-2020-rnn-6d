# Load the Pandas libraries with alias 'pd'
import pandas as pd
from skimage.transform import resize

import numpy as np

df = pd.read_csv("orb.txt").to_numpy()

df[:, 1:4] = df[:, 1:4] + np.random.normal(0, 0.05, (df.shape[0], 3))

df = resize(df, (19187, df.shape[1]), anti_aliasing=False)

df = pd.DataFrame(df)

df.to_csv("inertialORB.txt", index=False, header=False)

print(df.head())
