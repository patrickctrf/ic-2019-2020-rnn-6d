# Load the Pandas libraries with alias 'pd'
import pandas as pd
import scipy
import numpy as np
from scipy import signal
from skimage.transform import resize

import numpy as np

df = pd.read_csv("orb.txt", sep=" ").to_numpy()

df[:, 1:4] = df[:, 1:4] + np.random.normal(0, 0.05, (df.shape[0], 3))

df = signal.resample(df, 19721)

# ref = np.linspace(df[0, 0], df[-1, 0], 18132, axis=0)
#
# df = scipy.interpolate.interp1d(ref, df)
# df = resize(df, (18334, df.shape[1]), anti_aliasing=False)

df = pd.DataFrame(df)

df.to_csv("inertialORB.txt", index=False, header=False, sep=" ",) # float_format='%.9f')

print(df.head())
