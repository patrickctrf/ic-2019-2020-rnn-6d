# Load the Pandas libraries with alias 'pd' 
import pandas as pd

df0 = pd.read_csv("gt_mh04.csv").to_numpy()

df1 = pd.read_csv("trajectory.csv", sep=",").to_numpy()

import matplotlib.pyplot as plt

plt.close()
plt.plot(df0[:, 1], df0[:, 2])
plt.plot(-df1[:, 1], -df1[:, 2])
plt.legend(['Ground Truth', 'ORB+nosso'], loc='upper right')
plt.title("X vs Y")
plt.savefig("XvsY_basalt" + ".png", dpi=200)
plt.show()
plt.close()
