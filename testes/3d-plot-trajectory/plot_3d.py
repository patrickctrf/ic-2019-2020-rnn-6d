import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("data.csv")
ax0 = plt.subplot(121, projection='3d')

zline = df[" p_RS_R_z [m]"]
xline = df[" p_RS_R_x [m]"]
yline = df[" p_RS_R_y [m]"]

ax0.plot3D(xline, yline, zline, 'gray')

df = pd.read_csv("visualization_dataset.csv")

zline = df[" p_RS_R_z [m]"]
xline = df[" p_RS_R_x [m]"]
yline = df[" p_RS_R_y [m]"]

ax1 = plt.subplot(122, projection='3d')

ax1.plot3D(xline, yline, zline, 'gray')

plt.show()
