from matplotlib import pyplot as plt
from pandas import read_csv

# ======TUM==============================
# IMU
dataset = read_csv("dataset-files/dataset-room1_512_16/mav0/imu0/data.csv").to_numpy()
# Ground Truth
dataset = read_csv("dataset-files/dataset-room1_512_16/mav0/mocap0/data.csv").to_numpy()

# =======EUROC-MAV=======================
# IMU
dataset = read_csv("dataset-files/V2_01_easy/mav0/imu0/data.csv").to_numpy()
# Ground Truth
dataset = read_csv("dataset-files/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv").to_numpy()

print("Latência média: " + str((dataset[1:, 0] - dataset[:-1, 0]).mean() / 1e6) + "ms")

plt.plot(dataset[1:, 0] - dataset[:-1, 0])
plt.show()
