import numpy as np
import pandas as pd
import skinematics
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from models import *
from ptk.utils.numpytools import *


import numpy as np
from scipy.signal import filtfilt, butter
from quaternion import quaternion, from_rotation_vector, rotate_vectors

from src.mydatasets import AsymetricalTimeseriesDataset


def estimate_orientation(a, w, t, alpha=0.9, g_ref=(0., 0., 1.),
                         theta_min=1e-6, highpass=.01, lowpass=.05):
    """ Estimate orientation with a complementary filter.
    Fuse linear acceleration and angular velocity measurements to obtain an
    estimate of orientation using a complementary filter as described in
    `Wetzstein 2017: 3-DOF Orientation Tracking with IMUs`_
    .. _Wetzstein 2017: 3-DOF Orientation Tracking with IMUs:
    https://pdfs.semanticscholar.org/5568/e2100cab0b573599accd2c77debd05ccf3b1.pdf
    Parameters
    ----------
    a : array-like, shape (N, 3)
        Acceleration measurements (in arbitrary units).
    w : array-like, shape (N, 3)
        Angular velocity measurements (in rad/s).
    t : array-like, shape (N,)
        Timestamps of the measurements (in s).
    alpha : float, default 0.9
        Weight of the angular velocity measurements in the estimate.
    g_ref : tuple, len 3, default (0., 0., 1.)
        Unit vector denoting direction of gravity.
    theta_min : float, default 1e-6
        Minimal angular velocity after filtering. Values smaller than this
        will be considered noise and are not used for the estimate.
    highpass : float, default .01
        Cutoff frequency of the high-pass filter for the angular velocity as
        fraction of Nyquist frequency.
    lowpass : float, default .05
        Cutoff frequency of the low-pass filter for the linear acceleration as
        fraction of Nyquist frequency.
    Returns
    -------
    q : array of quaternions, shape (N,)
        The estimated orientation for each measurement.
    """

    # initialize some things
    N = len(t)
    dt = np.diff(t)
    g_ref = np.array(g_ref)
    q = np.ones(N, dtype=quaternion)

    # get high-passed angular velocity
    w = filtfilt(*butter(5, highpass, btype='high'), w, axis=0)
    w[np.linalg.norm(w, axis=1) < theta_min] = 0
    q_delta = from_rotation_vector(w[1:] * dt[:, None])

    # get low-passed linear acceleration
    a = filtfilt(*butter(5, lowpass, btype='low'), a, axis=0)

    for i in range(1, N):

        # get rotation estimate from gyroscope
        q_w = q[i - 1] * q_delta[i - 1]

        # get rotation estimate from accelerometer
        v_world = rotate_vectors(q_w, a[i])
        n = np.cross(v_world, g_ref)
        phi = np.arccos(np.dot(v_world / np.linalg.norm(v_world), g_ref))
        q_a = from_rotation_vector(
            (1 - alpha) * phi * n[None, :] / np.linalg.norm(n))[0]

        # fuse both estimates
        q[i] = q_a * q_w

    return q


def experiment(device):
    """
Runs the experiment itself.

    :return: Trained model.
    """
    dataset = ConcatDataset([euroc_v1_01_dataset])

    print("Montando dataset: ")
    x_total = []
    y_total = []
    for x, y in tqdm(dataset):
        x_total.append(x.flatten())
        y_total.append(y)

    x_total = np.array(x_total)
    y_total = np.array(y_total)

    timestamp_imu = pd.read_csv("dataset-files/V1_01_easy/mav0/imu0/data.csv").to_numpy()[:, 0:1]
    timestamp_dados_de_saida = pd.read_csv("dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv").to_numpy()[:, 0:1]

    _, index = find_nearest(timestamp_imu, timestamp_dados_de_saida[0])

    dados_de_saida = pd.read_csv("dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv").to_numpy()[:, 1:]

    dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    for i, dim_name in enumerate(dimensoes):
        plt.close()
        plt.plot(np.hstack((my_sensor.pos, my_sensor.quat))[:, i])
        plt.plot(range(dados_de_saida.shape[0]), dados_de_saida[:, i])
        plt.legend(['predict', 'reference'], loc='upper right')
        plt.title(dim_name)
        plt.savefig(dim_name + ".png", dpi=200)
        plt.show()

    # dimensoes = ["qw", "qx", "qy", "qz", "px", "py", "pz", ]
    # plt.close()
    # for i, dim_name in enumerate(dimensoes):
    #     plt.plot(np.hstack((my_sensor.quat, my_sensor.pos))[:, i])
    # plt.legend(dimensoes, loc='upper right')
    # plt.title('scikit-kinematics')
    # # plt.savefig('scikit-kinematics' + ".png", dpi=200)
    # plt.show()


if __name__ == '__main__':
    dev = "cpu"
    print("Usando CPU")
    device = torch.device(dev)

    euroc_v1_01_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V1_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None)

    # Esse daqui gera NAN no treino e na validacao, melhor nao usar
    euroc_v2_01_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V2_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None)

    euroc_v2_02_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V2_02_medium/mav0/imu0/data.csv", y_csv_path="dataset-files/V2_02_medium/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None)

    euroc_v2_03_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V2_03_difficult/mav0/imu0/data.csv",
                                                       y_csv_path="dataset-files/V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None)

    euroc_v1_02_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V1_02_medium/mav0/imu0/data.csv", y_csv_path="dataset-files/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None)

    euroc_v1_03_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V1_03_difficult/mav0/imu0/data.csv",
                                                       y_csv_path="dataset-files/V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None)

    euroc_mh1_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, noise=None)

    euroc_mh2_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_02_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, noise=None)

    euroc_mh3_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_03_medium/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, noise=None)

    euroc_mh4_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_04_difficult/mav0/imu0/data.csv",
                                                     y_csv_path="dataset-files/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, noise=None)

    euroc_mh5_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_05_difficult/mav0/imu0/data.csv",
                                                     y_csv_path="dataset-files/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, noise=None)

    experiment(device)
