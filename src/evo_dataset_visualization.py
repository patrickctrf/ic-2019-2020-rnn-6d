import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import arange, random, array
from pandas import read_csv
from torch.utils.data import DataLoader
from tqdm import tqdm

from mydatasets import *
from ptk.timeseries import *
from models import *


def experiment(device):
    """
Runs the experiment itself.

    :return: Trained model.
    """
    aux_list = []
    for i in range(2, 4000):
        euroc_v2_02_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V2_02_medium/mav0/imu0/data.csv", y_csv_path="dataset-files/V2_02_medium/mav0/state_groundtruth_estimate0/data.csv",
                                                           min_window_size=i, max_window_size=i+1, shuffle=False, noise=None, convert_first=False)

        aux_list.append(euroc_v2_02_dataset[0][1])

        print(euroc_v2_02_dataset[0][1])

    dataset_test = np.array(aux_list)

    import pandas as pd

    pd.DataFrame(dataset_test).to_csv("visualization_dataset.csv")

    return None


if __name__ == '__main__':

    # plot_csv()

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Usando GPU")
    else:
        dev = "cpu"
        print("Usando CPU")
    device = torch.device(dev)

    euroc_v1_01_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V1_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None, convert_first=True)

    # Esse daqui gera NAN no treino e na validacao, melhor nao usar
    euroc_v2_01_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V2_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None, convert_first=True)

    euroc_v2_02_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V2_02_medium/mav0/imu0/data.csv", y_csv_path="dataset-files/V2_02_medium/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None, convert_first=True)

    euroc_v2_03_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V2_03_difficult/mav0/imu0/data.csv",
                                                       y_csv_path="dataset-files/V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None, convert_first=True)

    euroc_v1_02_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V1_02_medium/mav0/imu0/data.csv", y_csv_path="dataset-files/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None, convert_first=True)

    euroc_v1_03_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V1_03_difficult/mav0/imu0/data.csv",
                                                       y_csv_path="dataset-files/V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=200, max_window_size=201, shuffle=False, noise=None, convert_first=True)

    euroc_mh1_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, noise=None, convert_first=True)

    euroc_mh2_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_02_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, noise=None, convert_first=True)

    euroc_mh3_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_03_medium/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, noise=None, convert_first=True)

    euroc_mh4_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_04_difficult/mav0/imu0/data.csv",
                                                     y_csv_path="dataset-files/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, noise=None, convert_first=True)

    euroc_mh5_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_05_difficult/mav0/imu0/data.csv",
                                                     y_csv_path="dataset-files/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, noise=None, convert_first=True)

    experiment(device=device)

    # model = [Parameter(torch.randn(2, 2, requires_grad=True))]
    # optimizer = SGD(model, 0.1)
    # scheduler1 = ExponentialLR(optimizer, gamma=0.15, last_epoch=-1)
    #
    # for i in range(10):
    #     print(optimizer.param_groups[0]["lr"])
    #     scheduler1.step()
