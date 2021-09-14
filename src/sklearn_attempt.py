import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import arange, array
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from mydatasets import *


def experiment(device):
    """
Runs the experiment itself.

    :return: Trained model.
    """

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

    dataset = ConcatDataset([euroc_v1_01_dataset, euroc_v1_02_dataset,
                             euroc_mh1_dataset, euroc_mh5_dataset,
                             euroc_v2_03_dataset, euroc_mh4_dataset])

    print("Montando dataset: ")
    x_total = []
    y_total = []
    for x, y in tqdm(dataset):
        x_total.append(x.flatten())
        y_total.append(y)

    x_total = np.array(x_total)
    y_total = np.array(y_total)

    print("Treinando! Pode demorar horas e não temos log...")

    regressor = MultiOutputRegressor(SVR(), n_jobs=4)

    regressor.fit(x_total, y_total)

    # save the model to disk
    filename = 'comparison_model_sklearn.sav'
    pickle.dump(regressor, open(filename, 'wb'))

    # load the model from disk
    regressor = pickle.load(open(filename, 'rb'))

    room2_tum_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/dataset-room2_512_16/mav0/imu0/data.csv", y_csv_path="dataset-files/dataset-room2_512_16/mav0/mocap0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, device=device, )

    euroc_v2_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V2_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                    min_window_size=200, max_window_size=201, shuffle=False, device=device, )

    predict = []
    reference = []
    for i, (x, y) in tqdm(enumerate(euroc_mh3_dataset), total=len(euroc_mh3_dataset)):
        y_hat = regressor.predict(x.reshape(1, -1)).reshape(-1)
        predict.append(y_hat)
        reference.append(y)

    predict = array(predict)
    reference = array(reference)

    dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    for i, dim_name in enumerate(dimensoes):
        plt.close()
        plt.plot(arange(predict.shape[0]), predict[:, i], arange(reference.shape[0]), reference[:, i])
        plt.legend(['predict', 'reference'], loc='upper right')
        plt.savefig(dim_name + ".png", dpi=200)
        plt.show()

    # dados_de_entrada_imu = read_csv("V1_01_easy/mav0/imu0/data.csv").to_numpy()[:, 1:]
    # dados_de_saida = read_csv("V1_01_easy/mav0/state_groundtruth_estimate0/data.csv").to_numpy()[:, 1:]
    #
    # predict = []
    # for i in tqdm(range(0, dados_de_entrada_imu.shape[0] - 30, 30)):
    #     predict.append(
    #         model(
    #             torch.tensor(dados_de_entrada_imu[i:i + 30].reshape(-1, 30, 6), device=device, dtype=torch.float)
    #         )
    #     )
    #
    # predict = torch.cat(predict).detach().cpu().numpy()
    # predict = np.cumsum(predict, axis=0)
    #
    # dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    # for i, dim_name in enumerate(dimensoes):
    #     plt.close()
    #     plt.plot(range(0, dados_de_saida.shape[0], dados_de_saida.shape[0] // predict.shape[0])[:predict.shape[0]], predict[:, i])
    #     plt.plot(range(dados_de_saida.shape[0]), dados_de_saida[:, i])
    #     plt.legend(['predict', 'reference'], loc='upper right')
    #     plt.savefig(dim_name + ".png", dpi=200)
    #     plt.show()

    # ===========FIM-DE-PREDICAO-["px", "py", "pz", "qw", "qx", "qy", "qz"]=====

    print(regressor)

    return regressor


if __name__ == '__main__':
    # plot_csv()

    dev = "cpu"
    print("Usando CPU")
    device = torch.device(dev)

    experiment(device=device)
