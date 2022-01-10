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

    # Recebe os arquivos do dataset e o aloca de no formato (numpy npz) adequado.
    # X, y = format_dataset(dataset_directory="dataset-room2_512_16", file_format="NPZ")
    # join_npz_files(files_origin_path="./tmp_x", output_file="./x_data.npz")
    # join_npz_files(files_origin_path="./tmp_y", output_file="./y_data.npz")
    # return

    model = InertialModule(input_size=6, hidden_layer_size=32, n_lstm_units=3, bidirectional=True, use_amp=False,
                           output_size=7 * 2, training_batch_size=1024, epochs=5, device=device, validation_percent=0.2)

    # model.load_state_dict(torch.load("best_model_state_dict.pth"))
    # model = torch.load("best_model.pth")

    # model = PreintegrationModule(device=device)

    # Carrega o extrator de features convolucional pretreinado e congela (grad)
    # model.load_feature_extractor()

    model.to(device)

    # Let's go fit! Comment if only loading pretrained model.
    # model.fit(X, y)
    model.fit()

    # ===========PREDICAO-["px", "py", "pz", "qw", "qx", "qy", "qz"]============
    device = torch.device("cpu")
    # model = InertialModule(input_size=6, hidden_layer_size=32, n_lstm_units=3, bidirectional=True, use_amp=False,
    #                        output_size=7 * 2, training_batch_size=512, epochs=50, device=device, validation_percent=0.2)
    # model.load_state_dict(torch.load("best_model_state_dict.pth"))
    model = torch.load("best_model.pth")
    model.eval()
    model.to(device)

    print("creation_time carregado: ", model.creation_time)

    predict = []
    reference = []
    dataloader = DataLoader(dataset=euroc_v1_01_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4, multiprocessing_context='spawn')
    for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y_hat = model(x.view(1, x.shape[1], x.shape[2])).view(-1)
        predict.append(y_hat.detach().cpu().numpy())
        reference.append(y.detach().cpu().numpy())

    predict = array(predict)[:, :7]
    reference = array(reference)[:, 0, :]

    dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    for i, dim_name in enumerate(dimensoes):
        plt.close()
        plt.plot(arange(predict.shape[0]), predict[:, i], arange(reference.shape[0]), reference[:, i])
        plt.legend(['predict', 'reference'], loc='upper right')
        plt.title(dim_name)
        plt.savefig(dim_name + ".png", dpi=200)
        # plt.show()

    dados_de_entrada_imu = read_csv("dataset-files/V1_01_easy/mav0/imu0/data.csv").to_numpy()[:, 1:]
    dados_de_saida = read_csv("dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv").to_numpy()[:, 1:]

    predict = []
    for i in tqdm(range(0, dados_de_entrada_imu.shape[0] - 200, 200)):
        predict.append(
            model(
                torch.tensor(dados_de_entrada_imu[i:i + 200].reshape(-1, 200, 6), device=device, dtype=torch.float)
            )
        )

    predict = torch.cat(predict).detach().cpu().numpy()
    predict = np.cumsum(predict, axis=0)

    dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    for i, dim_name in enumerate(dimensoes):
        plt.close()
        plt.plot(range(0, dados_de_saida.shape[0], dados_de_saida.shape[0] // predict.shape[0])[:predict.shape[0]], predict[:, i])
        plt.plot(range(dados_de_saida.shape[0]), dados_de_saida[:, i])
        plt.legend(['predict', 'reference'], loc='upper right')
        plt.title(dim_name)
        plt.savefig(dim_name + ".png", dpi=200)
        # plt.show()

    # ===========FIM-DE-PREDICAO-["px", "py", "pz", "qw", "qx", "qy", "qz"]=====

    print(model)

    return model


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
                                                       min_window_size=99, max_window_size=100, shuffle=False, noise=None, convert_first=True)

    # Esse daqui gera NAN no treino e na validacao, melhor nao usar
    euroc_v2_01_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V2_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=99, max_window_size=100, shuffle=False, noise=None, convert_first=True)

    euroc_v2_02_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V2_02_medium/mav0/imu0/data.csv", y_csv_path="dataset-files/V2_02_medium/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=99, max_window_size=100, shuffle=False, noise=None, convert_first=True)

    euroc_v2_03_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V2_03_difficult/mav0/imu0/data.csv",
                                                       y_csv_path="dataset-files/V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=99, max_window_size=100, shuffle=False, noise=None, convert_first=True)

    euroc_v1_02_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V1_02_medium/mav0/imu0/data.csv", y_csv_path="dataset-files/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=99, max_window_size=100, shuffle=False, noise=None, convert_first=True)

    euroc_v1_03_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/V1_03_difficult/mav0/imu0/data.csv",
                                                       y_csv_path="dataset-files/V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                       min_window_size=99, max_window_size=100, shuffle=False, noise=None, convert_first=True)

    euroc_mh1_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=99, max_window_size=100, shuffle=False, noise=None, convert_first=True)

    euroc_mh2_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_02_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=99, max_window_size=100, shuffle=False, noise=None, convert_first=True)

    euroc_mh3_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_03_medium/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=99, max_window_size=100, shuffle=False, noise=None, convert_first=True)

    euroc_mh4_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_04_difficult/mav0/imu0/data.csv",
                                                     y_csv_path="dataset-files/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=99, max_window_size=100, shuffle=False, noise=None, convert_first=True)

    euroc_mh5_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-files/MH_05_difficult/mav0/imu0/data.csv",
                                                     y_csv_path="dataset-files/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv",
                                                     min_window_size=99, max_window_size=100, shuffle=False, noise=None, convert_first=True)

    experiment(device=device)

    # model = [Parameter(torch.randn(2, 2, requires_grad=True))]
    # optimizer = SGD(model, 0.1)
    # scheduler1 = ExponentialLR(optimizer, gamma=0.15, last_epoch=-1)
    #
    # for i in range(10):
    #     print(optimizer.param_groups[0]["lr"])
    #     scheduler1.step()
