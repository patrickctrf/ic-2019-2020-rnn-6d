import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv

import ptk
from models import *
from ptk.utils.torchtools import *


def experiment(device):
    """
Runs the experiment itself.

    :return: Trained model.
    """

    model = EachSamplePreintegrationModule()

    n_steps = 10
    ones = torch.ones(1, n_steps, 1)
    zeros = torch.zeros(1, n_steps, 1)

    dados_de_entrada_imu = read_csv("dataset-files/V1_01_easy/mav0/imu0/data.csv").to_numpy()[1000:, 1:]
    dados_de_saida = read_csv("dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv").to_numpy()[:, 1:]

    timestamp_imu = read_csv("dataset-files/V1_01_easy/mav0/imu0/data.csv").to_numpy()[1000:, 0:1]
    timestamp_dados_de_saida = read_csv("dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv").to_numpy()[:, 0:1]

    output_tensor = model(torch.tensor(dados_de_entrada_imu, dtype=torch.float32).unsqueeze(dim=0))[0]

    quaternios_de_variacao = output_tensor[:, 9:13]

    _, index = ptk.utils.numpytools.find_nearest(timestamp_dados_de_saida, timestamp_imu[0])

    quaternios_absolutos = axis_angle_into_quaternion(
        *rotation_matrix_into_axis_angle(
            np.matmul(
                axis_angle_into_rotation_matrix(
                    *quaternion_into_axis_angle(
                        torch.tensor(dados_de_saida[index][3:], dtype=torch.float32).unsqueeze(dim=0)
                    )
                ),
                axis_angle_into_rotation_matrix(
                    *quaternion_into_axis_angle(
                        torch.tensor(quaternios_de_variacao, dtype=torch.float32)
                    )
                )
            )
        )
    )

    output_tensor[:, 9:13] = quaternios_absolutos

    dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    for i, dim_name in enumerate(dimensoes):
        plt.close()
        plt.plot(timestamp_imu, output_tensor[:, 6 + i])
        plt.plot(timestamp_dados_de_saida, dados_de_saida[:, i])
        plt.legend(['predict', 'reference'], loc='upper right')
        plt.title(dim_name)
        plt.savefig(dim_name + ".png", dpi=200)
        plt.show()

    print(model)

    return model


if __name__ == '__main__':
    dev = "cpu"
    print("Usando CPU")
    device = torch.device(dev)

    experiment(device)
