import numpy as np
import pandas as pd
import skinematics
from matplotlib import pyplot as plt

from models import *
from ptk.utils.numpytools import *


class ScikitOdometry(skinematics.imus.IMU_Base):
    """Concrete class based on abstract base class IMU_Base """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self, in_file, in_data=None):
        '''Get the sampling rate, as well as the recorded data,
        and assign them to the corresponding attributes of "self".

        Parameters
        ----------
        in_file : string
                Filename of the data-file
        in_data : not used here

        Assigns
        -------
        - rate : rate
        - acc : acceleration
        - omega : angular_velocity
        - mag : mag_field_direction
        '''

        dataset = pd.read_csv(in_file).to_numpy()
        omega = dataset[:, 1:4]
        acc = dataset[:, 4:7]

        # Extract the columns that you want, and pass them on
        in_data = {'rate': 200,
                   'acc': acc,
                   'omega': omega}
        self._set_data(in_data)


def experiment(device):
    """
Runs the experiment itself.

    :return: Trained model.
    """

    dataset = \
        pd.read_csv(
            "dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv"
        ).to_numpy()
    initial_position = dataset[0, 1:4]
    initial_orientation = dataset[0, 4:8]

    timestamp_imu = pd.read_csv("dataset-files/V1_01_easy/mav0/imu0/data.csv").to_numpy()[:, 0:1]
    timestamp_dados_de_saida = pd.read_csv("dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv").to_numpy()[:, 0:1]

    _, index = find_nearest(timestamp_dados_de_saida, timestamp_imu[0])

    rotation_matrix = \
        axis_angle_into_rotation_matrix(
            *quaternion_into_axis_angle(
                initial_orientation
            )
        )

    initial_orientation = rotation_matrix

    my_sensor = ScikitOdometry(in_file='dataset-files/V1_01_easy/mav0/imu0/data.csv',
                               R_init=initial_orientation,
                               pos_init=initial_position,
                               q_type="analytical")

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

    experiment(device)
