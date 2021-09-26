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

        # Get the sampling rate from the second line in the file
        try:
            fh = open(in_file)
            fh.readline()
            line = fh.readline()
            rate = np.float(line.split(':')[1].split('H')[0])
            fh.close()

        except FileNotFoundError:
            print('{0} does not exist!'.format(in_file))
            return -1

        # Read the data
        data = pd.read_csv(in_file,
                           sep='\t',
                           skiprows=4,
                           index_col=False)

        # Extract the columns that you want, and pass them on
        in_data = {'rate': rate,
                   'acc': data.filter(regex='Acc').values,
                   'omega': data.filter(regex='Gyr').values,
                   'mag': data.filter(regex='Mag').values}
        self._set_data(in_data)


def experiment(device):
    """
Runs the experiment itself.

    :return: Trained model.
    """

    quaternion = [0., 1., 0., 0.]

    rotation_matrix = \
        axis_angle_into_rotation_matrix(
            *quaternion_into_axis_angle(
                quaternion
            )
        )

    initial_orientation = rotation_matrix
    initial_position = np.r_[0, 0, 0]

    my_sensor = ScikitOdometry(in_file='data_xsens.txt',
                               R_init=initial_orientation,
                               pos_init=initial_position)

    dimensoes = ["qw", "qx", "qy", "qz", "px", "py", "pz", ]
    plt.close()
    for i, dim_name in enumerate(dimensoes):
        plt.plot(np.hstack((my_sensor.quat, my_sensor.pos))[:, i])
    plt.legend(dimensoes, loc='upper right')
    plt.title('scikit-kinematics')
    plt.savefig('scikit-kinematics' + ".png", dpi=200)
    plt.show()


if __name__ == '__main__':
    dev = "cpu"
    print("Usando CPU")
    device = torch.device(dev)

    experiment(device)
