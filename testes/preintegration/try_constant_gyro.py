import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import arange, array
from pandas import read_csv
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import *
from mydatasets import *
from ptk import *
from ptk.utils.torchtools import quaternion_into_axis_angle


def experiment(device):
    """
Runs the experiment itself.

    :return: Trained model.
    """

    model = EachSamplePreintegrationModule()

    n_steps = 10
    ones = torch.ones(1, n_steps, 1)
    zeros = torch.zeros(1, n_steps, 1)

    input_tensor = torch.cat((ones, zeros, zeros, zeros, zeros, zeros,), dim=2)

    output_tensor = model(input_tensor)

    axis_angle_list = [quaternion_into_axis_angle(output_tensor[:, i, 9:13]) for i in range(output_tensor.shape[1])]

    print("Axis and Angle: ", *axis_angle_list, sep='\n')

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

    experiment(device)
