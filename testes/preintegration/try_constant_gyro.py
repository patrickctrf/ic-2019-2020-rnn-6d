from models import *
from ptk.utils.torchtools import quaternion_into_axis_angle


def experiment(device):
    """
Runs the experiment itself.

    :return: Trained model.
    """

    model = EachSamplePreintegrationModule()

    n_steps = 10 ** 4
    ones = torch.ones(1, n_steps, 1)
    zeros = torch.zeros(1, n_steps, 1)

    input_tensor = torch.cat((ones, zeros, zeros, zeros, zeros, zeros,), dim=2)

    output_tensor = model(input_tensor)

    axis_angle_list = [quaternion_into_axis_angle(output_tensor[:, i, 9:13]) for i in range(output_tensor.shape[1])]

    print("Axis and Angle: ", *axis_angle_list, sep='\n')

    print(model)

    return model


if __name__ == '__main__':
    dev = "cpu"
    print("Usando CPU")
    device = torch.device(dev)

    experiment(device)
