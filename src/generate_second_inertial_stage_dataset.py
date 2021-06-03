import torch
from matplotlib import pyplot as plt
from numpy import save, load
from pandas import read_csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from models import *


def experiment():
    """
Runs the experiment itself.

    :return: Main model used to run the experiment.
    """
    # The model used for predictions
    sampling_window_size = 200
    imu_handler = IMUHandler(sampling_window_size=sampling_window_size, device=device, data_type=torch.double)
    imu_handler.load_feature_extractor()
    imu_handler.double()

    # ===========PREDICAO-TRAJETORIO-INTEIRA====================================
    input_data = read_csv("dataset-room2_512_16/mav0/imu0/data.csv").to_numpy()
    output_data = read_csv("dataset-room2_512_16/mav0/mocap0/data.csv").to_numpy()

    # =========SCALING======================================================
    # features without timestamp (we do not scale timestamp)
    input_features = input_data[:, 1:]
    output_features = output_data[:, 1:]

    # Scaling data
    input_scaler = StandardScaler()
    input_features = input_scaler.fit_transform(input_features)
    output_scaler = MinMaxScaler()
    output_features = output_scaler.fit_transform(output_features)

    # Replacing scaled data (we kept the original TIMESTAMP)
    input_data[:, 1:] = input_features
    output_data[:, 1:] = output_features
    # =========end-SCALING==================================================

    # Save timestamps for syncing samples.
    input_timestamp = input_data[:, 0]
    output_timestamp = output_data[:, 0]

    imu_handler.set_initial_position(output_features[0], output_timestamp[0])
    imu_handler.start()

    offset = 0  # offset_for_convolution_kernel
    predict = torch.zeros((input_features.shape[0], 1, 7)).to(device)

    with torch.no_grad():
        for i, x in tqdm(enumerate(input_features), total=input_features.shape[0]):
            imu_handler.push_imu_sample_to_buffer(torch.tensor(x), input_timestamp[i])
            if i >= sampling_window_size:
                imu_handler.new_predictions_arrived.wait()
                predict[i], _ = (imu_handler.get_current_position())
                imu_handler.new_predictions_arrived.clear()

    predict = predict.view(predict.shape[0], -1).detach().cpu().numpy()
    save("predictions.npy", predict)

    predict = load("predictions.npy")

    dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    for i, dim_name in enumerate(dimensoes):
        plt.close()
        plt.plot(input_timestamp[0 + offset:][:predict.shape[0]], predict[:, i], output_timestamp, output_features[:, i])
        plt.legend(['predict', 'reference'], loc='upper right')
        plt.savefig(dim_name + "_inteira.png", dpi=200)
        plt.show()

    # ===========fim-de-PREDICAO-TRAJETORIO-INTEIRA=============================

    return imu_handler


if __name__ == '__main__':

    # plot_csv()

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Usando GPU")
    else:
        dev = "cpu"
        print("Usando CPU")
    device = torch.device(dev)

    experiment()
