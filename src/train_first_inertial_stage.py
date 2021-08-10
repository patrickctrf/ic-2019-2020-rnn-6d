import torch
from matplotlib import pyplot as plt
from numpy import arange, random, array
from tqdm import tqdm

from mydatasets import *
from ptk.timeseries import *
from models import *


def experiment():
    """
Runs the experiment itself.

    :return: Trained model.
    """

    # Recebe os arquivos do dataset e o aloca de no formato (numpy npz) adequado.
    # X, y = format_dataset(dataset_directory="dataset-room2_512_16", file_format="NPZ")
    # join_npz_files(files_origin_path="./tmp_x", output_file="./x_data.npz")
    # join_npz_files(files_origin_path="./tmp_y", output_file="./y_data.npz")
    # return

    model = InertialModule(input_size=6, hidden_layer_size=32, n_lstm_units=3, bidirectional=True,
                           output_size=7, training_batch_size=4096, epochs=19, device=device, validation_percent=0.2)

    # model.load_state_dict(torch.load("best_model_state_dict.pth"))

    # model = PreintegrationModule(device=device)

    # Carrega o extrator de features convolucional pretreinado e congela (grad)
    # model.load_feature_extractor()

    model.to(device)

    # Gera os parametros de entrada aleatoriamente.
    hidden_layer_size = random.uniform(40, 80, 20).astype("int")
    n_lstm_units = arange(1, 4)

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'hidden_layer_size': hidden_layer_size, 'n_lstm_units': n_lstm_units}

    splitter = TimeSeriesSplitCV(n_splits=2,
                                 training_percent=0.8,
                                 blocking_split=False)
    regressor = model
    # cv_search = \
    #     BayesSearchCV(estimator=regressor, cv=splitter,
    #                   search_spaces=parametros,
    #                   refit=True,
    #                   n_iter=4,
    #                   verbose=1,
    #                   # n_jobs=4,
    #                   scoring=make_scorer(mean_squared_error,
    #                                       greater_is_better=False,
    #                                       needs_proba=False))

    # Let's go fit! Comment if only loading pretrained model.
    # model.fit(X, y)
    model.fit()

    model.eval()
    # ===========PREDICAO-["px", "py", "pz", "qw", "qx", "qy", "qz"]============
    room2_tum_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-room2_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room2_512_16/mav0/mocap0/data.csv",
                                                     min_window_size=100, max_window_size=101, shuffle=False, device=device, convert_first=True)

    predict = []
    reference = []
    for i, (x, y) in tqdm(enumerate(room2_tum_dataset), total=len(room2_tum_dataset)):
        y_hat = model(x.view(1, x.shape[0], x.shape[1])).view(-1)
        predict.append(y_hat.detach().cpu().numpy())
        reference.append(y.detach().cpu().numpy())

    predict = array(predict)
    reference = array(reference)

    dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    for i, dim_name in enumerate(dimensoes):
        plt.close()
        plt.plot(arange(predict.shape[0]), predict[:, i], arange(reference.shape[0]), reference[:, i])
        plt.legend(['predict', 'reference'], loc='upper right')
        plt.savefig(dim_name + ".png", dpi=200)
        plt.show()

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

    experiment()
