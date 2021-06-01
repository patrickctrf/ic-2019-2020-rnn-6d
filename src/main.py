import torch
from matplotlib import pyplot as plt
from numpy import arange, random, array
from tqdm import tqdm

from mydatasets import *
from ptk.timeseries import *
from src.models import InertialModule


def experiment(repeats):
    """
Runs the experiment itself.

    :param repeats: Number of times to repeat the experiment. When we are trying to create a good network, it is reccomended to use 1.
    :return: Error scores for each repeat.
    """

    # Recebe os arquivos do dataset e o aloca de no formato (numpy npz) adequado.
    # X, y = format_dataset(dataset_directory="dataset-room2_512_16", file_format="NPZ")
    # join_npz_files(files_origin_path="./tmp_x", output_file="./x_data.npz")
    # join_npz_files(files_origin_path="./tmp_y", output_file="./y_data.npz")
    # return

    model = InertialModule(input_size=6, hidden_layer_size=10, n_lstm_units=1, bidirectional=False,
                           output_size=7, training_batch_size=1024, epochs=50, device=device, validation_percent=0.2)

    # Carrega o extrator de features convolucional pretreinado e congela (grad)
    model.load_feature_extractor()

    model.to(device)

    # Gera os parametros de entrada aleatoriamente. Alguns sao uniformes nos
    # EXPOENTES.
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
    # model.fit()

    model.eval()
    # ===========PREDICAO-["px", "py", "pz", "qw", "qx", "qy", "qz"]============
    room2_tum_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-room2_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room2_512_16/mav0/mocap0/data.csv",
                                                     min_window_size=200, max_window_size=201, shuffle=False, device=device, convert_first=True)

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
        # plt.show()

    # ===========FIM-DE-PREDICAO-["px", "py", "pz", "qw", "qx", "qy", "qz"]=====

    print(model)

    error_scores = []

    return error_scores


if __name__ == '__main__':

    # plot_csv()

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Usando GPU")
    else:
        dev = "cpu"
        print("Usando CPU")
    device = torch.device(dev)

    experiment(1)

# # Plot com LSTM. Usar no futuro talvez
#
#     # Realizamos a busca atraves do treinamento
#     # cv_search.fit(X, y.reshape(-1, 1))
#     # print(cv_search.cv_results_)
#     # cv_dataframe_results = DataFrame.from_dict(cv_search.cv_results_)
#     # cv_dataframe_results.to_csv("cv_results.csv")
#
#     # =====================PREDICTION-TEST======================================
#     dataset_directory = "dataset-room2_512_16"
#
#     # Opening dataset.
#     input_data = read_csv(dataset_directory + "/mav0/imu0/data.csv").to_numpy()
#     output_data = read_csv(dataset_directory + "/mav0/mocap0/data.csv").to_numpy()
#
#     # Precisamos restaurar o time para alinhar os dados depois do "diff"
#     original_ground_truth_timestamp = output_data[:, 0]
#
#     # inutil agora, mas deixarei aqui pra nao ter que refazer depois
#     original_imu_timestamp = input_data[:, 0]
#
#     plt.close()
#     plt.plot(output_data[:, 1])
#     plt.show()
#
#     plt.close()
#     plt.plot(output_data[1:, 1] - output_data[:-1, 1])
#     plt.show()
#
#     # Queremos apenas a VARIACAO de posicao a cada instante.
#     output_data = diff(output_data, axis=0)
#
#     plt.close()
#     plt.plot(output_data[:, 1])
#     plt.show()
#
#     # Restauramos a referencia de time original.
#     output_data[:, 0] = original_ground_truth_timestamp[1:]
#
#     # features without timestamp (we do not scale timestamp)
#     input_features = input_data[:, 1:]
#     output_features = output_data[:, 1:]
#
#     # Scaling data
#     input_scaler = StandardScaler()
#     input_features = input_scaler.fit_transform(input_features)
#     output_scaler = MinMaxScaler()
#     output_features = output_scaler.fit_transform(output_features)
#
#     # These arrays/tensors are only helpful for plotting the prediction.
#     X_graphic = torch.from_numpy(input_features.astype("float32")).to(device)
#     y_graphic = output_features.astype("float32")
#
#     # model = cv_search.best_estimator_
#     model = torch.load("best_model.pth")
#     # model.load_state_dict(torch.load("best_model_state_dict.pth"))
#     model.to(device)
#     model.packing_sequence = False
#     yhat = []
#     model.hidden_cell = (torch.zeros(model.num_directions * model.n_lstm_units, 1, model.hidden_layer_size).to(model.device),
#                          torch.zeros(model.num_directions * model.n_lstm_units, 1, model.hidden_layer_size).to(model.device))
#     model.eval()
#     for X in X_graphic:
#         yhat.append(model(X.view(1, -1, 6)).detach().cpu().numpy())
#     # from list to numpy array
#     yhat = array(yhat).reshape(-1, 7)
#
#     # ======================PLOT================================================
#     dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
#     for i, dim_name in enumerate(dimensoes):
#         plt.close()
#         plt.plot(original_imu_timestamp, yhat[:, i], original_ground_truth_timestamp[1:], y_graphic[:, i])
#         plt.legend(['predict', 'reference'], loc='upper right')
#         plt.savefig(dim_name + ".png", dpi=200)
#         plt.show()
#     # rmse = mean_squared_error(yhat, y_graphic) ** 1 / 2
#     # print("RMSE trajetoria inteira: ", rmse)
