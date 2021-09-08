from collections import Counter

from matplotlib import pyplot as plt
from numpy import zeros, load, array, save, where, array_split
from tqdm import tqdm

from ptk import AsymetricalTimeseriesDataset

dataset = \
    AsymetricalTimeseriesDataset(
        x_csv_path="dataset-files/dataset-room2_512_16/mav0/imu0/data.csv",
        y_csv_path="dataset-files/dataset-room2_512_16/mav0/mocap0/data.csv", shuffle=False,
        convert_first=False, min_window_size=100, max_window_size=350)

try:
    tabela = load("tabela_elementos_dataset.npy")

except FileNotFoundError as e:
    tabela = zeros((len(dataset),))
    i = 0
    for element in tqdm(dataset):
        tabela[i] = element[0].shape[0]
        i = i + 1
    save("tabela_elementos_dataset.npy", tabela)

# Estou apenas contando quantas samples eu tenho de cada comprimento
dict_count = Counter(tabela)
# So transformei os valores do dict para um array
ocorrencias = array(list(dict_count.values()))

print(array_split(where(tabela == 167)[0], where(tabela == 167)[0].shape[0] // 64))

plt.plot(tabela)
plt.show()

x = 1
