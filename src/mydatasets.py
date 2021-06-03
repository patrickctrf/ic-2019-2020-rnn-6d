import torch
from numpy import arange, random, ones, hstack, array, cumsum, argwhere, load, zeros, save, array_split, where
from pandas import read_csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import int64, from_numpy, cat
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ptk import find_nearest, timeseries_split

# Specifying which modules to import when "import *" is called over this module.
# Also avoiding to import the smae things this module imports
__all__ = ["PackingSequenceDataloader", "AsymetricalTimeseriesDataset", "BatchTimeseriesDataset", "CustomDataLoader", "PlotLstmDataset"]


class PackingSequenceDataloader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
Utility class for loading data with input already formated as packed sequences
(for PyTorch LSTMs, for example).

        :param dataset: PyTorch-like Dataset to load.
        :param batch_size: Mini batch size.
        :param shuffle: Shuffle data.
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Esta equacao apenas ARREDONDA a quantidade de mini-batches para CIMA
        # Quando o resultado da divisao nao eh inteiro (sobra resto), significa
        # apenas que o ultimo batch sera menor que os demais
        self.length = len(dataset) // self.batch_size + (len(dataset) % self.batch_size > 0)

        self.shuffle_array = arange(self.length)
        if shuffle is True: random.shuffle(self.shuffle_array)

    def __iter__(self):
        """
Returns an iterable of itself.

        :return: Iterable around this class.
        """
        self.counter = 0
        return self

    def __next__(self):
        """
Intended to be used as iterator.

        :return: Tuple containing (input packed sequence, output targets)
        """
        if self.counter >= self.length:
            raise StopIteration()

        mini_batch_input = pack_sequence(self.dataset[self.shuffle_array[self.counter:self.counter + self.batch_size]][0], enforce_sorted=False)
        mini_batch_output = torch.stack(self.dataset[self.shuffle_array[self.counter:self.counter + self.batch_size]][1])
        self.counter = self.counter + 1

        return mini_batch_input, mini_batch_output

    def __len__(self):
        return self.length


class AsymetricalTimeseriesDataset(Dataset):
    def __init__(self, x_csv_path, y_csv_path, max_window_size=200, min_window_size=10, convert_first=False, device=torch.device("cpu"), shuffle=True):
        super().__init__()
        self.input_data = read_csv(x_csv_path).to_numpy()
        self.output_data = read_csv(y_csv_path).to_numpy()
        self.min_window_size = min_window_size
        self.convert_first = convert_first
        self.device = device

        # =========SCALING======================================================
        # features without timestamp (we do not scale timestamp)
        input_features = self.input_data[:, 1:]
        output_features = self.output_data[:, 1:]

        # Scaling data
        self.input_scaler = StandardScaler()
        input_features = self.input_scaler.fit_transform(input_features)
        self.output_scaler = MinMaxScaler()
        output_features = self.output_scaler.fit_transform(output_features)

        # Replacing scaled data (we kept the original TIMESTAMP)
        self.input_data[:, 1:] = input_features
        self.output_data[:, 1:] = output_features
        # =========end-SCALING==================================================

        # Save timestamps for syncing samples.
        self.input_timestamp = self.input_data[:, 0]
        self.output_timestamp = self.output_data[:, 0]

        # Throw out timestamp, we are not going to RETURN this.
        self.input_data = self.input_data[:, 1:]
        self.output_data = self.output_data[:, 1:]

        # We are calculating the number of samples that each window size produces.
        # We discount the window SIZE from the total number of samples (timeseries).
        # P.S.: It MUST use OUTPUT shape, because unlabeled data doesnt not help us.
        # P.S. 2: The first window_size is min_window_size, NOT 1.
        n_samples_per_window_size = ones((max_window_size - min_window_size,)) * self.output_data.shape[0] - arange(min_window_size + 1, max_window_size + 1)

        # Now, we know the last index where we can sample for each window size.
        # Concatenate element [0] in the begining to avoid error on first indices.
        self.last_window_sample_idx = hstack((array([0]), cumsum(n_samples_per_window_size))).astype("int")

        self.length = int(n_samples_per_window_size.sum())
        self.indices = arange(self.length)

        self.shuffle_array = arange(self.length)
        if shuffle is True: random.shuffle(self.shuffle_array)

        return

    def __getitem__(self, idx):
        """
Get itens from dataset according to idx passed. The return is in numpy arrays.

        :param idx: Index or slice to return.
        :return: 2 elements or 2 lists (x,y) values, according to idx.
        """

        # If we receive an index, return the sample.
        # Else, if receiving an slice or array, return an slice or array from the samples.
        if not isinstance(idx, slice):

            if idx >= len(self) or idx < 0:
                raise IndexError('Index out of range')

            # shuffling indices before return
            idx = self.shuffle_array[idx]

            argwhere_result = argwhere(self.last_window_sample_idx < idx)
            window_size = self.min_window_size + (argwhere_result[-1][0] if argwhere_result.size != 0 else 0)

            window_start_idx = idx - self.last_window_sample_idx[(argwhere_result[-1][0] if argwhere_result.size != 0 else 0)]

            _, x_start_idx = find_nearest(self.input_timestamp, self.output_timestamp[window_start_idx])
            _, x_finish_idx = find_nearest(self.input_timestamp, self.output_timestamp[window_start_idx + window_size])

            x = self.input_data[x_start_idx: x_finish_idx + 1]
            y = self.output_data[window_start_idx + window_size] - self.output_data[window_start_idx]

            # If we want to convert into torch tensors first
            if self.convert_first is True:
                return from_numpy(x.astype("float32")).to(self.device), \
                       from_numpy(y.astype("float32")).to(self.device)
            else:
                return x, y
        else:
            # If we received a slice(e.g., 0:10:-1) instead an single index.
            return self.__getslice__(idx)

    def __getslice__(self, slice_from_indices):
        return list(zip(*[self[i] for i in self.indices[slice_from_indices]]))

    def __len__(self):
        return self.length


class BatchTimeseriesDataset(Dataset):
    def __init__(self, x_csv_path, y_csv_path, max_window_size=200, min_window_size=10, shuffle=True, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

        self.base_dataset = \
            AsymetricalTimeseriesDataset(x_csv_path=x_csv_path,
                                         y_csv_path=y_csv_path,
                                         max_window_size=max_window_size,
                                         min_window_size=min_window_size,
                                         convert_first=True,
                                         device=torch.device("cpu"),
                                         shuffle=False)

        try:
            tabela = load(str(x_csv_path) + "_tabela_elementos_dataset.npy")

        except FileNotFoundError as e:
            tabela = zeros((len(self.base_dataset),))
            i = 0
            for element in tqdm(self.base_dataset):
                tabela[i] = element[0].shape[0]
                i = i + 1
            save(str(x_csv_path) + "_tabela_elementos_dataset.npy", tabela)

        # dict_count = Counter(tabela)
        # ocorrencias = array(list(dict_count.values()))

        # Os arrays nesta lista contem INDICES para elementos do dataset com
        # mesmo comprimento.
        self.lista_de_arrays_com_mesmo_comprimento = []

        # grupos de arrays com mesmo comprimento.
        # Eles serao separados em batches, entao alguns
        # batches sao de arrays com mesmo comprimento que outros. Alguns
        # batches serao um pouco maiores, pois a quantidade de elementos com o
        # mesmo tamanho talvez nao seja um multiplo inteiro do batch_size
        # escolhido
        for i in range(tabela.min().astype("int"), tabela.max().astype("int") + 1):
            self.lista_de_arrays_com_mesmo_comprimento.extend(
                array_split(where(tabela == i)[0],
                            where(tabela == i)[0].shape[0] // self.batch_size + (where(tabela == i)[0].shape[0] % self.batch_size > 0))
            )

        self.length = len(self.lista_de_arrays_com_mesmo_comprimento)

        self.shuffle_array = arange(self.length)

        if shuffle is True:
            random.shuffle(self.shuffle_array)

        return

    def __getitem__(self, idx):
        """
Get itens from dataset according to idx passed. The return is in numpy arrays.

        :param idx: Index to return.
        :return: 2 elements (batches), according to idx.
        """

        # If we are shuffling indices, we do it here. Else, we'll just get the
        # same index back
        idx = self.shuffle_array[idx]

        # concatenate tensor in order to assemble batches
        x_batch = \
            cat([self.base_dataset[dataset_element_idx][0].unsqueeze(0)
                 for dataset_element_idx
                 in self.lista_de_arrays_com_mesmo_comprimento[idx]], 0)
        y_batch = \
            cat([self.base_dataset[dataset_element_idx][1].unsqueeze(0)
                 for dataset_element_idx
                 in self.lista_de_arrays_com_mesmo_comprimento[idx]], 0)

        return x_batch, y_batch

    def __len__(self):
        return self.length


class CustomDataLoader(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.dataloader = DataLoader(*args, **kwargs)
        self.iterable = None

    def __iter__(self):
        self.iterable = iter(self.dataloader)
        return self

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        next_sample = next(self.iterable)
        # Discard batch dimension, since dataloader is going to add this anyway
        return next_sample[0].view(next_sample[0].shape[1:]), \
               next_sample[1].view(next_sample[1].shape[1:])


class PlotLstmDataset(Dataset):
    def __init__(self, x_csv_path, y_csv_path, max_window_size=200, min_window_size=10, convert_first=False, device=torch.device("cpu"), shuffle=True, offset=40, zeros_tuple=(1, 1, 20)):
        super().__init__()
        self.input_data = read_csv(x_csv_path).to_numpy()
        self.output_data = read_csv(y_csv_path).to_numpy()
        self.min_window_size = min_window_size
        self.convert_first = convert_first
        self.device = device
        self.zeros_tuple = zeros_tuple

        # =========SCALING======================================================
        # features without timestamp (we do not scale timestamp)
        input_features = self.input_data[:, 1:]
        output_features = self.output_data[:, 1:]

        # Scaling data
        self.input_scaler = StandardScaler()
        input_features = self.input_scaler.fit_transform(input_features)
        self.output_scaler = MinMaxScaler()
        output_features = self.output_scaler.fit_transform(output_features)

        # Replacing scaled data (we kept the original TIMESTAMP)
        self.input_data[:, 1:] = input_features
        self.output_data[:, 1:] = output_features
        # =========end-SCALING==================================================

        # Save timestamps for syncing samples.
        self.input_timestamp = self.input_data[:, 0]
        self.output_timestamp = self.output_data[:, 0]

        # Throw out timestamp, we are not going to RETURN this.
        self.input_data = self.input_data[:, 1:]
        self.output_data = self.output_data[:, 1:]

        self.X, _ = timeseries_split(self.input_data, enable_asymetrical=True)

        self.X = self.X[offset:]

        self.length = self.X.shape[0]

        return

    def __getitem__(self, idx):
        """
Get itens from dataset according to idx passed. The return is in numpy arrays.

        :param idx: Index or slice to return.
        :return: 2 elements or 2 lists (x,y) values, according to idx.
        """

        if idx >= len(self) or idx < 0:
            raise IndexError('Index out of range')

        return from_numpy(self.X[idx]), 0

    def __len__(self):
        return self.length
