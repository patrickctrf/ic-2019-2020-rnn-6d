from torch import cat
from tqdm import tqdm

from ptk.utils import BatchTimeseriesDataset

dataset = \
    BatchTimeseriesDataset(
        x_csv_path="dataset-files/dataset-room2_512_16/mav0/imu0/data.csv",
        y_csv_path="dataset-files/dataset-room2_512_16/mav0/mocap0/data.csv",
        convert_first=True, min_window_size=100, max_window_size=350,
        batch_size=64
    )

old_dataset = dataset.base_dataset

batch_talvez = cat((old_dataset[0][0].unsqueeze(0), old_dataset[1][0].unsqueeze(0)), 0)

element = dataset[0]

i = 0
for element in tqdm(dataset):
    i = i + 1

x = 1
