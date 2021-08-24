from mydatasets import ParallelBatchTimeseriesDataset

euroc_v2_01_dataset = ParallelBatchTimeseriesDataset(x_csv_path="V2_01_easy/mav0/imu0/data.csv", y_csv_path="V2_01_easy/mav0/state_groundtruth_estimate0/data.csv", n_threads=10,
                                                     min_window_size=25, max_window_size=35, batch_size=4096, shuffle=False, noise=None)
