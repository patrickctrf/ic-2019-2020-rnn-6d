import pandas as pd
from numpy import random


def apply_gaussian_noise(x, avg=0, desvpad=0.1):
    return x + random.normal(loc=avg, scale=desvpad)


# Open DataFrame from CSV file
df = pd.read_csv("energydata_complete.csv")

# Prints first 5 lines of this
print(df.head())

# Get columns names
nome_colunas = df.columns

print(nome_colunas)

# Select input and output desired data
input_df = df[['T1', 'RH_1', 'T2', 'RH_2']]
output_df = df[['T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7']]

# insert pseudo-timestamp column at begginig
idx = 0
new_col = range(1, len(df) + 1)
input_df.insert(loc=idx, column='timestamp', value=new_col)
output_df.insert(loc=idx, column='timestamp', value=new_col)

# Save dataframes into CSV files
input_df.to_csv("input_dataset.csv", index=False)
output_df.to_csv("output_dataset.csv", index=False)

# We also create a dataset with noise timestamp, to dislocate samples and test
# my dataset class effectiveness.
noisy_df = pd.read_csv("output_dataset.csv")
noisy_df["timestamp"] = noisy_df["timestamp"].apply(apply_gaussian_noise)
noisy_df.to_csv("noisy_output_dataset.csv", index=False)
