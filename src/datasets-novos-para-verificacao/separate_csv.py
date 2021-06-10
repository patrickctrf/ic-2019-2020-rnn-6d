import pandas as pd

# Open DataFrame from CSV file
df = pd.read_csv("energydata_complete.csv")

# Prints first 5 lines of this
print(df.head())

# Get columns names
nome_colunas = df.columns

print(nome_colunas)

input_df = df[['T1', 'RH_1', 'T2', 'RH_2']]

output_df = df[['T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7']]

input_df.to_csv("input_dataset.csv")

output_df.to_csv("output_dataset.csv")
