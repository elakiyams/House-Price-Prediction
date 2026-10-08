import pandas as pd

df = pd.read_csv("data/house_data.csv")

print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nInformation:")
print(df.info())

print("\nStatistics:")
print(df.describe())