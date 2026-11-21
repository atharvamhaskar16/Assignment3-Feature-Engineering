
import pandas as pd

# Load the Titanic dataset
df = pd.read_csv("data/dataset.csv")

print("Dataset Loaded Successfully!")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())
