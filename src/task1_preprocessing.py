import pandas as pd

DATA_PATH = "data/raw/complaints.csv"

print("Loading sample data...")

df = pd.read_csv(
    DATA_PATH,
    nrows = 10000,
    low_memory=False,
)

print("Dataset loaded successfully!")
print()

print("Shape: ")
print(df.shape)

print()
print("Columns: ")
print(df.columns.tolist())


