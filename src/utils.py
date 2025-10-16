import pandas as pd

def summarize_dataset(df):
    print(f"Dataset shape: {df.shape}")
    print(df.describe())
    print("\nMissing values:\n", df.isnull().sum())
