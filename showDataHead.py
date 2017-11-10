import pandas as pd

# open features data file
try:
    features_df = pd.read_csv('data.csv')
    print(features_df.head(2))
except FileNotFoundError:
    print("data.csv not found")
