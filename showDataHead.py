import pandas as pd
import sys

# Script takes 1 argument - how many rows to write

if len(sys.argv) == 2:
    showNum = int(sys.argv[1])
else:
    showNum = 2

# open features data file
try:
    features_df = pd.read_csv('data.csv')
    print(features_df.head(showNum))
except FileNotFoundError:
    print("data.csv not found")
