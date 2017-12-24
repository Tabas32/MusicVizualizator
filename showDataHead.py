import pandas as pd
import sys

# Script takes 2 argument - how many rows to write
#                         - if to write length of row (1 or 0)

if len(sys.argv) == 3:
    showNum = int(sys.argv[1])
    if int(sys.argv[2]) == 1:
        showLen = True
    else:
        showLen = False
else:
    showNum = 2
    showLen = False

# open features data file
try:
    features_df = pd.read_csv('data.csv')
    print(features_df.head(showNum))
    #print(features_df['Mfcc'][20][1:-1].split(' '))
    #if(showLen):
    #    for i in range(showNum):
    #       print(str(i) + ". row : " + str(features_df.iloc[i]))
except FileNotFoundError:
    print("data.csv not found")
