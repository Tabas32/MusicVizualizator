import pandas as pd
import numpy as np

# open features data file and return pandas dataframe
def openDataFile(file_name):
    features_df = None
    try:
        features_df = pd.read_csv(file_name)
        print(file_name + " opened")
    except FileNotFoundError:
        print(file_name + " not found")
        quit()

    return features_df

def getAcusticNp(data_csv_file_name):
    dataframe = openDataFile(data_csv_file_name)
    featuresNames = [
        'Mfcc', 
        'Tempo', 
        'Mean_spectral_centroids', 
        'Zero_crossing_rate',
        'Total_zero_crossings'
    ]

    acusticNp = []

    for i in range(len(dataframe.index)):
        row = []
        for col in featuresNames:
            value = dataframe[col][i]
            #TODO : parse string to proper list
            if type(value) is not str:
                row.append(value)
        acusticNp.append(row)

    return np.array(acusticNp)


def makeClrClassNpFile(data_csv_file_name):
    dataframe = openDataFile(data_csv_file_name)
    #TODO

print(getAcusticNp('data.csv'))
