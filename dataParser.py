import pandas as pd
import numpy as np
import analyzer as alz
import os
import gan.procesImages as prImg

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

def makeNpSData():
    directory = "images_S"
    data = []

    for each in os.listdir(directory):
        img = os.path.join(directory, each)
        img_arr = prImg.process_img_S(img)
        
        try:
            song = alz.analyzeByName(each[:-4])
            
            data.append([img_arr, song])
        except ValueError as err:
            print("Error: " + repr(err))

    return np.array(data)

def normalizeNpMusicData(np_file):
    data = np.load(np_file)
    songs = np.array(list(data[:, 1]), dtype=np.float)
    
    mean_norm = songs - np.mean(songs, axis=0)
    
    stand = np.power(mean_norm, 2)
    stand = np.divide(mean_norm, np.mean(stand, axis=0))

    norm = (stand - np.amin(stand, axis=0)) / np.ptp(stand, axis=0) -0.5

    for i in range(data.shape[0]):
        data[i, 1] = norm[i]

    return data

data = normalizeNpMusicData("data_S.npy")
np.save("data_S", data)

data2 = np.load("data_S.npy")
print(data2.shape)
