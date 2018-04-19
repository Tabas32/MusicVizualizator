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

"""
Makes dataset of skices and songs
@Returns:
    np array of data
"""
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

"""
Makes dataset of realistic images and songs
@Returns:
    np array of data
"""
def makeNpRData():
    directory = "images_I"
    data = []

    for each in os.listdir(directory):
        img = os.path.join(directory, each)
        img_arr = prImg.process_img_R(img)
        
        try:
            song = alz.analyzeByName(each[:-4])
            
            data.append([img_arr, song])
        except ValueError as err:
            print("Error: " + repr(err))

    return np.array(data)

""" 
@Parms:
    np_file: name of unnormalized data file

@Returns:
    normalized np array of data
"""
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

""" 
@Parms:
    np_file: name of unnormalized data file
    np_song: np array of analyzed song

@Returns:
    normalized np array of song
"""
def normalizeSong(np_file, np_song):
    data = np.load(np_file)
    songs = np.array(list(data[:, 1]), dtype=np.float)
    
    mean_norm_mean = np.mean(songs, axis=0)
    mean_norm = songs - mean_norm_mean
    np_song = np_song - mean_norm_mean
    
    stand = np.power(mean_norm, 2)
    stand_mean = np.mean(stand, axis=0)
    stand = np.divide(mean_norm, stand_mean)
    np_song = np.divide(np_song, stand_mean)

    norm = (stand - np.amin(stand, axis=0)) / np.ptp(stand, axis=0) -0.5
    np_song = (np_song - np.amin(stand, axis=0)) / np.ptp(np_song, axis=0) -0.5

    return np_song

#data = normalizeNpMusicData("mini_data_S_notNorm.npy")
#np.save("mini_data_S", data)

#data2 = np.load("mini_data_S.npy")
#print(data2.shape)
