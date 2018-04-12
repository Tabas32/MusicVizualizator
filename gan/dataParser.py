import numpy as np

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
    np_song = (np_song - np.amin(stand, axis=0)) / np.ptp(stand, axis=0) -0.5

    return np_song
