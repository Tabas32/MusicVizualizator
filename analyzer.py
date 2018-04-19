import matplotlib.pyplot as plt
import librosa, librosa.display
from IPython.display import display, Audio
from os import listdir, path
import numpy as np
import pandas as pd
   

NEW_ROW_INDEX = [
        'Sample_name', 
        'Mfcc', 
        'Tempo', 
        'Mean_spectral_centroids', 
        'Zero_crossing_rate', 
        'Total_zero_crossings'
        ]

def analyzeByName(song_name):
    sample = "samples/" + song_name + ".wav"
    print('Scaning ' + sample)

    analyzed = []
    try:
        y, sr = librosa.load(sample)
    except:
        raise ValueError("Something wrong with load of " + sample)
    
    # MFCC
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=22050, n_fft=22050)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=20)
    mfcc_1d_vector = mfcc.flatten()

    analyzed.extend(mfcc_1d_vector)

    # TEMPO
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo = librosa.beat.estimate_tempo(onset_env, sr=sr)

    analyzed.append(tempo)

    # SPECTRAL CENTROIDS
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = np.mean(cent)

    analyzed.append(cent_mean)

    # ZERO CROSSING RATE
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=22050, hop_length=22050).flatten()
    total_zc = sum(librosa.core.zero_crossings(y))

    analyzed.extend(zcr)
    analyzed.append(total_zc)
    return np.array(analyzed)

def analyzeLoadedSong(y):
    # MFCC
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=22050, n_fft=22050)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=20)
    mfcc_1d_vector = mfcc.flatten()

    analyzed.extend(mfcc_1d_vector)

    # TEMPO
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo = librosa.beat.estimate_tempo(onset_env, sr=sr)

    analyzed.append(tempo)

    # SPECTRAL CENTROIDS
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = np.mean(cent)

    analyzed.append(cent_mean)

    # ZERO CROSSING RATE
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=22050, hop_length=22050).flatten()
    total_zc = sum(librosa.core.zero_crossings(y))

    analyzed.extend(zcr)
    analyzed.append(total_zc)
    return np.array(analyzed)	

def musicToCsv():
    # open features data file
    try:
        features_df = pd.read_csv('data.csv', index_col=0)
        print('data.csv opened')
    except FileNotFoundError:
        print("data.csv not found, making new")
        features_df = pd.DataFrame({
            'Sample_name':[], 
            'Mfcc':[], 
            'Tempo':[], 
            'Mean_spectral_centroids':[], 
            'Zero_crossing_rate':[],
            'Total_zero_crossings':[]
            })
        features_df.to_csv('data.csv')

    for filename in listdir("samples"):
        sample = "samples/" + filename
        print('Scaning ' + sample)

        if filename not in features_df.Sample_name.tolist():
            y, sr = librosa.load(sample)
            
            # MFCC
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=22050, n_fft=22050)
            log_S = librosa.logamplitude(S, ref_power=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=20)
            mfcc_1d_vector = mfcc.flatten()

            # TEMPO
            onset_env = librosa.onset.onset_strength(y, sr=sr)
            tempo = librosa.beat.estimate_tempo(onset_env, sr=sr)

            # SPECTRAL CENTROIDS
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            cent_mean = np.mean(cent)

            # ZERO CROSSING RATE
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=22050, hop_length=22050).flatten()
            total_zc = sum(librosa.core.zero_crossings(y))

            new_row = [
                    filename,
                    mfcc_1d_vector,
                    tempo,
                    cent_mean,
                    zcr,
                    total_zc
                    ]

            features_df = features_df.append(pd.Series(new_row, index=NEW_ROW_INDEX), ignore_index=True)

    print('Saving dataframe to data.csv')
    features_df.to_csv('data.csv')
