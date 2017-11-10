import matplotlib.pyplot as plt
import librosa, librosa.display
from IPython.display import display, Audio
from os import listdir, path
import numpy as np

#open if exist mfcc.npy
mfccFilename = "mfcc.npy"
if path.isfile(mfccFilename):
    print("Opening " + mfccFilename)
    mfccList = list(np.load(mfccFilename))
else:
    print(mfccFilename + " not found, creating new")
    mfccList = []

#open if exist samples.npy
samplesFilename = "samples.npy"
if path.isfile(samplesFilename):
    print("Opening " + samplesFilename)
    samplesList = list(np.load(samplesFilename))
else:
    print(samplesFilename + " not found, creating new")
    samplesList = []
    

for filename in listdir("samples"):
    sample = "samples/" + filename
    y, sr = librosa.load(sample)
    # Mean amplitude (i gues) print(np.mean(np.absolute(y)))

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
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=22050, hop_length=22050)
    total_zc = sum(librosa.core.zero_crossings(y))
    #print(zcr)
    #print(total_zc)

    if filename not in samplesList:
        print("Saving " + filename + " data")
        mfccList.append(mfcc_1d_vector)
        mfccList.append(zcr)
        mfccList.append(tempo)
        mfccList.append(cent_mean)
        mfccList.append(total_zc)
        samplesList.append(filename)


np.save(mfccFilename, mfccList)
np.save(samplesFilename, samplesList)

#plt.figure()
#librosa.display.specshow(mfcc, x_axis='time')
#plt.colorbar()
#plt.title('Monophonic')

#plt.show()
