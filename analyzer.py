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

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=13)

    if filename not in samplesList:
        print("Saving " + filename + " data")
        mfccList.append(mfcc)
        samplesList.append(filename)


np.save(mfccFilename, mfccList)
np.save(samplesFilename, samplesList)

#plt.figure()
#librosa.display.specshow(mfcc, x_axis='time')
#plt.colorbar()
#plt.title('Monophonic')

#plt.show()
