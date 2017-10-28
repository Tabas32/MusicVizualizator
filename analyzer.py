import matplotlib.pyplot as plt
import librosa, librosa.display
from IPython.display import display, Audio

# 1. Get the file path to the included audio example
sample = "samples/sample0.wav"

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(sample)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
log_S = librosa.logamplitude(S, ref_power=np.max)
mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=13)


plt.figure()
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('Monophonic')

plt.show()
