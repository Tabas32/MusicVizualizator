import librosa
import numpy as np

def analyzeLoadedSong(y, sr):
    analyzed = []

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


def analyzeLoadedSongMini(y, sr):
    analyzed = []

    # TEMPO
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo = librosa.beat.estimate_tempo(onset_env, sr=sr)

    analyzed.append(tempo)

    # ZERO CROSSING RATE
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=22050, hop_length=22050).flatten()
    total_zc = sum(librosa.core.zero_crossings(y))

    analyzed.extend(zcr)
    analyzed.append(total_zc)
    return np.array(analyzed)	
