import librosa
import os
import numpy as np
import scipy.io.wavfile as wave

N_FFT = 1024
AUDIO_PATH = '../Suites-for-solo-cello/mo1_16k.wav'
OUTPUT_PATH  = './data/train.npy'
fs, X = wave.read(AUDIO_PATH) 
print fs
print np.max(X)
print X.shape
S = np.abs(librosa.stft(X, N_FFT))
print np.max(S)


np.save(OUTPUT_PATH, S)

