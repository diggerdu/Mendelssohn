import librosa
import os



N_FFT = 512
AUDIO_PATH = '../Suites-for-solo-cello/mo1_16k.wav'
OUTPUT_PATH  = './data/train.npy'
X, fs = librosa.load(AUDIO_PATH)
S = np.abs(librosa.stft(X, 512))



np.save(OUTPUT_PATH, S)

