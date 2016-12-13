from stft import *
import scipy.io.wavfile as wave
import numpy as np

def wav2spec(file_path, file_name):
    (rate,rawData) = wave.read(file_path + file_name)
    rawData = rawData/30000.00
    a = np.concatenate(map(lambda x: np.expand_dims(x, axis=-1), 
        (np.real(stft(rawData, fftsize=256)), np.imag(stft(rawData, fftsize=256)))), axis=-1)
    return a/36.0 


file_path = "../Suites-for-solo-cello/"
file_names = ["mo1_16k.wav","mo2.wav"]

np.save("mo1_16k.npy", wav2spec(file_path, file_names[0]))
