from stft import *
import scipy.io.wavfile as wave
import numpy as np

def wav2spec(file_path, file_name):
    (rate,rawData) = wave.read(file_path + file_name)
    a = np.concatenate(map(lambda x: np.expand_dims(x, axis=-1), 
        (np.real(stft(rawData)), np.imag(stft(rawData)))), axis=-1)
    return np.divide(np.sign(a)*(np.log10(abs(a)+10) -1), 6.00)



file_path = "../Suites-for-solo-cello/"
file_names = ["mo1_16k.wav","mo2.wav"]

np.save("mo1_16k.npy", wav2spec(file_path, file_names[0]))
