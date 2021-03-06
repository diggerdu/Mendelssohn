from stft import *
import scipy.io.wavfile as wave
import numpy as np

def wav2spec(file_path, file_name):
    (rate,rawData) = wave.read(file_path + file_name)
    rawData = rawData
    rawData_half = rawData[0:rawData.shape[0]/10]
    del rawData
    a = np.abs(stft(rawData_half, fftsize=1024))
    '''
    a = np.concatenate(map(lambda x: np.expand_dims(x, axis=-1), 
        (np.real(stft(rawData_half, fftsize=1024)), np.imag(stft(rawData_half, fftsize=1024)))), axis=-1)
    '''
    return a 


file_path = "../Suites-for-solo-cello/"
file_names = ["mo1_16k.wav","mo2.wav"]

np.save("im_mo1_16k.npy", wav2spec(file_path, file_names[0]))
