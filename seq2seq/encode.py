from stft import *
import scipy.io.wavfile as wave
import numpy as np

def wav2spec(file_path, file_name):
    (rate,rawData) = wave.read(file_path + file_name)
    a = np.concatenate(map(lambda x: np.expand_dims(x, axis=-1), 
        (np.real(stft(rawData, fftsize=512)), np.imag(stft(rawData, fftsize=512)))), axis=-1)
    return a


file_path = "./"
file_names = "data/spring_8k.wav"


np.save("./data/spring_8k.npy", wav2spec(file_path, file_names))
