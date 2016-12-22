from stft import *
import scipy.io.wavfile as wave
import numpy as np
import shutil, os



def spec2wav(Spec):
    Spec = 36.00 * Spec
    return istft(Spec, scale = 30000.000)



if __name__ == '__main__':
    data_real = np.load("./data/real.npy")
    data_imag = np.load("./data/imag.npy")
    
    data = data_real + 1j*data_imag
    wave.write("data/spring_16k_reconstruct.wav", 16000, spec2wav(data))


