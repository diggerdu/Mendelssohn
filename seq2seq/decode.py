from stft import *
import scipy.io.wavfile as wave
import numpy as np
import shutil, os



def spec2wav(Spec):
    return istft(Spec, scale = 1)



if __name__ == '__main__':
    data = np.load("./data/spring_16k_recon.npy")
    

    wave.write("data/spring_16k_reconstruct.wav", 16000, spec2wav(data))


