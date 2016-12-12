from stft import *
import scipy.io.wavfile as wave
import numpy as np
import shutil, os



def spec2wav(logSpec):
    Spec = np.multiply(np.sign(logSpec), np.power(10, np.multiply(np.abs(logSpec), 6)+1) - 10)
    Spec = Spec[...,0] + 1j*Spec[...,1]
    return istft(Spec)



if __name__ == '__main__':
    data = np.load("train_669_0014.npy")
    print data.shape
    shutil.rmtree("Suites_for_Cello", ignore_errors=True)
    os.mkdir("Suites_for_Cello")
    for i in range(data.shape[0]):
        wave.write("./Suites_for_Cello/BWV{0}_Bach.wav".format(i), 16000, spec2wav(data[i]))

'''
data = np.load("mo1_16k.npy")
wave.write("test.wav", 16000, spec2wav(data[1213:1213+512,:512]))
'''
