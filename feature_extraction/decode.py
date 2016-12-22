from stft import *
import scipy.io.wavfile as wave
import numpy as np
import shutil, os



def spec2wav(Spec):
    Spec = 36.00 * Spec
    Spec = Spec[...,0] + 1j*Spec[...,1]
    return istft(Spec, scale = 30000.000)



if __name__ == '__main__':
    data = np.load("train_2943_0007.npy")
    print data.shape
    shutil.rmtree("Suites_for_Cello", ignore_errors=True)
    os.mkdir("Suites_for_Cello")
    for i in range(data.shape[0]):
        wave.write("./Suites_for_Cello/BWV{0}_Bach.wav".format(i), 16000, spec2wav(data[i]))

'''
data = np.load("mo1_16k.npy")
rawData = spec2wav(data[1213:1213+12313, :128])
np.save("rawData.npy", rawData)
wave.write("test.wav", 16000, rawData)

'''
