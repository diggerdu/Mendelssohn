import scipy.io.wavfile as wave 
import numpy as np
import librosa



(rate, x) = wave.read('spring_16k.wav')
print np.sum(x)
x = x.T
print np.sum(librosa.istft(librosa.stft(x),dtype=np.int32))
