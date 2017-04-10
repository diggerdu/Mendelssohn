import numpy as np
import scipy.io.wavfile as wave
import librosa

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('au_0', type=str)
parser.add_argument('au_1', type=str)
path_0 = parser.parse_args().au_0
path_1 = parser.parse_args().au_1

_, y0 = wave.read(path_0)
_, y1 = wave.read(path_1)


y0 = librosa.util.fix_length(y0, min(y0.shape[0], y1.shape[0]))
y1 = librosa.util.fix_length(y1, min(y0.shape[0], y1.shape[0]))

print np.max(y0)
print np.max(y1)

print librosa.stft(y0).shape
print librosa.stft(y1).shape
print 'Pearson', np.linalg.norm(y0-y1)/y0.shape[0]
print 'LSD', np.linalg.norm(np.log1p(np.square(librosa.stft(y0))) -
                            np.log1p(np.square(librosa.stft(y1))))/y0.shape[0]



