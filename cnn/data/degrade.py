import librosa
import numpy as np
import scipy.io.wavfile as wave

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('in_path', type=str)
in_path = parser.parse_args().in_path



sr, y = wave.read(in_path)
S = librosa.stft(y, n_fft=1024)
S[257:,::] = 0+0j
de_y = librosa.istft(S).astype(np.int16)
wave.write('de_'+in_path, sr, de_y)
print S.shape

