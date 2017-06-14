'''
File name add_noise.py
Author : Du Xingjian
'''

import os
from random import randint
import numpy as np
import soundfile as sf
import librosa
from scipy import stats

RE_TIMES = 1
SR = 16000
OUT_DIR = './MIX_AUDIO'
SNR = 20

if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

NOISE_DIR = ['/home/diggerdu/acoustic-simulator/noise-samples/ambience-transportation']
VOICE_DIR = ['/home/diggerdu/dataset/VCTK-Corpus/wav48/p226']

NOISE_FILE = list()
VOICE_FILE = list()

for Dir in NOISE_DIR:
    NOISE_FILE += [os.path.join(Dir, f) for f in os.listdir(Dir) if f.endswith('.wav')]

for Dir in VOICE_DIR:
    VOICE_FILE += [os.path.join(Dir, f) for f in os.listdir(Dir) if f.endswith('.wav')]

# Load Audio
NOISE = dict()
VOICE = dict()

def load_data(data_dict, file_list):
    for f in file_list:
        data, samplerate = sf.read(f)
        print(f)
        try:
            assert samplerate == 16000 and len(data.shape) == 1
        except AssertionError:
            data = librosa.resample(data, samplerate, SR)
        data_dict.update({f.split('/')[-1][:-4]:data})

load_data(NOISE, NOISE_FILE)
load_data(VOICE, VOICE_FILE)



def mix_data(clean, clean_fn, noise_dict, out_dir, snr):
    length = clean.shape[0]
    for fn, data in noise_dict.items():
        tmp = data
        if data.shape[0] > clean.shape[0]:
            start = randint(0, data.shape[0]-length)
            tmp = data[start: start+length]
        if data.shape[0] < clean.shape[0]:
            end = length - data.shape[0]
            tmp = np.concatenate((data, data[:end]))
        try:
            assert tmp.shape[0] == length
        except AssertionError:
            print('length not equal')
        noise_amp = np.mean(np.square(clean)) / np.power(10, (snr / 10.))
        scale = np.sqrt(noise_amp / np.mean(np.square(tmp)))
        print(scale)
        output = tmp * scale + clean
        measure_snr = stats.signaltonoise(output)
        sf.write(os.path.join(out_dir, clean_fn+'_'+fn+'.wav'), output, SR)
        print(measure_snr, clean_fn+'_'+fn+'.wav')


for fn, data in VOICE.items():
    for i in range(RE_TIMES):
        mix_data(data, fn, NOISE, OUT_DIR, SNR)






