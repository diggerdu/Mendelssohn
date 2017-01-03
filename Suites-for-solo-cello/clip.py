import scipy.io.wavfile as wave
import numpy as np

rate, rawData = wave.read("mo1_16k.wav")
wave.write("spring_16k.wav", 16000, rawData[1000:16000*60])
