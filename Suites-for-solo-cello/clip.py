import scipy.io.wavfile as wave
import numpy as np

rate, rawData = wave.read("mo1_8k.wav")
wave.write("clip_8k.wav", 8000, rawData[1000:22100*60])
