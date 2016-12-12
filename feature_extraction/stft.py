import scipy, pylab
import numpy as np
import scipy.io.wavfile as wave

def stft(x, fftsize=1024, overlap=2):   
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]  
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])

def istft(X, overlap=2):   
    fftsize=(X.shape[1]-1)*2
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop) 
    for n,i in enumerate(range(0, len(x)-fftsize, hop)): 
        x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x.astype(np.int16)

if __name__ == '__main__':
    FILE = "spring.wav"
    fs = 8000        # sampled at 8 kHz
    framesz = 0.050  # with a frame size of 50 milliseconds
    hop = 0.025      # and hop size of 25 milliseconds.

    (rate, rawData) = wave.read(FILE)
    
    LEN = rawData.shape[0]
    spec = stft(rawData)
    
    reCon = istft(np.abs(spec).astype(np.complex64))
    print type(rawData[0])
    print type(rawData[0])
    print type(reCon[0])
    wave.write("spring_re.wav", rate, reCon)
    print type(rawData), (reCon)
    print rawData.shape, reCon.shape
