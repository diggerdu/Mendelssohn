import numpy as np
import librosa
def icqt_recursive(C, sr=22050, hop_length=512, fmin=None, bins_per_octave=12, filter_scale=1, norm=1, scale=True, window='hann'):
    
    n_octaves = int(np.ceil(float(C.shape[0]) / bins_per_octave))


    if fmin is None:
        fmin = librosa.note_to_hz('C1')
    
    freqs = librosa.cqt_frequencies(C.shape[0], fmin,
                            bins_per_octave=bins_per_octave)[-bins_per_octave:]

    fmin_t = np.min(freqs)
    fmax_t = np.max(freqs)

    # Make the filter bank
    f, lengths = librosa.filters.constant_q(sr=sr,
                                            fmin=fmin_t,
                                            n_bins=bins_per_octave,
                                            bins_per_octave=bins_per_octave,
                                            filter_scale=filter_scale,
                                            norm=norm, window=window)
    
    if scale:
        f = f / np.sqrt(lengths[:, np.newaxis])
    
    else:
        f = f / lengths[:, np.newaxis]
        
    n_trim = f.shape[1] // 2
    
    # Hermitian the filters and sparsify
    f = librosa.util.sparsify_rows(f)
    
    y = None
    
    for octave in range(n_octaves - 1, -1, -1):
    
        # Compute the slice index for the current octave
        slice_ = slice(-(octave+1) * bins_per_octave - 1, -(octave) * bins_per_octave - 1)
        
        # Project onto the basis        
        C_ = C[slice_]
        fb = f[-C_.shape[0]:] #/ np.sqrt(lengths[-C_.shape[0]:, np.newaxis])
        Cf = fb.conj().T.dot(C_) 
        
        # Overlap-add the responses
        y_oct = np.zeros(int(f.shape[1] + (2**(-octave) *  hop_length * C.shape[1])), dtype=f.dtype)
        for i in range(Cf.shape[1]):
            y_oct[int(i * hop_length * 2**(-octave)):int(i * hop_length * 2**(-octave) + Cf.shape[0])] += Cf[:, i]
        
        if y is None:
            y = y_oct
            continue

        # Up-sample the previous buffer and add in the new one
        y = (librosa.core.resample(y.real, 1, 2, scale=True) + 
             1.j * librosa.core.resample(y.imag, 1, 2, scale=True))
        
        y = y[n_trim:-n_trim] / 2 + y_oct
        
    # Chop down the length
    y = librosa.util.fix_length(y.real, f.shape[1] + hop_length * C.shape[1])
    
    
    y *= 2**n_octaves
    return np.ascontiguousarray(y[n_trim : -n_trim])

icqt = icqt_recursive
cqt = librosa.core.cqt

def rcqt(PS, iters=500, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12):
    sig_len = (PS.shape[1]-1)*hop_length
    p = 2 * np.pi * np.random.random_sample(PS.shape) - np.pi
    for i in range(iters):
        print i
        S = PS * np.exp(1j*p)
        X = icqt(S, sr=sr, hop_length=hop_length,
                 bins_per_octave=bins_per_octave)
        X = librosa.util.fix_length(X, sig_len)
        p = np.angle(cqt(X, sr=sr, hop_length=hop_length, n_bins=n_bins,
                     bins_per_octave=bins_per_octave))
    return X

if __name__ == '__main__':
    (Y, sr) = librosa.load('quintessence.wav', sr=44100)
    print Y.shape
    S = librosa.cqt(Y, sr=44100, hop_length=256, n_bins=9*12*3, bins_per_octave=12*3)
    print S.shape
    PS = np.abs(S)
    Y1 = rcqt(PS, iters=5000, sr=44100, hop_length=256, n_bins=9*12*3, bins_per_octave=12*3)
    print Y1.shape
    from IPython.display import Audio
    librosa.output.write_wav('re.wav', Y1, sr=44100)
