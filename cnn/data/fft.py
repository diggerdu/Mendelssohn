import tensorflow as tf
import numpy as np
import scipy


def stft(wav, n_fft=1024, overlap=4, dt=tf.int32, absp=False):
    assert (wav.shape[0] > n_fft)
    X = tf.placeholder(dtype=dt,shape=wav.shape)
    X = tf.cast(X,tf.float32)
    hop = n_fft / overlap
    
    ## prepare constant variable
    Pi = tf.constant(np.pi, dtype=tf.float32)
    W = tf.constant(scipy.hanning(n_fft), dtype=tf.float32)
    S = tf.pack([tf.fft(tf.cast(tf.multiply(W,X[i:i+n_fft]),\
            tf.complex64)) for i in range(1, wav.shape[0] - n_fft, hop)])
    abs_S = tf.complex_abs(S)
    sess = tf.Session()
    if absp:
        return sess.run(abs_S, feed_dict={X:wav})
    else:
        return sess.run(S, feed_dict={X:wav})

def istft(spec, overlap=4):
    assert (spec.shape[0] > 1)
    S = placeholder(dtype=tf.complex64, shape=spec.shape)
    X = tf.complex_abs(tf.concat(0, [tf.ifft(frame) \
            for frame in tf.unstack(S)]))
    sess = tf.Session()
    return sess.run(X, feed_dict={S:spec})
if __name__ == '__main__':
    a = np.arange(300000)
    import time
    import librosa
    s = time.time()
    print stft(a).shape
    print time.time() - s
    s = time.time()
    print librosa.stft(a).shape
    print time.time() -s

