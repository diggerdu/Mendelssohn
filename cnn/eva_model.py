from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import librosa
import scipy.io.wavfile as wave
from scikits.samplerate import resample

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('ckpt_name', type=str)
ckpt_name = parser.parse_args().ckpt_name

# Network Parameters
n_input = 257
n_output = 256
n_len = 32

#audio coefficient
w_len = 0.032
w_step = 0.032
sample_rate = 8000


class Predictor():
    def __init__(self, ckpt_name):
        saver = tf.train.import_meta_graph(ckpt_name+'.meta')
        self.sess = tf.Session()
        saver.restore(self.sess, ckpt_name)
        self.input_op = self.sess.graph.get_operation_by_name('model_input').outputs[0]
        self.eva_op = self.sess.graph.get_operation_by_name('eva_op').outputs[0]
        self.debug_op = self.sess.graph.get_operation_by_name('model_output/G_output/Conv2d_transpose/Relu').outputs[0]

    def expand(self, audio):
        ori_len = audio.shape[0]
        tmp = resample(audio, r=0.5, type='sinc_best')
        down_len = tmp.shape[0]
        tmp = resample(tmp, r=(ori_len+1) / float(down_len), type='sinc_best')
        tmp = librosa.stft(audio, 1024)
        phase = np.divide(tmp, np.abs(tmp))
        spec_input = np.abs(librosa.stft(audio, 1024))[0:n_input, ::]
        spec_input = spec_input[::, 0:spec_input.shape[1]//n_len*n_len]
        spec_input = np.split(spec_input,
                              spec_input.shape[1]//n_len, axis=1)
        spec_input = np.asarray(spec_input)
        spec_input = np.expand_dims(spec_input, axis=-1)
        feed_dict = {self.input_op: np.log1p(spec_input) / 12.0}
        debug = self.sess.run(self.debug_op, feed_dict=feed_dict)
        np.save('debug.npy', debug)
        S = self.sess.run(self.eva_op, feed_dict=feed_dict)
        S[S >= 5e3] = 5e3
        S[S <= 0] = 0
        print ('mean', np.mean(S))
        print (np.sum(np.isinf(S)))
        S = np.squeeze(np.concatenate(np.split(S, S.shape[0]), axis=2),
                       axis=(0, -1))
        phase = phase[..., :S.shape[1]]
        print (phase.shape)
        print (S.shape)
        print (np.sum(np.isinf(np.multiply(S, phase))))

        X = librosa.istft(np.multiply(S, phase))
        return X

    def test(self):
        return 'I love heqinglin'

if __name__ == '__main__':
    p = Predictor(ckpt_name)
    print ('model prepared')
    au_path = 'data/train.wav'
    sr, y = wave.read(au_path)
    ry = p.expand(y).astype(np.int17)
    wave.write('re_train.wav', sr, ry)
