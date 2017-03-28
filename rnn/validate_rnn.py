import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## hyper parameter
n_step = 12  # bidirectional
n_input = 257
n_output = 256
n_hidden = 256
n_h_layer = 4
learning_rate = 0.00002
batch_size = 1024
train_iters = np.inf
save_step = 30000
## load data
input_data = np.log1p(np.load("./data/eva.npy").T[::, 0:n_input]) / 12.0
target_data= np.log1p(np.load("./data/train.npy").T[::, n_input:]) / 8.0

# with tf.device('/gpu:0'):
x = tf.placeholder("float", [None, n_step, n_input])
rnn_input = tf.transpose(x, [1, 0, 2])
rnn_input = tf.unstack(rnn_input, axis=0)
y = tf.placeholder("float", [None, n_output])

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
fw_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * n_h_layer, state_is_tuple=True)
#bw_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * n_h_layer, state_is_tuple=True)
outputs,_ = tf.nn.rnn(fw_cell, rnn_input, dtype=tf.float32)
w_output = tf.Variable(tf.truncated_normal([n_hidden, n_output], stddev=0.1))
b_output = tf.Variable(tf.constant(0., shape=[n_output]))
logits = tf.sigmoid(tf.matmul(outputs[-1], w_output) + b_output)
loss = tf.reduce_mean(tf.multiply(tf.log1p(y), tf.abs(tf.subtract(y, logits))))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

saver.restore(sess, './checkpoint/mlp_model_0')

low_band = input_data[::,:n_input]                                            
print low_band.shape
low_band_input = np.asarray([low_band[i-n_step:i] for i in range(n_step, low_band.shape[0]+1)])
print low_band_input.shape
high_band = sess.run(logits, feed_dict = {x:low_band_input})                                    
a = np.hstack((np.expm1(low_band[n_step-1:]*12.0), np.expm1(high_band*8.0))).T                             
print np.max(a)                                                                                 
p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi                                        
#print p.shape    
import librosa
import scipy.io.wavfile as wave
for i in range(500):                                                                            
    S = a * np.exp(1j*p)                                                                        
    X = librosa.istft(S)                                                                        
    print np.max(X)                                                                             
    p = np.angle(librosa.stft(X, 1024))                                                         
                                                                                                
    OUTPUT_PATH = './data/spring_16k_recon.wav'                                                     
np.save('fallback.npy',X)                                                                       
wave.write(OUTPUT_PATH, 16000, X.T.astype(np.int16))    
