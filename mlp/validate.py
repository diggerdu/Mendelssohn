import tensorflow as tf
import numpy as np
import librosa

## hyper parameter
n_input = 257
n_output = 256
n_hidden = 256
n_h_layer = 16
learning_rate = 0.00001
batch_size = 256
train_iters = np.inf
save_step = 30000

input_data = np.log1p(np.load("./data/train.npy").T[::, 0:n_input]) / 4.9
target_data= np.log1p(np.load("./data/train.npy").T[::, n_input:]) / 1.1


with tf.device('/gpu:0'):
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])

    op_list = list()
    op_list.append(x)
    
    w_input = tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1))
    b_input = tf.Variable(tf.constant(0., shape=[n_hidden]))
    op_list.append(tf.nn.relu(tf.matmul(op_list[-1], w_input) + b_input))
    for i in range(n_h_layer):
        w = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[n_hidden]))
        op_list.append(tf.nn.relu(tf.matmul(op_list[-1], w) + b))
    
    w_output = tf.Variable(tf.truncated_normal([n_hidden, n_output], stddev=0.1))
    b_output = tf.Variable(tf.constant(0., shape=[n_output]))
    op_list.append(tf.sigmoid(tf.matmul(op_list[-1], w_output) + b_output))
    loss = tf.reduce_mean(tf.multiply(y, tf.abs(tf.subtract(y, op_list[-1]))))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
step = 1
saver.restore(sess, "./checkpoint/mlp_model")

print 'model prepared'

'''
FILE_PATH = './data/spring_8k.wav'
FFT_SIZE = 512
MAX_ITERS = 2000
SAMPLE_RATE = 16000
X, fs = librosa.load(FILE_PATH, sr=8000)
print X.shape,fs
'''
low_band = input_data[:int(24*16000/1024),:n_input]

high_band = sess.run(op_list[-1], feed_dict = {x:low_band})
a = np.hstack((np.expm1(low_band*4.9), np.expm1(high_band*1.1))).T
print np.max(a)
p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
#print p.shape
for i in range(5000):
    S = a * np.exp(1j*p)
    X = librosa.istft(S)
    print np.max(X)
    p = np.angle(librosa.stft(X, 1024))

OUTPUT_PATH = './data/spring_16k_recon.wav'
np.save('fallback.npy',X)
librosa.output.write_wav(OUTPUT_PATH, y=X.astype(np.int16), sr=16000)
