import tensorflow as tf
import numpy as np
import os
from model import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## hyper parameter
n_input = 257
n_output = 256
n_len = 32
learning_rate = 0.0001
batch_size = 256
train_iters = np.inf
save_step = 30000
## load data
input_data = np.log1p(np.load("./data/train.npy")[0:n_input, ::]) / 12.0
target_data= np.log1p(np.load("./data/train.npy")[n_input:, ::]) / 8.0

M = HQModel(in_size=[None, n_input, n_len, 1], ou_size=[None, n_output, n_len], \
        lr=learning_rate)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
step = 1
saver.restore(sess, "./checkpoint/no_deconv_model_0")
while step * batch_size < train_iters: 
    idx = np.random.choice(input_data.shape[0] - n_len  , batch_size)
    batch_input = np.expand_dims(np.asarray([input_data[::,i:i+n_len] for i in idx.tolist()]),\
            axis = -1)
    batch_target = np.asarray([target_data[::,i:i+n_len] for i in idx.tolist()])
    print batch_input.shape
    print batch_target.shape
    assert batch_input.shape == tuple([batch_size, n_input, n_len, 1])
    assert batch_target.shape == tuple([batch_size, n_output, n_len])
    loss_, _ = sess.run([M.loss, M.opt], feed_dict={M.input:batch_input, \
            M.target:batch_target})

    print ('at epoch {}, loss is {}'.format(step, loss_))
    if step % save_step == 0:
        saver.save(sess, "./checkpoint/no_deconv_model_0")
    step += 1 
