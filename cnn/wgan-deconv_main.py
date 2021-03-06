import tensorflow as tf
import numpy as np
import os
from wgan_model import *
from ops import *
from six.moves import xrange
os.environ["CUDA_VISIBLE_DEVICES"]="0"

  
## hyper parameter
n_input = 257
n_output = 256
n_len = 32
learning_rate = 1e-4
batch_size = 128
train_iters = 10000000000 
C_iters = 3
save_step = 30000
clamp_lower = -10
clamp_upper = 10

log_dir = './logs'
ckpt_dir = './checkpoint'

## load data
input_data = np.log1p(np.load("./data/train.npy")[0:n_input, ::]) / 12.0
target_data= np.log1p(np.load("./data/train.npy")[n_input:, ::]) / 8.0
input_data = np.expand_dims(input_data, axis=-1)
target_data = np.expand_dims(target_data, axis=-1)

def next_batch():
    idx = np.random.choice(input_data.shape[0] - n_len  , batch_size)
    batch_input = np.asarray([input_data[::,i:i+n_len] for i in idx.tolist()])
    batch_target = np.asarray([target_data[::,i:i+n_len] for i in idx.tolist()])
    assert batch_input.shape == tuple([batch_size, n_input, n_len, 1])
    assert batch_target.shape == tuple([batch_size, n_output, n_len, 1])
    return batch_input, batch_target

M = WGAN(in_size=[None, n_input, n_len, 1], \
            ou_size=[None, n_output, n_len, 1],\
            c_lr=learning_rate,\
            g_lr=learning_rate,\
            b_s=batch_size,\
            cl_l=clamp_lower,\
            cl_u=clamp_upper
            )
def next_batch(M):
    idx = np.random.choice(input_data.shape[0] - n_len  , batch_size)
    batch_input = np.asarray([input_data[::,i:i+n_len] for i in idx.tolist()])
    batch_target = np.asarray([target_data[::,i:i+n_len] for i in idx.tolist()])
    assert batch_input.shape == tuple([batch_size, n_input, n_len, 1])
    assert batch_target.shape == tuple([batch_size, n_output, n_len, 1])
    return {M.input:batch_input, M.target:batch_target}


init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
#saver.restore(sess, os.path.join(ckpt_dir, "wgan_model-14999")) 
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

for i in xrange(train_iters):
    print ('{} epoch'.format(i))
    if i < 25 or i % 100 == 0:
        c_iters = 8
    else:
        c_iters = C_iters
    for j in range(c_iters):
        feed_dict = next_batch(M)
        if j % 1 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, summary = sess.run([M.c_opt, M.summary], feed_dict=feed_dict,\
                options=run_options, run_metadata=run_metadata)
            #import time
            #s = time.time()
            summary_writer.add_summary(summary, i)
            summary_writer.flush()
            #print time.time() -s 
                #summary_writer.add_run_metadata(run_metadata, \
                        #'critic_metadata {}'.format(i), i)
        else:
                sess.run(M.c_opt, feed_dict=feed_dict)                
    feed_dict = next_batch(M)
    if i % 1 == 0:
        _, summary = sess.run([M.g_opt, M.summary], feed_dict=feed_dict,\
                        options=run_options, run_metadata=run_metadata)
        summary_writer.add_summary(summary, i)
        summary_writer.flush()
                #summary_writer.add_run_metadata(run_metadata, \
            #summary_writer.add_run_metadata(run_metadata, \
                        #'generator_metadata {}'.format(i), i)
    else:
        sess.run(M.g_opt, feed_dict=feed_dict)
    if i % 1000 == 999:
        saver.save(sess, os.path.join(ckpt_dir, "wgan_model"), global_step=i) 
