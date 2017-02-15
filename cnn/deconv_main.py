import tensorflow as tf
import numpy as np
import os
from model import *
from ops import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Deco_Model(HQModel):
    def _generator(self, input_, name):
        with tf.variable_scope(name):
            tar_s = ten_sh(input_)
            co_0 = tf.nn.relu(conv2d(self.input, o_dim=64, \
                k_size=[3,1], st=[2,1], name='co_0'))
            co_1 = tf.nn.relu(conv2d(co_0, o_dim=128, \
                k_size=[3,2], st=[2,1], name='co_1'))
            co_2 = tf.nn.relu(conv2d(co_1, o_dim=256, \
                k_size=[3,2], st=[2,2], name='co_2'))
            co_3 = tf.nn.relu(conv2d(co_2, o_dim=512, \
                k_size=[3,2], st=[3,3], name='co_3'))
            co_4 = tf.nn.relu(conv2d(co_3, o_dim=1024, \
                k_size=[2,2], st=[3,3], name='co_4'))
            dc_5 = tf.nn.relu(deconv2d(co_4, o_size=[None,16,2,512], \
                    k_size=[4,2], name='dc_5'))
            dc_6 = tf.nn.relu(deconv2d(dc_5, o_size=[None,64,8,128],\
                    k_size=[4,4], name='dc_6'))
            dc_7 = tf.nn.relu(deconv2d(dc_6, o_size=[None,128,16,32],\
                    k_size=[2,2], name='dc_7'))
            G_output = tf.nn.relu(deconv2d(dc_7,\
                    o_size=self.o_size, \
                    k_size=[2,2], name='G_output'))

            assert ten_sh(G_output) == ten_sh(self.target)
            test = tf.concat(1, (self.input, G_output))
            print ten_sh(test)
            return G_output
            
   
# hyper parameter
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
input_data = np.expand_dims(input_data, axis=-1)
target_data = np.expand_dims(target_data, axis=-1)

M = Deco_Model(in_size=[None, n_input, n_len, 1], ou_size=[None, n_output, n_len, 1], lr=learning_rate)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
step = 1
while step * batch_size < train_iters: 
    idx = np.random.choice(input_data.shape[0] - n_len  , batch_size)
    batch_input = np.asarray([input_data[::,i:i+n_len] for i in idx.tolist()])
    batch_target = np.asarray([target_data[::,i:i+n_len] for i in idx.tolist()])
    #print batch_input.shape
    #print batch_target.shape
    assert batch_input.shape == tuple([batch_size, n_input, n_len, 1])
    assert batch_target.shape == tuple([batch_size, n_output, n_len, 1])
    loss_, _ = sess.run([M.loss, M.opt], feed_dict={M.input:batch_input, \
            M.target:batch_target})

    print ('at epoch {}, loss is {}'.format(step, loss_))
    if step % save_step == 0 or loss < 1e-8:
        saver.save(sess, "./checkpoint/deconv_model")
    step += 1 
