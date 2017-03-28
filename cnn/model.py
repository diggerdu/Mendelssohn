########################
# File Name : model.py
# Authot : Du Xingjian
########################

import numpy as np
import tensorflow as tf
from ops import *

class HQModel(object):
    def __init__(self, in_size, ou_size, lr):
        self.o_size = ou_size
        self.i_size = in_size
        self.lr = lr
        self.input = tf.placeholder(tf.float32, shape=in_size, name='model_input')
        self.target = tf.placeholder(tf.float32, shape=ou_size, name='model_target')
        self.build()
    def build(self):
        self.output = self._generator(self.input, name='model_output')
        self.eva_op = tf.concat(1, \
                (tf.exp(self.input*12.0)-1, tf.exp(self.output*8.0)-1), name='eva_op')
        assert ten_sh(self.output) == ten_sh(self.target)
        self.loss = self._get_loss('model_loss') 
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss) 
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
            co_5 = tf.nn.relu(conv2d(co_4, o_dim=self.o_size[-2]*self.o_size[-1],\
                k_size=[3,1], st=[4,2], name='co_5'))
            co_5_shape = co_5.get_shape().as_list()
            assert co_5_shape[1] == 1 and co_5_shape[2] == 1
            G_output = tf.reshape(tf.squeeze(co_5), [-1, self.o_size[-2], self.o_size[-1]])
            return G_output
    
    def _get_loss(self, name):
        return tf.reduce_mean(tf.multiply(self.target/tf.reduce_mean(self.target),
                              tf.div(tf.log1p(self.target), tf.log1p(self.output))),
                              name=name)

        '''
        return tf.reduce_mean(tf.multiply(tf.log1p(self.target/tf.reduce_mean(self.target)),
                              tf.abs(tf.subtract(self.target, self.output))),
                              name=name)

        return tf.reduce_mean(tf.multiply(self.target/tf.reduce_mean(self.target),
                              tf.abs(tf.subtract(self.target, self.output))),
                              name=name)
        return tf.reduce_mean(tf.multiply(1.0,
                              tf.abs(tf.subtract(self.target, self.output))),
                              name=name)
        '''

