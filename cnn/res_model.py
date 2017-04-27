########################
# File Name : res_model.py
# Authot : Du Xingjian
########################

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly
from ops import ten_sh

class HQModel(object):
    def __init__(self, in_size, ou_size, lr):
        self.relu_leakiness = 0.1
        self.o_size = ou_size
        self.i_size = in_size
        self.lr = lr
        self.input = tf.placeholder(tf.float32, shape=in_size, name='model_input')
        self.target = tf.placeholder(tf.float32, shape=ou_size, name='model_target')
        self.model = 'train'
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
            co_0 = tf.nn.relu(conv2d(self.input, o_dim=64,
                              k_size=[3, 1], st=[2, 1], name='co_0'))
            co_1 = tf.nn.relu(conv2d(co_0, o_dim=128,
                              k_size=[3, 2], st=[2, 1], name='co_1'))
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
    def _res_block(self, x, o_dim, k_size, st):
        x = self._batch_norm(x, 'init_bn')
        x = self._relu(x, self.relu_leakiness)
        orig_x = x
        with tf.variable_scope('sub0'):
            x = self._conv(x, o_dim, k_size, st)
        with tf.variable_scope('sub1'):
            x = self._batch_norm(x)
            x = self._relu(x, self.relu_leakiness)
            x = self._conv(x, o_dim, k_size, [1, 1])
        with tf.variable_scope('sub_add'):
            i_dim = ten_sh(x)[-1]
            if i_dim != o_dim:
                add_st = [1] + st + [1]
                orig_x = tf.nn.avg_pool(orig_x, add_st, add_st, 'VALID')
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],
                                [(o_dim-i_dim)//2, (o_dim-i_dim)//2]])
            x += orig_x
        return x
        
    def _conv(self, input_, o_dim, k_size, st, name='conv2d'):
        with tf.variable_scope(name):
            init = ly.xavier_initializer_conv2d()
            fil = tf.get_variable('co_f', k_size + [ten_sh(input_)[-1], o_dim], initializer=init)
            co = tf.nn.conv2d(input_, fil, strides=[1]+st+[1], padding='SAME')
            bia = tf.get_variable('co_b', [o_dim])
            co = tf.nn.bias_add(co, bia)
            return co

    def _batch_norm(self, x, name):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))
            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                        'moving_variance', params_shape, tf.float32,
                        initializer=tf.constant_initializer(1.0, tf.float32),
                        trainable=False)

            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

