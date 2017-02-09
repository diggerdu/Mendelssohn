import numpy as np
import tensorflow as tf


def ten_sh(tensor):
    return tensor.get_shape().as_list()

def conv2d(input_, o_dim, k_size, st, name='conv2d'):
    with tf.variable_scope(name):
        init = tf.contrib.layers.xavier_initializer_conv2d()
        fil = tf.get_variable('co_f', k_size+\
                [ten_sh(input_)[-1], o_dim],initializer=init)
        co = tf.nn.conv2d(input_, fil, strides=[1]+st+[1], \
                padding='SAME')
        bia = tf.get_variable('co_b', [o_dim])
        co = tf.nn.bias_add(co, bia)
        return co

def deconv2d(input_, o_size, k_size, name='deconv2d'):
    with tf.variable_scope(name):
        init = tf.contrib.layers.xavier_initializer_conv2d()
        fil = tf.get_variable('dc_f', k_size+[o_size[-1], \
                ten_sh(input_)[-1]], initializer=init)
        dc = tf.nn.conv2d_transpose(input_, fil, \
                output_shape=o_size, strides=[1, 1, 1, 1])
        bia = tf.get_variable('dc_b', [output_shape[-1]])
        dc = tf.nn.bias_add(dc, bia)
        return dc


