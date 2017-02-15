import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly


def ten_sh(tensor):
    return tensor.get_shape().as_list()

def conv2d(input_, o_dim, k_size, st, name='conv2d'):
    with tf.variable_scope(name):
        init = ly.xavier_initializer_conv2d()
        output = ly.conv2d(input_, num_outputs=o_dim, kernel_size=k_size, stride=st,\
                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME',\
                weights_initializer=init)
        return output
    '''
    with tf.variable_scope(name):
        init = ly.xavier_initializer_conv2d()
        fil = tf.get_variable('co_f', k_size+\
                [ten_sh(input_)[-1], o_dim],initializer=init)
        co = tf.nn.conv2d(input_, fil, strides=[1]+st+[1], \
                padding='SAME')
        bia = tf.get_variable('co_b', [o_dim])
        co = tf.nn.bias_add(co, bia)
        return co
    '''
    

def deconv2d(input_, o_size, k_size, name='deconv2d'):
    print name, 'input', ten_sh(input_)
    print name, 'output', o_size
    assert np.sum(np.mod(o_size[1:3], ten_sh(input_)[1:3]) - [0,0]) == 0
    with tf.variable_scope(name):
        init = ly.xavier_initializer_conv2d()
        output = ly.convolution2d_transpose(input_, num_outputs=o_size[-1], \
                kernel_size=k_size, stride=np.divide(o_size[1:3], ten_sh(input_)[1:3]), \
                padding='SAME', weights_initializer=init, \
                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm)
        return output
'''
    with tf.variable_scope(name):
        init = tf.contrib.layers.xavier_initializer_conv2d()
        fil = tf.get_variable('dc_f', k_size+[o_size[-1], \
                ten_sh(input_)[-1]], initializer=init)
        dc = tf.nn.conv2d_transpose(input_, fil, \
                output_shape=o_size, strides=[1, 1, 1, 1])
        bia = tf.get_variable('dc_b', [output_shape[-1]])
        dc = tf.nn.bias_add(dc, bia)
        return dc
'''

