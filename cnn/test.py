import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly


img = tf.placeholder(tf.float32, shape=(8,512,512,5))
#fil = tf.get_variable(name='filter', shape=[22,12,5,10])
#test = tf.nn.conv2d(img, filter=fil, strides=[1,128,128,1], padding='SAME')
'''
biases = tf.get_variable(name = 'b', shape=test.get_shape().as_list()[-1])
test = tf.nn.bias_add(test, biases)
'''
test = ly.convolution2d_transpose(img, num_outputs=10, kernel_size=[2,3], stride=6, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
print test.get_shape().as_list()

