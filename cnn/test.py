import tensorflow as tf
import numpy as np

img = tf.placeholder(tf.float32, shape=(8,512,512,5))
fil = tf.get_variable(name='filter', shape=[22,12,5,10])
test = tf.nn.conv2d(img, filter=fil, strides=[1,128,128,1], padding='SAME')
'''
biases = tf.get_variable(name = 'b', shape=test.get_shape().as_list()[-1])
test = tf.nn.bias_add(test, biases)
'''
print test.get_shape().as_list()

