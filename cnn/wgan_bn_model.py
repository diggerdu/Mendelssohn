########################
# File Name : wgan_model.py
# Authot : Du Xingjian
########################

import numpy as np
import tensorflow as tf
from bn_ops import *
import tensorflow.contrib.layers as ly

class WGAN(object):
    def __init__(self, in_size, ou_size, c_lr, g_lr, b_s, cl_l, cl_u):
        self.o_size = ou_size
        self.i_size = in_size
        self.c_lr = c_lr
        self.g_lr = g_lr
        self.clamp_lower = cl_l
        self.clamp_upper = cl_u
        self.input = tf.placeholder(tf.float32, shape=in_size, name='model_input')
        self.target = tf.placeholder(tf.float32, shape=ou_size, name='model_target')
        self.build()
    def build(self):
        self.output = self._generator(self.input, name='gene')
        self.content_loss = tf.reduce_mean(tf.multiply(tf.log1p(self.output),\
                tf.abs(tf.subtract(self.target, self.output))))
        assert ten_sh(self.output) == ten_sh(self.target)
        self.concat_output  = tf.concat(1, (self.input, self.output))
        self.concat_target  = tf.concat(1, (self.input, self.target))
        self.fake_em = self._critic(self.concat_output, name='critic')
        self.true_em = self._critic(self.concat_target, name='critic', reuse=True)
        self.c_loss = tf.reduce_mean(self.fake_em - self.true_em, name='c_loss')
        self.g_loss = tf.reduce_mean(-self.fake_em, name='g_loss')
        

        ####summary####
        conntent_loss_sum = tf.summary.scalar('content_loss', self.content_loss)
        c_loss_sum = tf.summary.scalar('c_loss', self.c_loss)
        g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        img_sum = tf.summary.image('gene_img', self.concat_output, max_outputs=1)
        img_sum = tf.summary.image('tar_img', self.concat_target, max_outputs=1)
        self.summary = tf.summary.merge_all()
        ##############

        theta_g = tf.get_collection(
                         tf.GraphKeys.TRAINABLE_VARIABLES, scope='gene')
        theta_c = tf.get_collection(
                          tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        self.c_opt = ly.optimize_loss(loss=self.c_loss, learning_rate=self.c_lr,\
                optimizer=tf.train.RMSPropOptimizer,\
                variables=theta_c,\
                global_step=counter_c)
        self.g_opt = ly.optimize_loss(loss=self.g_loss, learning_rate=self.g_lr,\
                optimizer=tf.train.RMSPropOptimizer,\
                variables=theta_g,\
                global_step=counter_g)
        self.content_opt = ly.optimize_loss(loss=self.content_loss, learning_rate=self.g_lr,\
                optimizer=tf.train.RMSPropOptimizer,\
                variables=theta_g,\
                global_step=counter_g)
        clipped_c_var = [tf.assign(var, tf.clip_by_value(var, self.clamp_lower, self.clamp_upper)) \
                for var in theta_c]
        with tf.control_dependencies([self.c_opt]):
            self.c_opt = tf.tuple(clipped_c_var)
    
    def _generator(self, input_, name):
        with tf.variable_scope(name) as scope:
            co_0 = tf.nn.relu(conv2d(input_, o_dim=64, \
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

        return G_output


    def _critic(self, input_, name='critic', reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            co_0 = tf.nn.relu(conv2d(input_, o_dim=64, \
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
            em_distance = ly.fully_connected(tf.reshape(co_5, [-1, np.prod(ten_sh(co_5)[1:])]), \
                    1, activation_fn=None, scope='em_dis')
        return em_distance

    def _get_loss(self, name):
        return tf.reduce_mean(tf.multiply(tf.log1p(self.output),\
                tf.abs(tf.subtract(self.target, self.output))), name = name) 


