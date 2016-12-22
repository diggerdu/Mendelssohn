import tensorflow as tf
import numpy as np


## hyper parameter
n_input = 257
n_output = 256
n_hidden = 256
n_h_layer = 8
learning_rate = 0.00001
batch_size = 256
train_iters = np.inf
save_step = 30000

## load data
inputs= np.load("./data/spring_8k.npy")[..., 1]


with tf.device('/cpu:0'):
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])

    op_list = list()
    op_list.append(x)
    
    w_input = tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1))
    b_input = tf.Variable(tf.constant(0., shape=[n_hidden]))
    op_list.append(tf.matmul(op_list[-1], w_input) + b_input)
    for i in range(n_h_layer):
        w = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[n_hidden]))
        op_list.append(tf.matmul(op_list[-1], w) + b)
    
    w_output = tf.Variable(tf.truncated_normal([n_hidden, n_output], stddev=0.1))
    b_output = tf.Variable(tf.constant(0., shape=[n_output]))
    op_list.append(tf.matmul(op_list[-1], w_output) + b_output)
    #loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(op_list[-1], y)))
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, './checkpoint/model_1')
pred_data = sess.run(op_list[-1], feed_dict={x:inputs})
np.save("./data/imag.npy", np.hstack((inputs, pred_data)))

