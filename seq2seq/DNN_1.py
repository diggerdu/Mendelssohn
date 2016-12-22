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
input_data = np.load("../feature_extraction/mo1_16k.npy")[::, 0:n_input, 1]
target_data= np.load("../feature_extraction/mo1_16k.npy")[::, n_input:, 1]

with tf.device('/gpu:1'):
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
    loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(op_list[-1], y)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
step = 1
while step * batch_size < train_iters: 
    idx = np.random.choice(input_data.shape[0], batch_size)
    batch_inputs = input_data[idx, ::]
    batch_targets = target_data[idx, ::]
    loss_, _ = sess.run([loss, optimizer], feed_dict={x:batch_inputs, y:batch_targets})
    print ('at epoch {}, loss is {}'.format(step, loss_))
    if step % save_step == 0:
        saver.save(sess, "./checkpoint/model_1")
    step += 1 
