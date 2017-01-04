import tensorflow as tf
import numpy as np


## hyper parameter
n_step = 12  # bidirectional
n_input = 257
n_output = 256
n_hidden = 256
n_h_layer = 4
learning_rate = 0.00002
batch_size = 768
train_iters = np.inf
save_step = 30000
## load data
input_data = np.log1p(np.load("./data/train.npy").T[:5000, 0:n_input]) / 12.0
target_data= np.log1p(np.load("./data/train.npy").T[:5000, n_input:]) / 8.0

with tf.device('/gpu:1'):
    x = tf.placeholder("float", [None, n_step, n_input])
    rnn_input = tf.transpose(x, [1, 0, 2])
    rnn_input = tf.unstack(rnn_input, axis=0)
    y = tf.placeholder("float", [None, n_output])
    
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    fw_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * n_h_layer, state_is_tuple=True)
    #bw_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * n_h_layer, state_is_tuple=True)
    outputs,_ = tf.nn.rnn(fw_cell, rnn_input, dtype=tf.float32)
    w_output = tf.Variable(tf.truncated_normal([n_hidden, n_output], stddev=0.1))
    b_output = tf.Variable(tf.constant(0., shape=[n_output]))
    logits = tf.sigmoid(tf.matmul(outputs[-1], w_output) + b_output)
    loss = tf.reduce_mean(tf.multiply(y, tf.abs(tf.subtract(y, logits))))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
step = 1
while step * batch_size < train_iters: 
    idx = np.random.choice(input_data.shape[0] - n_step*2, batch_size) + n_step
    batch_inputs = np.asarray([input_data[i+1-n_step:i+1] for i in idx.tolist()])
    print batch_inputs.shape
    batch_targets = target_data[idx]
    loss_, _ = sess.run([loss, optimizer], feed_dict={x:batch_inputs, y:batch_targets})

    print ('at epoch {}, loss is {}'.format(step, loss_))
    if step % save_step == 0:
        saver.save(sess, "./checkpoint/mlp_model_0")
    step += 1 
