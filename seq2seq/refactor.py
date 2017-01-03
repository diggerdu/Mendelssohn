import tensorflow as tf
import numpy as np


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



## hyper parameter
NUM_GPUS = 2
n_input = 257
n_output = 256
n_hidden = 256
n_h_layer = 16
learning_rate = 0.0001
batch_size = 4096
train_iters = np.inf
save_step = 30000
## load data
input_data = np.load("../feature_extraction/mo1_16k.npy")[::, 0:n_input]
input_data = np.reshape(input_data, (-1, n_input)) / 4000000.00
target_data= np.load("../feature_extraction/mo1_16k.npy")[::, n_input:]
target_data = np.reshape(target_data, (-1, n_output)) / 60000.00
inputs = tf.placeholder("float", [batch_size, n_input])
targets = tf.placeholder("float", [batch_size, n_output])

def mlp(x, y):
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
    loss = tf.nn.l2_loss(tf.sub(op_list[-1], y))
    return op_list[-1], loss
    
with tf.device('/gpu:0'):
    logits, loss = mlp(inputs, targets)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

'''
tower_grads = list()
for i in range(NUM_GPUS):
    with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % ('tower', i)):
            logits, loss = mlp(inputs, targets)
            tf.get_variable_scope().reuse_variables()
            grads = optimizer.compute_gradients(loss)
            tower_grads.append(grads)

apply_gradient_op = optimizer.apply_gradients(average_gradients(tower_grads))
'''

init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
step = 1

saver.restore(sess, "./checkpoint/model_0")
while step * batch_size < train_iters: 
    idx = np.random.choice(input_data.shape[0], batch_size)
    batch_inputs = input_data[idx, ::]
    batch_targets = target_data[idx, ::]
    _, loss_ = sess.run([optimizer, loss], \
		feed_dict={inputs:batch_inputs, targets:batch_targets})
    print ('at epoch {}, loss is {}'.format(step, loss_))
    if step % save_step == 0:
        saver.save(sess, "./checkpoint/model_0")
    step += 1 
