import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 99999999, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.00002, "Learning rate of for adam [0.00002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [32]")
flags.DEFINE_integer("sample_size", 16, "The size of batch images [32]")
flags.DEFINE_integer("image_size", 512, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 512, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 2, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        dcgan = DCGAN(sess, batch_size=FLAGS.batch_size, sample_size = FLAGS.sample_size)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)

        if FLAGS.visualize:
            # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
            #                               [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
            #                               [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
            #                               [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
            #                               [dcgan.h4_w, dcgan.h4_b, None])

            # Below is codes for visualization
            OPTION = 2
            visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
