# -*-coding:UTF-8-*-
from __future__ import print_function

import os

import tensorflow as tf
from selftf.lib.ml_job_framework import MLJobFramework
from selftf.tf_job.alexnet_imagenet.alexnet_model import classifier
import selftf.tf_job.alexnet_imagenet.alexnet_train_util as tu

class AlexNet_imagenet(MLJobFramework):
    def __init__(self):
        MLJobFramework.__init__(self)
        self.train_img_path = None
        self.num_features = 0

        self.y_ = None
        self.x = None
        self.wnid_labels = None
        self.observe_loss = None

        tf.app.flags.DEFINE_string("data_dir", "", "")
        tf.app.flags.DEFINE_integer("num_class", 8,"")

    def model_definition(self, context):
        """
        :param selftf.lib.tf_program_util.TFProgramUtil context:
        :return:
        """
        num_class = context.FLAGS.num_class
        num_partition = context.get_n_partition()

        lmbda = 5e-04

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(1), trainable=False)
        context.set_global_step(global_step)
        self.wnid_labels = tu.get_winds(num_class, os.path.join(context.FLAGS.data_dir, "train"))

        # input images
        with tf.name_scope('input'):
            # None -> batch size can be any size, [224, 224, 3] -> image
            self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="x-input")
            # target 1000 output classes
            self.y_ = tf.placeholder(tf.float32, shape=[None, num_class], name="y-input")

        # creat an AlexNet
        pred, _ = classifier(self.x, 0.5, num_class, num_partition)

        # specify cost function
        # cross-entropy and weight decay
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y_,
                                                        name='cross-entropy'))

        with tf.name_scope('l2_loss'):
            l2_loss = tf.reduce_sum(lmbda * tf.stack(
                [tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
            tf.summary.scalar('l2_loss', l2_loss)

        with tf.name_scope('loss'):
            loss = cross_entropy + l2_loss
            tf.summary.scalar('loss', loss)

        context.set_train_op(loss=loss)

    def get_feed_dict(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        batch_size = context.get_batch_size()

        val_x, val_y = tu.read_batch(batch_size, context.FLAGS.data_dir +"/train/", self.wnid_labels)
        return {self.x: val_x,
                self.y_: val_y,
               }


model = AlexNet_imagenet()
model.run()











































