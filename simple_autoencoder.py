# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


class SimpleAE(object):

    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        self.params = {
            "H": 50,
            "BATCH_SIZE": 100,
            "DROP_OUT_RATE": 0.5,
            "LEARNING_RATE": 0.001,
            "BETA1": 0.9,
            "BETA2": 0.999,
            "EPSILON": 1e-8,
            "STDDEV": 0.1,
            "BIAS": 0.1
        }

        self.loss = float("+inf")

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=self.params["STDDEV"])
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(self.params["BIAS"], shape=shape)
        return tf.Variable(initial)

    def set_param(self, params):
        self.params.update(params)

    def fit(self):
        x = tf.placeholder(tf.float32, [None, 784])
        W = self.weight_variable((764, self.params["H"]))
        b1 = self.bias_variable([self.params["H"]])

        h = tf.nn.softsign(tf.matmul(x, W) + b1)
        keep_prob = tf.placeholder("float")
        h_drop = tf.nn.dropout(h, keep_prob)

        W2 = tf.transpose(W)
        b2 = self.bias_variable([784])
        y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)

        loss = tf.nn.l2_loss(y - x) / self.params["BATCH_SIZE"]

        tf.summary.scalar("l2_loss", loss)

        adam = tf.train.AdamOptimizer(
            learning_rate=self.params["LEARNING_RATE"], beta1=self.params[
                "BETA1"], beta2=self.params["BETA2"], epsilon=self.params["EPSILON"]
        )

        train_step = adam.minimize(loss)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(
            "summary/l2_loss", graph_def=sess.graph_def
        )

        for step in range(2000):
            batch_xs, batch_ys = mnist.train.next_batch(
                self.params["BATCH_SIZE"])
            sess.run(train_step, feed_dict={
                x: batch_xs, keep_prob: (1 - self.params["DROP_OUT_RATE"])})

            summary_op = tf.summary.merge_all()
            summary_str = sess.run(summary_op, feed_dict={
                x: batch_xs, keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)
            self.loss = loss.eval(session=sess, feed_dict={x: batch_xs})

    def predict(self):
        return self.loss
