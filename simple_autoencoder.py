# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, bias):
    initial = tf.constant(bias, shape=shape)
    return tf.Variable(initial)


def fit(H=50, BATCH_SIZE=100, DROP_OUT_RATE=0.5, LEARNING_RATE=0.001, BETA1=0.9, BETA2=0.999, STDDEV=0.1, BIAS=0.1, EPOCH=500, EPSILON=1e-08):
    x = tf.placeholder(tf.float32, [None, 784])
    W = weight_variable((784, H), STDDEV)
    b1 = bias_variable([H], BIAS)

    h = tf.nn.softsign(tf.matmul(x, W) + b1)
    keep_prob = tf.placeholder("float")
    h_drop = tf.nn.dropout(h, keep_prob)

    W2 = tf.transpose(W)
    b2 = bias_variable([784], BIAS)
    y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)

    loss = tf.nn.l2_loss(y - x) / BATCH_SIZE

    tf.summary.scalar("l2_loss", loss)

    adam = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2, epsilon=EPSILON
    )

    train_step = adam.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(
        "summary/l2_loss", graph_def=sess.graph_def
    )

    for step in range(EPOCH):
        batch_xs, _ = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={
            x: batch_xs,
            keep_prob: (1 - DROP_OUT_RATE)
        })

        summary_op = tf.summary.merge_all()
        summary_str = sess.run(summary_op, feed_dict={
            x: batch_xs,
            keep_prob: 1.0
        })
        summary_writer.add_summary(summary_str, step)

    loss_val = sess.run([loss], feed_dict={
        x: batch_xs, keep_prob: 1.0})
    return loss_val


if __name__ == "__main__":
    loss = fit()
    print("Loss: {}".format(loss))
