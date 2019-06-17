# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
H = 50
BATCH_SIZE = 100
DROP_OUT_RATE = 0.5
LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSIRON = 1e-08
STDDEV = 0.1
BIAS = 0.1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=STDDEV)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(BIAS, shape=shape)
    return tf.Variable(initial)


x = tf.placeholder(tf.float32, [None, 784])
W = weight_variable((784, H))
b1 = bias_variable([H])

h = tf.nn.softsign(tf.matmul(x, W) + b1)
keep_prob = tf.placeholder("float")
h_drop = tf.nn.dropout(h, keep_prob)

W2 = tf.transpose(W)
b2 = bias_variable([784])
y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)

loss = tf.nn.l2_loss(y - x) / BATCH_SIZE

tf.summary.scalar("l2_loss", loss)

adam = tf.train.AdamOptimizer(
    learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2, epsilon=EPSIRON)

train_step = adam.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
summary_writer = tf.summary.FileWriter(
    "summary/l2_loss", graph_def=sess.graph_def)

for step in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={
             x: batch_xs, keep_prob: (1 - DROP_OUT_RATE)})

    summary_op = tf.summary.merge_all()
    summary_str = sess.run(summary_op, feed_dict={x: batch_xs, keep_prob: 1.0})
    summary_writer.add_summary(summary_str, step)
    if step % 100 == 0:
        print(loss.eval(session=sess, feed_dict={x: batch_xs, keep_prob: 1.0}))

N_COL = 10
N_ROW = 2
plt.figure(figsize=(N_COL, N_ROW * 2.5))
batch_xs, _ = mnist.train.next_batch(N_COL * N_ROW)

for row in range(N_ROW):
    for col in range(N_COL):
        i = row * N_COL + col
        data = batch_xs[i:i + 1]

        # Draw Input Data(x)
        plt.subplot(2*N_ROW, N_COL, 2*row*N_COL+col+1)
        plt.title('IN:%02d' % i)
        plt.imshow(data.reshape((28, 28)), cmap="magma",
                   clim=(0, 1.0), origin='upper')
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")

        # Draw Output Data(y)
        plt.subplot(2*N_ROW, N_COL, 2*row*N_COL + N_COL+col+1)
        plt.title('OUT:%02d' % i)
        y_value = y.eval(session=sess, feed_dict={x: data, keep_prob: 1.0})
        plt.imshow(y_value.reshape((28, 28)), cmap="magma",
                   clim=(0, 1.0), origin='upper')
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")

plt.savefig("result.png")
plt.show()
