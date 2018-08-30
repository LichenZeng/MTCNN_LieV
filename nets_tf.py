import numpy as np
import tensorflow as tf
from tensorflow import layers as nn


class PNet:

    def __init__(self):
        # 1. Input:
        self.x = tf.placeholder(tf.float32, shape=[None, 12, 12, 3])

    def forward(self):
        # 2. Common Networks Layers
        self.conv1 = nn.conv2d(inputs=self.x, filters=10, kernel_size=[3, 3], activation=tf.nn.relu)
        self.conv1 = nn.max_pooling2d(self.conv1, pool_size=2, strides=2)
        self.conv2 = nn.conv2d(inputs=self.conv1, filters=16, kernel_size=[3, 3], activation=tf.nn.relu)
        self.conv3 = nn.conv2d(inputs=self.conv2, filters=32, kernel_size=[3, 3], activation=tf.nn.relu)
        self.conv4_1 = nn.conv2d(inputs=self.conv3, filters=1, kernel_size=[1, 1], activation=tf.nn.sigmoid)
        self.conv4_2 = nn.conv2d(inputs=self.conv3, filters=4, kernel_size=[1, 1])


class RNet:

    def __init__(self):
        # 1. Input:
        self.x = tf.placeholder(tf.float32, shape=[None, 24, 24, 3])

    def forward(self):
        # 2. Common Networks Layers
        self.conv1 = nn.conv2d(inputs=self.x, filters=28, kernel_size=[3, 3], activation=tf.nn.relu)
        self.conv1 = nn.max_pooling2d(self.conv1, pool_size=3, strides=2)
        self.conv2 = nn.conv2d(inputs=self.conv1, filters=48, kernel_size=[3, 3], activation=tf.nn.relu)
        self.conv2 = nn.max_pooling2d(self.conv2, pool_size=3, strides=2)
        self.conv3 = nn.conv2d(inputs=self.conv2, filters=64, kernel_size=[2, 2], activation=tf.nn.relu)

        self.conv4 = nn.conv2d(inputs=self.conv3, filters=64, kernel_size=[2, 2], activation=tf.nn.relu)
        self.conv5_1 = nn.conv2d(inputs=self.conv4, filters=1, kernel_size=[1, 1], activation=tf.nn.sigmoid)
        self.conv5_2 = nn.conv2d(inputs=self.conv4, filters=4, kernel_size=[1, 1])

        # # self.conv_flat = tf.reshape(self.conv3, [self.conv3.get_shape()[0], -1])
        # self.conv_flat = tf.reshape(self.conv3, [2, -1])  # If use FC, Must need the batch number
        # self.fc1 = tf.layers.dense(inputs=self.conv_flat, units=128, activation=tf.nn.relu)
        # self.fc2_1 = tf.layers.dense(inputs=self.fc1, units=1, activation=tf.nn.sigmoid)
        # self.fc2_2 = tf.layers.dense(inputs=self.fc1, units=4)


class ONet:

    def __init__(self):
        # 1. Input:
        self.x = tf.placeholder(tf.float32, shape=[None, 48, 48, 3])

    def forward(self):
        # 2. Common Networks Layers
        self.conv1 = nn.conv2d(inputs=self.x, filters=32, kernel_size=[3, 3], activation=tf.nn.relu)
        self.conv1 = nn.max_pooling2d(self.conv1, pool_size=3, strides=2)

        self.conv2 = nn.conv2d(inputs=self.conv1, filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
        self.conv2 = nn.max_pooling2d(self.conv2, pool_size=3, strides=2)

        self.conv3 = nn.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
        self.conv3 = nn.max_pooling2d(self.conv3, pool_size=2, strides=2)

        self.conv4 = nn.conv2d(inputs=self.conv3, filters=128, kernel_size=[2, 2], activation=tf.nn.relu)

        self.conv5 = nn.conv2d(inputs=self.conv4, filters=64, kernel_size=[2, 2], activation=tf.nn.relu)

        self.conv6_1 = nn.conv2d(inputs=self.conv5, filters=1, kernel_size=[1, 1], activation=tf.nn.sigmoid)
        self.conv6_2 = nn.conv2d(inputs=self.conv5, filters=4, kernel_size=[1, 1])


if __name__ == '__main__':
    net = ONet()
    net.forward()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        x = np.random.random((2, 48, 48, 3))
        print(x.shape)
        cls, off = sess.run([net.conv6_1, net.conv6_2], feed_dict={net.x: x})
        print(cls, cls.reshape((2, 1)), cls.shape)
        print(off, off.reshape((2, 4)), off.shape)
