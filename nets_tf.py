import os
import numpy as np
import tensorflow as tf
from tensorflow import layers as nn
from torch.utils.data import DataLoader
from simpling import FaceDataset


class PNet:

    def __init__(self):
        # 1. Input:
        self.x = tf.placeholder(tf.float32, shape=[None, 12, 12, 3])

        # 2. Common Networks Layers
        self.conv1 = nn.conv2d(inputs=self.x, filters=10, kernel_size=[3, 3], activation=tf.nn.relu)
        self.conv1 = nn.max_pooling2d(self.conv1, pool_size=2, strides=2)
        self.conv2 = nn.conv2d(inputs=self.conv1, filters=16, kernel_size=[3, 3], activation=tf.nn.relu)
        self.conv3 = nn.conv2d(inputs=self.conv2, filters=32, kernel_size=[3, 3], activation=tf.nn.relu)
        self.conv4_1 = nn.conv2d(inputs=self.conv3, filters=1, kernel_size=[1, 1], activation=tf.nn.sigmoid)
        self.conv4_2 = nn.conv2d(inputs=self.conv3, filters=4, kernel_size=[1, 1])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def forward(self, x):
        cls, off = self.sess.run([self.conv4_1, self.conv4_2], feed_dict={self.x: x})
        return cls.reshape(x.shape[0], 1), off.reshape(x.shape[0], 4)


class RNet:

    def __init__(self):
        # 1. Input:
        self.x = tf.placeholder(tf.float32, shape=[None, 24, 24, 3])

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

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def forward(self, x):
        cls, off = self.sess.run([self.conv5_1, self.conv5_2], feed_dict={self.x: x})
        return cls.reshape(x.shape[0], 1), off.reshape(x.shape[0], 4)


class ONet:

    def __init__(self):
        # 1. Input:
        self.x = tf.placeholder(tf.float32, shape=[None, 48, 48, 3])

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

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def forward(self, x):
        cls, off = self.sess.run([self.conv6_1, self.conv6_2], feed_dict={self.x: x})
        return cls.reshape(x.shape[0], 1), off.reshape(x.shape[0], 4)


class Trainer:
    def __init__(self, net, save_path, dataset_path):
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path

        if os.path.exists(self.save_path):
            pass

    def train(self):
        faceDataset = FaceDataset(self.dataset_path)
        dataloader = DataLoader(faceDataset, batch_size=512, shuffle=True, num_workers=4)

        while True:
            for i, (_img_data, _category, _offset) in enumerate(dataloader):
                pass

                # img_data_ = Variable(_img_data)
                # category_ = Variable(_category)
                # offset_ = Variable(_offset)
                #
                # _output_category, _output_offset = self.net(img_data_)
                # output_category = _output_category.view(-1, 1)
                # output_offset = _output_offset.view(-1, 4)
                #
                # # 计算分类的损失
                # category_mask = torch.lt(category_, 2)  # part样本不参与分类损失计算
                # category = torch.masked_select(category_, category_mask)
                # output_category = torch.masked_select(output_category, category_mask)
                # cls_loss = self.cls_loss_fn(output_category, category)
                #
                # # 计算bound的损失
                # offset_mask = torch.gt(category_, 0)  # 负样本不参与计算
                # offset_index = torch.nonzero(offset_mask)[:, 0]  # 选出非负样本的索引
                # offset = offset_[offset_index]
                # output_offset = output_offset[offset_index]
                # offset_loss = self.offset_loss_fn(output_offset, offset)  # 损失
                # # print("debug:\n", offset_mask, offset_index, offset, offset_)
                #
                # loss = cls_loss + offset_loss
                #
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()

            print("save success")


if __name__ == '__main__':
    net = ONet()

    x = np.random.random((2, 48, 48, 3))
    print(x.shape)
    cls, off = net.forward(x)
    print(cls, cls.shape)
    print(off, off.shape)
