# MTCNN_LieV
Implement MTCNN base on Celeba dataset at LieV

Note:
======

Author: Lycan
Date: 20180829
Subject: The note of MTCNN bug debug

1, LongTensor 使用主要事项
start_index = start_index.data.numpy()  # Important: LongTensor to numpy
LongTensor类型的张量 除以 浮点数，会自动去掉 浮点数小数点后面的数。所以需要小心其出现除零错误，如：

LongTensor(3)/1.5 = 3
LongTensor(3)/0.5 会报错“interrupted by signal 8: SIGFPE”


2, 在使用DataLoader时，如果样本集少于batch_size指定的数量，在后面使用时可能会出现 实际测试数据 不足batch_size指定的数量 的错误。
dataloader = DataLoader(faceDataset, batch_size=10, shuffle=True, num_workers=2)
offset_index = torch.nonzero(offset_mask)[:, 0]  # 选出非负样本的索引


3, 调试技巧
可以使用小批量测试集合来测试过拟合，判断代码是否有问题；
可以使用小尺寸图片（如24x24）来调试代码错误，方便问题跟踪；


Author: Lycan
Date: 20180831

1, Tensorflow形状变换问题
# This is a wrong method, because the batch number is unknown
# self.conv_flat = tf.reshape(self.conv3, [self.conv3.get_shape()[0], -1])
在已知后面形状时，可以使用如下方法：
self.conv_flat = tf.reshape(self.conv3, (-1, 2 * 2 * 64))
或者用全卷积


2, 在Tensorflow中，可以运用这种方式来接收任意形状的张量输入
self.input = tf.placeholder(tf.float32, shape=[None, None, None, 3])


3, 在Tensorflow中，可以通过这个方式来加载不同的训练参数模型
self.saver = tf.train.Saver([v for v in tf.global_variables() if v.name[0:4] == self.scope])

with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
    ...
