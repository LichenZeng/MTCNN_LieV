import torch
import tensorflow as tf

# a = torch.Tensor([1, 2, 3, 4, 5])
# mask = torch.lt(a, 3)
# print(a[mask])
# print(a.unsqueeze(0))
# print(a.unsqueeze(1))

# a = tf.constant([1, 2, 3, 4, 5])
# mask = tf.where(a < 3)
# b = tf.gather(a, mask)
#
# with tf.Session() as sess:
#     print(sess.run(mask))
#     print(sess.run(b))


# a = tf.constant([[1, 0, 2, 1]])
# a = tf.transpose(a)
# mask = tf.where(a < 2)
# b = tf.gather(a, mask)[:, 0]
#
# with tf.Session() as sess:
#     print(sess.run(a))
#     print(sess.run(mask))
#     print(sess.run(b))


# [[1.]
#  [0.]
#  [1.]
#  [1.]]
# [[ 0.25714287 -0.08571429  0.         -0.05714286]
#  [ 0.          0.          0.          0.        ]
#  [ 0.15151516  0.03787879 -0.06060606  0.12878788]
#  [-0.03694581 -0.2635468  -0.14039409 -0.02216749]]

cls = tf.constant([[2, 0, 1, 1]])
cls = tf.transpose(cls)
mask = tf.where(cls > 0)
cls_ok = tf.gather(cls, mask)[:, 0]

off = tf.constant([[0.25714287, -0.08571429, 0., -0.05714286],
                   [0., 0., 0., 0.],
                   [0.15151516, 0.03787879, -0.06060606, 0.12878788],
                   [-0.03694581, -0.2635468, -0.14039409, -0.02216749]])

off_ok = tf.gather(off, mask)[:, 0]

with tf.Session() as sess:
    print(sess.run(cls))
    print(sess.run(mask))
    print(sess.run(cls_ok))
    print(sess.run(off_ok))
