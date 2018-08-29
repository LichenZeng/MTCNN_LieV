import torch
import tensorflow as tf

a = torch.Tensor([1, 2, 3, 4, 5])
mask = torch.lt(a, 3)
print(a[mask])
print(a.unsqueeze(0))
print(a.unsqueeze(1))

a = tf.constant([1, 2, 3, 4, 5])
mask = tf.where(a < 3)
b = tf.gather(a, mask)

with tf.Session() as sess:
    print(sess.run(mask))
    print(sess.run(b))
