from PIL import Image, ImageDraw
import numpy as np

im = Image.open(r"/home/tensorflow01/workspace/MTCNN/img_celeba/000001.jpg")
print(im.size)

imDraw = ImageDraw.Draw(im)
imDraw.rectangle((95, 71, 95 + 226, 71 + 313), outline='red')
im.show()

boxes = np.array([[22, 2, 3, 4], [2, 6, 7, 8]])
box = np.array([5, 6, 7, 8])
print(np.minimum(box[0], boxes[:, 0]))

x1 = np.array([[1, 2], [3, 4]])
y1 = np.array([[6, 7], [8, 9]])
print(np.concatenate([x1, y1], axis=0))
print(np.concatenate([x1, y1], axis=1))

a1 = np.array([1, 2, 3, 4])
b1 = np.array([5, 6, 7, 8])
print(np.stack([a1, b1], axis=0))
print(np.stack([a1, b1], axis=1))
