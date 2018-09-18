from PIL import Image, ImageDraw
import cv2
import os
import sys
import time
from matplotlib import pyplot as plt
import random
from mtcnn_utils import iou, nms

img_path = "../img_celeba_4dbg"
label_path = "../img_celeba_4dbg/list_bbox_celeba.txt"
SIZE = [12, 24, 48]

# print(os.listdir(img_path))

# plt.ion()
for i, line in enumerate(open(label_path)):
    if i < 2:
        continue
    try:
        # print(line.split())
        line = line.split()
        img_name = line[0]
        x0 = int(line[1])
        y0 = int(line[2])
        width = int(line[3])
        height = int(line[4])
        x1 = x0 + width
        y1 = y0 + height
        # print(x0, y0, width, height, x1, y1)
        img = cv2.imread(os.path.join(img_path, img_name))
        (h, w, c) = img.shape  # (h, w, c)

        # print(h, w, c)

        off_w = round(random.uniform(-0.8, 0.8) * width)
        off_h = round(random.uniform(-0.8, 0.8) * height)
        # print(off_w, off_h)
        off_x0 = 0 if x0 + off_w <= 0 else x0 + off_w
        off_x1 = w if x1 + off_w >= w else x1 + off_w

        off_y0 = 0 if y0 + off_h <= 0 else y0 + off_h
        off_y1 = h if y1 + off_h >= h else y1 + off_h

        box = [x0, y0, x1, y1]
        boxes = [off_x0, off_y0, off_x1, off_y1]
        print("area: ", iou(boxes, box, True))
        # img = img[y0:y1, x0:x1]
        # save_name = "./crip{}.jpg".format(i)
        # cv2.imwrite(save_name, img)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.rectangle(img, (off_x0, off_y0), (off_x1, off_y1), (255, 0, 0), 2)

        cv2.imshow("image", img)
        cv2.waitKey(5000)

        # plt.clf()
        # img = Image.open(os.path.join(img_path, img_name))  # type: Image.Image
        # draw = ImageDraw.Draw(img)
        # draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0))
        # plt.imshow(img)
        # plt.pause(1)

    except:
        print("End of file")
    finally:
        print("game over")
        cv2.destroyAllWindows()
        # plt.ioff()

# for size in SIZE:
#     print(size)
