from PIL import Image, ImageDraw
import cv2
import os
import sys
import time
from matplotlib import pyplot as plt

img_path = "../img_celeba_4dbg"
label_path = "../img_celeba_4dbg/list_bbox_celeba.txt"
SIZE = [12, 24, 48]

# print(os.listdir(img_path))

# plt.ion()
for i, line in enumerate(open(label_path)):
    if i < 2:
        continue
    try:
        print(line.split())
        line = line.split()
        img_name = line[0]
        x0 = int(line[1])
        y0 = int(line[2])
        width = int(line[3])
        height = int(line[4])
        x1 = x0 + width
        y1 = y0 + height
        print(x0, y0, width, height, x1, y1)
        img = cv2.imread(os.path.join(img_path, img_name))
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(1000)

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
