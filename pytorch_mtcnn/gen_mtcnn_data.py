from PIL import Image, ImageDraw
import cv2
import os
import sys
import time
from matplotlib import pyplot as plt
import random
import numpy as np
from mtcnn_utils import iou, nms, dbg

img_path = "../img_celeba_4dbg"
label_path = "../img_celeba_4dbg/list_bbox_celeba.txt"
save_path = "../img_celeba_4dbg_simple"
positive = "positive"
part = "part"
negative = "negative"

COUNT = 50
SCALE = 0.4
SIZE = [48]

# print(os.listdir(img_path))

for size in SIZE:
    dbg("sample size:", size, lv=1)
    size_save_path = os.path.join(save_path, str(size))

    pos_dir = os.path.join(size_save_path, positive)
    part_dir = os.path.join(size_save_path, part)
    neg_dir = os.path.join(size_save_path, negative)
    pos_file = pos_dir + ".txt"
    part_file = part_dir + ".txt"
    neg_file = neg_dir + ".txt"

    for dir in [pos_dir, part_dir, neg_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    dbg("dir", pos_dir, part_dir, neg_dir, lv=0)
    dbg("file:", pos_file, part_file, neg_file, lv=0)

    for i, line in enumerate(open(label_path)):
        if i < 2:
            continue
        try:
            dbg(line.split(), lv=0)
            line = line.split()
            img_name = line[0]
            x0 = int(line[1])
            y0 = int(line[2])
            width = int(line[3])
            height = int(line[4])
            x1 = x0 + width
            y1 = y0 + height
            box = [x0, y0, x1, y1]
            dbg(x0, y0, width, height, x1, y1, lv=0)

            image = cv2.imread(os.path.join(img_path, img_name))
            (h, w, c) = image.shape  # (h, w, c)
            dbg(h, w, c, lv=0)
            img = image

            count_pos = 0
            count_part = 0
            count_neg = 0

            while True:
                # img = image.copy()
                # Negative
                neg_side = round(random.randint(size, width))
                neg_x0 = round(random.randint(0, w - neg_side))
                neg_y0 = round(random.randint(0, h - neg_side))
                neg_x1 = neg_x0 + neg_side
                neg_y1 = neg_y0 + neg_side

                boxes = [neg_x0, neg_y0, neg_x1, neg_y1]
                area = iou(boxes, box, False)
                area_min = iou(boxes, box, True)
                dbg(area, lv=1)
                if area > 0.65 and count_pos < COUNT:
                    count_pos += 1
                    dbg("positive", lv=1)
                    crop_img = img[neg_y0:neg_y1, neg_x0:neg_x1]
                    save_name = "{}/{}_{}.jpg".format(pos_dir, i, count_pos)
                    cv2.imwrite(save_name, crop_img)

                elif area > 0.4 and count_part < COUNT:
                    count_part += 1
                    dbg("part", lv=1)
                    crop_img = img[neg_y0:neg_y1, neg_x0:neg_x1]
                    save_name = "{}/{}_{}.jpg".format(part_dir, i, count_part)
                    cv2.imwrite(save_name, crop_img)

                elif area < 0.3 and area_min < 0.3 and count_neg < COUNT:
                    count_neg += 1
                    print("negative")
                    crop_img = img[neg_y0:neg_y1, neg_x0:neg_x1]
                    save_name = "{}/{}_{}.jpg".format(neg_dir, i, count_neg)
                    cv2.imwrite(save_name, crop_img)

                # Positive / Part
                off_w = round(random.uniform(- SCALE, SCALE) * width)
                off_h = round(random.uniform(- SCALE, SCALE) * height)
                dbg(off_w, off_h, lv=0)

                off_x0 = 0 if x0 + off_w <= 0 else x0 + off_w
                off_x1 = w if x1 + off_w >= w else x1 + off_w
                off_y0 = 0 if y0 + off_h <= 0 else y0 + off_h
                off_y1 = h if y1 + off_h >= h else y1 + off_h

                boxes = [off_x0, off_y0, off_x1, off_y1]
                area = iou(boxes, box, False)
                if area > 0.65 and count_pos < COUNT:
                    count_pos += 1
                    dbg("positive", lv=1)
                    crop_img = img[off_y0:off_y1, off_x0:off_x1]
                    save_name = "{}/{}_{}.jpg".format(pos_dir, i, count_pos)
                    cv2.imwrite(save_name, crop_img)

                elif area > 0.4 and count_part < COUNT:
                    count_part += 1
                    dbg("part", lv=1)
                    crop_img = img[off_y0:off_y1, off_x0:off_x1]
                    save_name = "{}/{}_{}.jpg".format(part_dir, i, count_part)
                    cv2.imwrite(save_name, crop_img)

                dbg(count_pos, count_part, count_neg, lv=2)
                if count_pos >= COUNT and count_part >= COUNT and count_neg >= COUNT:
                    break

                # cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # cv2.rectangle(img, (off_x0, off_y0), (off_x1, off_y1), (255, 0, 0), 2)
                # cv2.rectangle(img, (neg_x0, neg_y0), (neg_x1, neg_y1), (0, 0, 255), 2)
                # cv2.imshow("image", img)
                # cv2.waitKey(200)

        except:
            print("End of file")
            # raise
        finally:
            print("game over")
            cv2.destroyAllWindows()
