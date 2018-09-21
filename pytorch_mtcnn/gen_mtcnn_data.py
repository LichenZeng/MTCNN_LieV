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
save_path = "../../img_celeba_4dbg_simple"
positive = "positive"
part = "part"
negative = "negative"

COUNT = 50
SCALE = 0.8
SIZE = [48, 24, 12]

# print(os.listdir(img_path))

for size in SIZE:
    dbg("sample size:", size, lv=1)
    size_save_path = os.path.join(save_path, str(size))

    pos_dir = os.path.join(size_save_path, positive)
    part_dir = os.path.join(size_save_path, part)
    neg_dir = os.path.join(size_save_path, negative)
    pos_name = pos_dir + ".txt"
    part_name = part_dir + ".txt"
    neg_name = neg_dir + ".txt"

    for dir in [pos_dir, part_dir, neg_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    dbg("dir", pos_dir, part_dir, neg_dir, lv=0)
    dbg("name:", pos_name, part_name, neg_name, lv=0)

    try:
        pos_file = open(pos_name, "w")
        part_file = open(part_name, "w")
        neg_file = open(neg_name, "w")

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

                image = cv2.imread(os.path.join(img_path, img_name))
                (h, w, c) = image.shape  # (h, w, c)
                dbg(h, w, c, lv=0)
                if h < 40 or w < 40:
                    dbg(img_name, lv=2)
                    continue

                cx = x0 + width // 2
                cy = y0 + height // 2
                box = [x0, y0, x1, y1]
                dbg(x0, y0, width, height, x1, y1, lv=0)

                count_pos = 0
                count_part = 0
                count_neg = 0
                img = image
                while True:
                    # img = image.copy()
                    # Negative
                    neg_side = random.randint(size, min(h, w))  # Todo
                    neg_x0 = round(random.randint(0, w - neg_side))
                    neg_y0 = round(random.randint(0, h - neg_side))
                    neg_x1 = neg_x0 + neg_side
                    neg_y1 = neg_y0 + neg_side

                    boxes = [neg_x0, neg_y0, neg_x1, neg_y1]
                    area = iou(boxes, box, False)
                    area_min = iou(boxes, box, True)
                    dbg(area, area_min, lv=0)

                    off_x0 = float((x0 - neg_x0)) / neg_side
                    off_y0 = float((y0 - neg_y0)) / neg_side
                    off_x1 = float((x1 - neg_x1)) / neg_side
                    off_y1 = float((y1 - neg_y1)) / neg_side

                    if area > 0.65 and count_pos < COUNT:
                        count_pos += 1
                        dbg("positive", lv=0)
                        crop_img = img[neg_y0:neg_y1, neg_x0:neg_x1]
                        crop_img = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_CUBIC)
                        save_name = "{}/{}_{}.jpg".format(pos_dir, i, count_pos)
                        cv2.imwrite(save_name, crop_img)
                        record = "{} 1 {} {} {} {}\n".format(save_name, off_x0, off_y0, off_x1, off_y1)
                        pos_file.write(record)
                        pos_file.flush()

                    elif area > 0.4 and count_part < COUNT:
                        count_part += 1
                        dbg("part", lv=0)
                        crop_img = img[neg_y0:neg_y1, neg_x0:neg_x1]
                        crop_img = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_CUBIC)
                        save_name = "{}/{}_{}.jpg".format(part_dir, i, count_part)
                        cv2.imwrite(save_name, crop_img)
                        record = "{} 2 {} {} {} {}\n".format(save_name, off_x0, off_y0, off_x1, off_y1)
                        part_file.write(record)
                        part_file.flush()

                    elif area < 0.3 and area_min < 0.3 and count_neg < COUNT:
                        count_neg += 1
                        dbg("negative", lv=0)
                        crop_img = img[neg_y0:neg_y1, neg_x0:neg_x1]
                        crop_img = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_CUBIC)
                        save_name = "{}/{}_{}.jpg".format(neg_dir, i, count_neg)
                        cv2.imwrite(save_name, crop_img)
                        record = "{} 0 0 0 0 0\n".format(save_name)
                        neg_file.write(record)
                        neg_file.flush()

                    # Positive / Part
                    p_side = min(height, width)
                    pp_cx = cx + round(random.uniform(- SCALE, SCALE) * p_side // 2)  # Todo
                    pp_cy = cy + round(random.uniform(- SCALE, SCALE) * p_side // 2)  # Todo
                    pp_side = random.randint(size, min(pp_cx, pp_cy, w - pp_cx, h - pp_cy))  # Todo
                    pp_x0 = pp_cx - pp_side
                    pp_y0 = pp_cy - pp_side
                    pp_x1 = pp_x0 + pp_side * 2
                    pp_y1 = pp_y0 + pp_side * 2

                    boxes = [pp_x0, pp_y0, pp_x1, pp_y1]
                    area = iou(boxes, box, False)
                    area_min = iou(boxes, box, True)
                    dbg(area, area_min, lv=0)

                    off_x0 = float((x0 - pp_x0)) / pp_side
                    off_y0 = float((y0 - pp_y0)) / pp_side
                    off_x1 = float((x1 - pp_x1)) / pp_side
                    off_y1 = float((y1 - pp_y1)) / pp_side

                    if area > 0.65 and count_pos < COUNT:
                        count_pos += 1
                        dbg("positive", lv=0)
                        crop_img = img[pp_y0:pp_y1, pp_x0:pp_x1]
                        crop_img = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_CUBIC)
                        save_name = "{}/{}_{}.jpg".format(pos_dir, i, count_pos)
                        cv2.imwrite(save_name, crop_img)
                        record = "{} 1 {} {} {} {}\n".format(save_name, off_x0, off_y0, off_x1, off_y1)
                        pos_file.write(record)
                        pos_file.flush()

                    elif area > 0.4 and count_part < COUNT:
                        count_part += 1
                        dbg("part", lv=0)
                        crop_img = img[pp_y0:pp_y1, pp_x0:pp_x1]
                        crop_img = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_CUBIC)
                        save_name = "{}/{}_{}.jpg".format(part_dir, i, count_part)
                        cv2.imwrite(save_name, crop_img)
                        record = "{} 2 {} {} {} {}\n".format(save_name, off_x0, off_y0, off_x1, off_y1)
                        part_file.write(record)
                        part_file.flush()

                    elif area < 0.3 and area_min < 0.5 and count_neg < COUNT:
                        count_neg += 1
                        dbg("negative", lv=0)
                        crop_img = img[pp_y0:pp_y1, pp_x0:pp_x1]
                        crop_img = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_CUBIC)
                        save_name = "{}/{}_{}.jpg".format(neg_dir, i, count_neg)
                        cv2.imwrite(save_name, crop_img)
                        record = "{} 0 0 0 0 0\n".format(save_name)
                        neg_file.write(record)
                        neg_file.flush()

                    dbg(img_name, count_pos, count_part, count_neg, lv=0)
                    if count_pos >= COUNT and count_part >= COUNT and count_neg >= COUNT:
                        break

                    # cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    # cv2.rectangle(img, (pp_x0, pp_y0), (pp_x1, pp_y1), (255, 0, 0), 2)
                    # cv2.rectangle(img, (neg_x0, neg_y0), (neg_x1, neg_y1), (0, 0, 255), 2)
                    # cv2.imshow("image", img)
                    # cv2.waitKey(0)

            except:
                print("End of file")
                # raise
    finally:
        print("sample over")
        cv2.destroyAllWindows()
