import os
from PIL import Image
import numpy as np
from tool import utils
import traceback

label_path = r"./img_celeba_4dbg/list_bbox_celeba.txt"
img_path = r"./img_celeba_4dbg/"
save_path = r"../img_celeba_4dbg/"

for face_size in [12, 24, 48]:

    print("gen %i image" % face_size)
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 样本描述存储路径
    positive_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_file = open(positive_filename, "w")
        negative_file = open(negative_filename, "w")
        part_file = open(part_filename, "w")

        for i, line in enumerate(open(label_path)):
            if i < 2:
                continue
            try:
                strs = line.strip().split()
                image_filename = strs[0].strip()
                print(image_filename)
                image_file = os.path.join(img_path, image_filename)

                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    boxes = [[x1, y1, x2, y2]]
                    _boxes = np.array(boxes)

                    # 计算出人脸中心点位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    # 使正样本和部分样本数量翻倍
                    for _ in range(5):
                        # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(-w * 0.2, w * 0.2)
                        h_ = np.random.randint(-h * 0.2, h * 0.2)
                        cx_ = cx + w_
                        cy_ = cy + h_

                        # 让人脸形成正方形，并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                        x1_ = np.maximum(0, cx_ - side_len / 2)
                        y1_ = np.maximum(0, cy_ - side_len / 2)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])

                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        # 剪切下图片，并进行大小缩放
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                        iou = utils.iou(crop_box, _boxes, False)[0]
                        iou_min = utils.iou(crop_box, _boxes, True)[0]

                        if iou > 0.65 and iou_min > 0.65:  # 正样本
                            positive_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(positive_count, 1, offset_x1, offset_y1,
                                                                                offset_x2, offset_y2))
                            positive_file.flush()
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1

                        elif iou > 0.4 and iou_min > 0.4:  # 部分样本
                            part_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1,
                                                                            offset_x2, offset_y2))
                            part_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1

                        elif iou < 0.3 and iou_min < 0.3:
                            negative_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                            negative_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

                    # 生成负样本
                    for _ in range(5):
                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if utils.iou(crop_box, _boxes, False) < 0.3 and utils.iou(crop_box, _boxes, True) < 0.3:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                            negative_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                            negative_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

            except Exception as e:
                print("List empty now!")
                # traceback.print_exc()

    finally:
        positive_file.close()
        negative_file.close()
        part_file.close()
