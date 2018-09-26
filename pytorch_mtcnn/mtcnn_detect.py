import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from mtcnn_net import pnet, rnet, onet
from mtcnn_utils import dbg, nms, iou
from mtcnn_datasets import transform

img_path = "../img_celeba_4dbg"
label_path = "../img_celeba_4dbg/list_bbox_celeba.txt"
save_path = "../../img_celeba_4dbg_simple"
dlv = 1


def pnet_detect(net, img):
    conf, off = net(img)  # type: pnet
    conf = conf.squeeze()
    # print(torch.lt(conf, 0.5))
    coor = []
    box = []
    height, width = conf.shape
    for h in range(height):
        for w in range(width):
            if conf[h, w] > 0.5:
                coor.append([h, w])
                # print(conf[h, w].item())
    print(coor)
    # exit()
    dbg("pd", conf.shape, off.shape, lv=dlv)
    for c in coor:
        # print(c[0], c[1])
        box.append([2 * c[1], 2 * c[0], 2 * c[1] + 12, 2 * c[0] + 12, conf[c[0], c[1]].item()])
    # print(box)
    return box


def rnet_detect():
    pass


def onet_detect():
    pass


# , rnet, rnet_model, onet, onet_model,
def detect(pnet, pnet_model, img):
    pnet.load_state_dict(torch.load(pnet_model))
    img = np.array(img)
    img = transform(img)
    img = img.unsqueeze(0)
    dbg("detect", img.shape, lv=dlv)

    boxes = pnet_detect(pnet, img)
    return boxes


if __name__ == '__main__':
    # img = cv2.imread(os.path.join(img_path, "000001.jpg"))
    # b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # img = np.stack((r, g, b), axis=2)
    # print(type(img), img.shape)
    net = pnet()
    img1 = img = Image.open(os.path.join(img_path, "000007.jpg"))
    img = img.resize((int(img.size[0] * 0.04), int(img.size[1] * 0.04)))
    print(img.size)
    # img1.show()
    box = detect(net, "./save_model/pnet.pkl", img)
    print(box)
    box = nms(box, 0.5)

    color = 0
    draw = ImageDraw.Draw(img1)
    for b in box:
        x1, y1, x2, y2 = b[0] * 25, b[1] * 25, b[2] * 25, b[3] * 25
        color += 40
        draw.rectangle((x1, y1, x2, y2), outline=(0, color, 0))
    img1.show()

    # img.show()
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    pass
