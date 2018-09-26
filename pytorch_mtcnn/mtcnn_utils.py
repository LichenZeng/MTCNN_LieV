import numpy as np
import random


def dbg(*args, lv=0):
    if lv == 1:
        print("OVO:", *args)
    elif lv == 2:
        print("#" * 16 + "\n", *args)
    elif lv == 3:
        print("#" * 8 + __file__ + "#" * 8 + "\n", *args)


def iou(boxes, box, ismin=False):
    # print("iou")
    maxx0 = np.maximum(boxes[0], box[0])
    maxy0 = np.maximum(boxes[1], box[1])
    minx1 = np.minimum(boxes[2], box[2])
    miny1 = np.minimum(boxes[3], box[3])
    # print(maxx0, maxy0, minx1, miny1)
    width = 0 if minx1 - maxx0 <= 0 else minx1 - maxx0
    height = 0 if miny1 - maxy0 <= 0 else miny1 - maxy0
    area_com = width * height
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])

    # print("area com: {}, boxes: {}, box: {}".format(area_com, area_boxes, area_box))

    if ismin:
        return area_com / np.minimum(area_boxes, area_box)
    else:
        return area_com / (area_boxes + area_box - area_com)


def nms(box, area, ismin=False):
    print("nms")
    dbg(box, lv=2)

    boxx = sorted(box, key=lambda b: b[4])
    print(boxx)
    # box = np.array(box)
    # boxn = box[np.argsort(box[:, 4])]
    # print(boxn)
    box22 = []
    while len(boxx) > 0:
        bb = boxx[-1]
        box22.append(bb)
        boxx.remove(bb)
        print(boxx)
        dbg("zz", bb, len(boxx), lv=1)

        for i in range(len(boxx)):
            bx = boxx[-1]
            ar = iou(bb, bx)
            print("IOU is \n", ar)
            if ar > area:
                boxx.remove(bx)
        dbg(boxx, lv=2)

    print(box)
    return box


if __name__ == '__main__':
    dbg(1, 3, 20, 40, lv=3)
    exit()
    width, height = 30, 35
    box1 = [0, 0, 10, 10]
    box2 = [0, 5, 10, 15]
    box1 = np.array(box1)
    box2 = np.array(box2)
    area1 = iou(box2, box1, True)
    area2 = iou(box2, box1)
    print(area1, area2)
    off_w = round(random.uniform(-1, 1) * (box1[2] - box1[0]))
    off_h = round(random.uniform(-1, 1) * (box1[3] - box1[1]))
    print(off_w, off_h)
    off_x0 = 0 if box1[0] + off_w <= 0 else box1[0] + off_w
    off_x1 = width if box1[2] + off_w >= width else box1[2] + off_w

    off_y0 = 0 if box1[1] + off_h <= 0 else box1[1] + off_h
    off_y1 = height if box1[3] + off_w >= height else box1[3] + off_h

    print(off_x0, off_y0, off_x1, off_y1)
