import numpy as np


# [x1,y1,x2,y2,c]
def iou(box, boxes, mode="UNION"):
    top_x = np.maximum(box[0], boxes[:, 0])
    top_y = np.maximum(box[1], boxes[:, 1])
    bottom_x = np.minimum(box[2], boxes[:, 2])
    bottom_y = np.minimum(box[3], boxes[:, 3])
    # w = 0 if (bottom_x - top_x) <= 0 else (bottom_x - top_x)
    # h = 0 if (bottom_y - top_y) <= 0 else (bottom_y - top_y)
    w = np.maximum(0, (bottom_x - top_x))
    h = np.maximum(0, (bottom_y - top_y))

    j_area = w * h
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    if mode == "UNION":
        fm = box_area + boxes_area - j_area
    else:
        fm = np.minimum(box_area, boxes_area[:])
    return j_area / fm


# [x1,y1,x2,y2,c]
def rect2squar(boxes):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    side_len = np.maximum(w, h)
    cx = boxes[:, 0] + w / 2
    cy = boxes[:, 1] + h / 2

    x1 = cx - side_len / 2
    y1 = cy - side_len / 2
    x2 = cx + side_len / 2
    y2 = cy + side_len / 2

    return np.stack([x1, y1, x2, y2, boxes[:, 4]], axis=1)
