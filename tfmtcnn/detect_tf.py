import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from tool import utils
import nets_tf


class Detector:
    def __init__(self, pnet_param="./tfparam/12"):

        self.pnet = nets_tf.Net()
        self.pnet.forword()
        self.pnet.backward()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        checkpoint = tf.train.get_checkpoint_state(pnet_param)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def detect(self, image):

        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        return pnet_boxes

    def __pnet_detect(self, image):

        boxes = []
        img = image
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1
        while min_side_len > 12:
            img_data = np.array(img) / 255. - 0.5
            img_data = img_data[np.newaxis, :]
            # print(img_data, img_data.shape)

            _cls, _offest = self.sess.run([self.pnet.cls_pre, self.pnet.off_pre], feed_dict={self.pnet.input: img_data})
            # print(_cls, _cls.shape)
            # print(_offest, _offest.shape)

            cls, offest = _cls[0, :, :, 0], _offest[0]
            # print(cls, cls.shape)
            # print(offest, offest.shape)

            idxs = np.where(cls > 0.999999)
            # print(np.stack(idxs, axis=1))
            # print(list(zip(idxs[0], idxs[1])))

            for idx in list(zip(idxs[0], idxs[1])):
                print("debug", idx[0], idx[1])
                print(cls[idx[0], idx[1]])
                boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)

        return utils.nms(utils.nms(np.array(boxes), 0.5), 0.7, isMin=True)

    # 将回归量还原到原图上去
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):

        _x1 = (start_index[1] * stride) / scale
        _y1 = (start_index[0] * stride) / scale
        _x2 = (start_index[1] * stride + side_len) / scale
        _y2 = (start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[start_index[0], start_index[1], :]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]


# save_path_base = "./tfparam"
# dataset_path_base = "../samples/"
# size = 12
# channel = 3
# save_path = os.path.join(save_path_base, str(size))
# dataset_path = os.path.join(dataset_path_base, str(size))

if __name__ == '__main__':
    # image_file = "../../img_celeba_4dbg/24/positive/0.jpg"
    image_file = "../img_celeba_4dbg/000002.jpg"
    detector = Detector()

    with Image.open(image_file) as im:
        # im.show()
        boxes = detector.detect(im)
        print(im.size)
        imDraw = ImageDraw.Draw(im)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            print(box[4])
            imDraw.rectangle((x1, y1, x2, y2), outline='red')

        im.show()
