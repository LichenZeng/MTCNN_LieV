import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from mtcnn_utils import dbg

dlv = -1
img_path = "../img_celeba_4dbg"
label_path = "../img_celeba_4dbg/list_bbox_celeba.txt"
save_path = "../../img_celeba_4dbg_simple"
positive = "positive"
part = "part"
negative = "negative"

COUNT = 50
SCALE = 0.8
SIZE = [48, 24, 12]

transform = transforms.Compose([
    transforms.ToTensor(),
])


class mtcnn_dataset(Dataset):
    def __init__(self, root, size, transform=None):
        super(mtcnn_dataset, self).__init__()
        dbg("init", lv=dlv)
        self.transform = transform
        self.labels = []
        self.labels.extend(list(open(os.path.join(root, str(size), "positive.txt"))))
        self.labels.extend(list(open(os.path.join(root, str(size), "part.txt"))))
        self.labels.extend(list(open(os.path.join(root, str(size), "negative.txt"))))
        dbg(self.labels, len(self.labels), lv=dlv)

    def __getitem__(self, index):
        dbg("getitem", lv=dlv)
        item = self.labels[index].split()
        dbg(item, lv=dlv)
        self.img_path = item[0]
        self.cls = float(item[1])
        self.off_x0 = float(item[2])
        self.off_y0 = float(item[3])
        self.off_x1 = float(item[4])
        self.off_y1 = float(item[5])
        self.cls = torch.Tensor([self.cls])
        self.off = torch.Tensor([self.off_x0, self.off_y0, self.off_x1, self.off_y1])
        dbg(self.img_path, self.cls, self.off_x0, self.off_y0, self.off_x1, self.off_y1, lv=dlv + 1)
        img = Image.open(self.img_path)
        if self.transform is not None:
            dbg("Transform is not None", lv=dlv)
            img = self.transform(img)

        return img, self.cls, self.off

    def __len__(self):
        dbg("len", lv=dlv)
        return len(self.labels)


if __name__ == '__main__':
    dataset = mtcnn_dataset(save_path, 48, transform)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    plt.ion()
    for step, (img, cls, off) in enumerate(data_loader):
        print(img.shape)
        grid = utils.make_grid(img)
        plt.clf()
        plt.imshow(grid.data.numpy().transpose((1, 2, 0)))
        plt.pause(1)
    plt.ioff()

    # print(len(dataset))
    # print(dataset[0][0])
    # dbg(list(open(os.path.join(save_path, str(12), "positive.txt"))), lv=dlv)
    # torchvision.datasets.ImageFolder()
    # label = [
    #     '/positive/2_1.jpg 1 0.2916666666666667 -0.24305555555555555 -0.1388888888888889 -0.06944444444444445\n',
    #     '/positive/2_1.jpg 1 0.2916666666666667 -0.24305555555555555 -0.1388888888888889 -0.06944444444444445\n']
    # print("zeng>>", label[0])
    # path = label[0].split()
    # for i in path:
    #     print(i)
