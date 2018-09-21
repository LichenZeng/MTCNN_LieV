import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from mtcnn_datasets import transform, mtcnn_dataset, DataLoader
from mtcnn_utils import dbg

dlv = 1


class pnet(nn.Module):

    def __init__(self):
        super(pnet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(10, 16, kernel_size=3),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.PReLU(),
        )
        self.conv_1 = nn.Conv2d(32, 1, kernel_size=1)
        self.conv_2 = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        conv = self.conv(x)
        dbg(conv.shape, lv=dlv)
        conv_1 = self.conv_1(conv)
        conv_2 = self.conv_2(conv)
        return conv_1, conv_2


class rnet(nn.Module):

    def __init__(self):
        super(rnet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(28, 48, kernel_size=3),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2),
            nn.PReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.PReLU(),
        )
        self.fc_1 = nn.Linear(128, 1)
        self.fc_2 = nn.Linear(128, 4)

    def forward(self, x):
        conv = self.conv(x)
        dbg(conv.shape, lv=dlv)
        fc = conv.view(-1, 3 * 3 * 64)
        fc = self.fc(fc)
        dbg(fc.shape, lv=dlv)
        fc_1 = self.fc_1(fc)
        fc_2 = self.fc_2(fc)

        return fc_1, fc_2


class onet(nn.Module):

    def __init__(self):
        super(onet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.PReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.PReLU(),
        )
        self.fc_1 = nn.Linear(256, 1)
        self.fc_2 = nn.Linear(256, 4)

    def forward(self, x):
        conv = self.conv(x)
        dbg(conv.shape, lv=dlv)
        fc = conv.view(-1, 3 * 3 * 128)
        fc = self.fc(fc)
        dbg(fc.shape, lv=dlv)
        fc_1 = self.fc_1(fc)
        fc_2 = self.fc_2(fc)

        return fc_1, fc_2


if __name__ == '__main__':
    pnet = pnet()
    rnet = rnet()
    onet = onet()

    test = torch.randn(1, 3, 12, 12)
    t1, t2 = pnet(test)
    dbg(t1.shape, t2.shape, lv=dlv)

    test = torch.randn(1, 3, 24, 24)
    t1, t2 = rnet(test)
    dbg(t1.shape, t2.shape, lv=dlv)

    test = torch.randn(1, 3, 48, 48)
    t1, t2 = onet(test)
    dbg(t1.shape, t2.shape, lv=dlv)
