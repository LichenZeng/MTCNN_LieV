import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from mtcnn_datasets import DataLoader, mtcnn_dataset, transform, save_path
from mtcnn_utils import dbg
from mtcnn_net import pnet, rnet, onet

dlv = 0


class train_net:

    def __init__(self, net, size, model):
        self.net = net.cuda()  # type: nn.Module
        self.model = model
        self.dataset = mtcnn_dataset(save_path, size, transform)
        self.data_loader = DataLoader(self.dataset, batch_size=128, shuffle=True, drop_last=True)
        self.cls_loss_fn = nn.MSELoss()
        self.opt = optim.Adam(self.net.parameters(), lr=0.0001)
        self.off_loss_fn = nn.MSELoss()

        if os.path.exists(self.model):
            dbg(self.model + " is exist", lv=dlv)
            self.net.load_state_dict(torch.load(self.model))

    def forward(self):
        self.net.train()
        for step, (img, cls, off) in enumerate(self.data_loader):
            img = img.cuda()
            cls = cls.cuda()
            off = off.cuda()
            dbg(img.shape, cls.shape, off.shape, lv=dlv)
            _cls, _off = self.net(img)
            _cls = _cls.view(-1, 1)
            _off = _off.view(-1, 4)

            # Train cls
            index = torch.lt(cls, 2)
            # dbg(index, lv=dlv)
            cls_s = cls[index]
            _cls_s = _cls[index]
            # dbg(cls_s, _cls_s, lv=dlv + 1)
            cls_loss = self.cls_loss_fn(_cls_s, cls_s)

            # Train off
            index = torch.gt(cls, 0)
            # dbg(index[:, 0], lv=dlv)
            off_s = off[index[:, 0]]
            _off_s = _off[index[:, 0]]
            # dbg(off_s.shape, _off_s.shape, lv=dlv)
            off_loss = self.off_loss_fn(_off_s, off_s)

            loss = cls_loss + off_loss
            if step % 100 == 0:
                dbg("loss: {}, cls: {}, off: {}".format(loss, cls_loss, off_loss), lv=dlv + 1)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def save_model(self):
        torch.save(self.net.state_dict(), self.model)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', type=str,
                        help='the net model will be trained',
                        default='pnet')
    parser.add_argument('--model_dir', type=str,
                        help='The directory of trained model',
                        default='./save_model/pnet.pkl')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = sys.argv
    args = parse_arguments(args[1:])
    if args.net == "pnet":
        net = pnet()
        size = 12
    elif args.net == "rnet":
        net = rnet()
        size = 24
    elif args.net == "onet":
        net = onet()
        size = 48

    train = train_net(net, size, args.model_dir)
    for i in range(1000):
        train.forward()
        train.save_model()

    # x = torch.Tensor([1, 0, 2])
    # off = torch.Tensor([[0.2, -0.3, -0.1, 0.4], [0.3, -0.4, -0.7, 0.4], [0.5, -0.3, -0.1, 0.6]])
    # index = torch.lt(x, 2)  # x.gt(0)
    # print(index)
    # r_x = x[index]
    # r_off = off[index]
    # print(r_x)
    # print(r_off)
