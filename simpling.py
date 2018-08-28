from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class FaceDataset(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.dataset = []
        self.transform = transform
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split()
        img_path = os.path.join(self.path, strs[0])
        cond = torch.Tensor([int(strs[1])])
        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
        if self.transform is not None:
            img_data = self.transform(np.array(Image.open(img_path)))
        else:
            img_data = torch.Tensor(np.array(Image.open(img_path)) / 255. - 0.5)

        return img_data, cond, offset

    def __len__(self):
        return len(self.dataset)


tf = transforms.Compose([transforms.ToTensor()])

if __name__ == '__main__':
    dataset = FaceDataset(r"../img_celeba_4dbg/12", transform=tf)
    print(dataset[0])
    print(dataset[0][0].shape)
