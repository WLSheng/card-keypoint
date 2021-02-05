import shutil
import cv2
import torch
import random
import json
import numpy as np
import os
import config
from torch.utils.data import Dataset
from torch.autograd import Variable


def cv_imread(file_path):
    # 读取中文路径下的图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def make_mask(content_list):    # 第0个是图片路径，后面才是坐标和label
    # 采用分割的思想
    gt_mask = np.zeros((1, 6, 64, 64))  # 输出特征图是64
    for i in range(1, len(content_list), 1):
        label_list = content_list[i].split(",")
        label = int(label_list[0])
        x = int(int(label_list[1])/8)   # 因为resnet原本的步幅是32，512进来16出，被我调成步幅为8，512进来64出
        y = int(int(label_list[2])/8)
        gt_mask[0][label][x][y] = 1.0
        # print("write: ", label, x, y)
        # print(coor, type(coor))
    for l in range(6):
        coor = np.where(gt_mask[0][l] == np.max(gt_mask[0][l]))
        # print(gt_mask[0][l][ll])
        # print("max index", l, coor[0][0], coor[1][0])
    # exit(0)
    return torch.from_numpy(np.cast['float32'](gt_mask))


class cardDataset(Dataset):
    def __init__(self,transform):
        self.train_content = open(config.train_txt, 'r').readlines()
        self.transform = transform

    def __getitem__(self, index):
        one_content = self.train_content[index]
        content_list = one_content.split(" ")
        img = cv_imread(content_list[0])
        img = self.transform(img)
        # img = torch.unsqueeze(img, 0)
        # img = Variable(img.cuda())
        gt_mask = make_mask(content_list)
        # gt_mask = Variable(gt_mask.cuda()).float()
        # print("???", type(img))
        return img, gt_mask

    def __len__(self):
        return len(self.train_content)


def collate_fn(batch):
    imgs = []
    labels = []

    for sample in batch:
        imgs.append(sample[0])
        labels.append(sample[1])
    return torch.stack(imgs, 0), torch.stack(labels, 0)


if __name__ == '__main__':
    pass