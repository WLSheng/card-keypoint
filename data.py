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
import matplotlib.pyplot as plt


def cv_imread(file_path):
    # 读取中文路径下的图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def show_six_feature(feature, figure_num=2, title='target', img=None):      # feature.shape = (1, 6, 64, 64), numpy
    plt.figure(figure_num)
    for i in range(6):
        plt.subplot(2, 4, i+1)
        plt.axis('off')
        plt.title("{}_{}".format(title, str(i)))
        # plt.imshow(feature[0][i])
        plt.imshow(feature[0][i])

    if img is not None:
        plt.subplot(2, 4, 7)
        plt.axis('off')
        plt.title("img")
        plt.imshow(img[0][0].cpu().numpy())

    plt.subplot(2, 4, 8)
    plt.axis('off')
    plt.title("all")
    plt.imshow(np.max(feature[0], axis=0))


def make_target(content_list):    # 第0个是图片路径，后面才是坐标和label
    # 采用分割的思想
    gaussian_heatmap = np.zeros((6, config.feature_size, config.feature_size))  # 输出特征图是64
    # gaussian_heatmap = []
    for i in range(1, len(content_list), 1):
        label_list = content_list[i].split(",")
        label = int(label_list[0])
        x = int(int(label_list[1]))   # 因为resnet原本的步幅是32，512进来16出，被我调成步幅为8，512进来64出
        y = int(int(label_list[2]))
        one_gaussian_heatmap = CenterLabelHeatMap(config.feature_size, config.feature_size, x, y, 7, stride=8)  # 13
        gaussian_heatmap[label] = one_gaussian_heatmap

        # gt_mask[label][y][x] = 1.0
        # print("write: ", label, x, y)
        # print(coor, type(coor))
    # for l in range(6):
    #     coor = np.where(gt_mask[0][l] == np.max(gt_mask[0][l]))
        # print(gt_mask[0][l][ll])
        # print("max index", l, coor[0][0], coor[1][0])
    # exit(0)

    # 不区分类，只检测关键点
    # result_heatmap = np.zeros((1, config.feature_size, config.feature_size))
    # for i in range(6):
    #     result_heatmap = np.maximum(result_heatmap, gaussian_heatmap[i])
    return torch.from_numpy(np.cast['float32'](gaussian_heatmap))
    # return torch.from_numpy(np.cast['float32'](result_heatmap))


class cardDataset(Dataset):
    def __init__(self,transform):
        self.train_content = open(config.train_txt, 'r').readlines()
        self.transform = transform

    def __getitem__(self, index):
        one_content = self.train_content[index]
        content_list = one_content.split(" ")
        img = cv_imread(content_list[0])
        # img = cv2.resize(img, (320, 320))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.subplot(2, 1, 1)
        # plt.imshow(img)
        img = self.transform(img)
        target = make_target(content_list)

        # debug 显示target是否正确，不能单单验证坐标点对了就结束了，每一步都要验证
        # one_t = target.numpy()
        # one_t = np.max(one_t, axis=0)
        # plt.figure(1)
        # plt.subplot(2, 1, 2)
        # plt.imshow(one_t)
        # show_six_feature([target])
        # plt.show()
        return img, target

    def __len__(self):
        return len(self.train_content)


def collate_fn(batch):
    imgs = []
    labels = []

    for sample in batch:
        imgs.append(sample[0])
        labels.append(sample[1])

    return torch.stack(imgs, 0), torch.stack(labels, 0)


class Normalize(object):
    def __init__(self, mean, std):
        """
        :param mean: RGB order
        :param std:  RGB order
        """
        self.mean = np.array(mean)#.reshape(3, 1, 1)
        self.std = np.array(std)#.reshape(3, 1, 1)

    def __call__(self, image):
        """
        :param image:  (H,W,3)  RGB
        :return:
        """
        # plt.figure(1)
        # plt.imshow(image)
        # plt.show()
        image = (image / 255. - self.mean) / self.std
        image = np.cast['float32'](image)
        return image


def CenterLabelHeatMap(fea_width, fea_height, c_x, c_y, sigma, stride=32):
    Y1 = np.linspace(0, fea_height-1, fea_height)
    X1 = np.linspace(0, fea_width-1, fea_width)
    [X,Y] = np.meshgrid(X1, Y1)
    X = X * stride + stride / 2
    Y = Y * stride + stride / 2
    X = X - c_x
    Y = Y - c_y
    D2 = X*X + Y*Y
    E2 = 2. * sigma * sigma
    Exponet = D2/E2
    heatmap = np.exp(-Exponet)
    return heatmap


if __name__ == '__main__':
    # make gaussian
    # gaussian_heatmap = CenterLabelHeatMap(64, 64, center_x, center_y, 7, stride=8)
    pass