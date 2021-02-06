import shutil
import cv2
import torch
import random
import json
import numpy as np
import os
import glob
import codecs
import models
import config
import time
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from data import cv_imread, make_target, cardDataset, collate_fn, Normalize, show_six_feature
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


def card_infer():
    transform = transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), transforms.ToTensor()])

    model = models.resnet50(pretrained=False)
    model.fc = models.conv1x1(512 * 4, 6)
    model.load_state_dict(torch.load(r'./save_model/epoch_15.pth'))
    model.cuda()
    model.train(False)

    train_content = open(config.train_txt, 'r').readlines()
    for i, one_content in enumerate(train_content):
        content_list = one_content.split(" ")
        img = cv_imread(content_list[0])
        draw_img = img.copy()
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = model(img)
        # out = torch.sigmoid(out)
        out = out.cpu()
        out = out.detach().numpy()
        show_six_feature(out)
        plt.figure(2)
        plt.subplot(2, 1, 1)
        show_out = np.max(out[0], axis=0)
        plt.imshow(show_out)

        # print(out, out.shape)
        result = []
        for i in range(out.shape[1]):
            one_layer = out[0][i]
            # one_layer =
            coor = np.where(one_layer == np.max(one_layer))
            print(i, coor[0][0], coor[1][0], "max: ", np.max(one_layer))
            x = coor[1][0] * 8
            y = coor[0][0] * 8
            draw_img = cv2.putText(draw_img, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255], 2)
            draw_img = cv2.circle(draw_img, (x, y), 3, [0, 255, 0], 2)
        plt.subplot(2, 1, 2)
        plt.imshow(draw_img)
        plt.show()
        # cv2.imshow("test", draw_img)
        # cv2.waitKey(0)
        # break


if __name__ == '__main__':
    card_infer()
