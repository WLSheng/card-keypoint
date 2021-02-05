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
from data import cv_imread
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable


def card_infer():
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ])

    model = models.resnet50(pretrained=False)
    model.fc = models.conv1x1(512 * 4, 6)
    model.load_state_dict(torch.load(r'./save_model/epoch_08.pth'))
    model.cuda()
    model.train(False)

    train_content = open(config.train_txt, 'r').readlines()
    for i, one_content in enumerate(train_content):
        content_list = one_content.split(" ")
        img = cv_imread(content_list[0])
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = model(img)
        out = torch.sigmoid(out)
        out = out.cpu()
        out = out.detach().numpy()
        # print(out, out.shape)
        result = []
        for i in range(out.shape[1]):
            one_layer = out[0][i]
            # one_layer =
            coor = np.where(one_layer == np.max(one_layer))
            print(i, coor[0][0], coor[1][0])

        break


if __name__ == '__main__':
    card_infer()
