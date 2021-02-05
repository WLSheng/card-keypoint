import shutil
import cv2
import torch
import random
import json
import numpy as np
import os
import models
import config
import time
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from data import cv_imread, make_mask, cardDataset, collate_fn
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


def train():
    transform = transforms.Compose([transforms.ToTensor()])  # range [0, 255] -> [0.0,1.0]

    card_dataset = cardDataset(transform)
    train_dataloader = DataLoader(card_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    model = models.resnet50(pretrained=True)
    model.fc = models.conv1x1(512 * 4, 6)
    model.cuda()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    criterion = CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.70)
    # loss_fn = torch.nn.BCELoss(reduce=True, size_average=False)
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.L1Loss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    for e in range(config.epoch):
        for i, (img, gt_mask) in enumerate(train_dataloader):
            img = Variable(img.cuda())
            gt_mask = Variable(gt_mask.cuda()).float()

            out = model(img)
            loss = loss_fn(torch.sigmoid(out), gt_mask)
            loss = loss.sum()
            loss.backward()
            optimizer.step()
            print("=====>epoch:{}, iter:{}/{}, loss:{}, lr:{}".format(e, i, card_dataset.__len__()//config.batch_size,
                                                                      round(loss.item(), 7), exp_lr_scheduler.get_lr()[0]))
        save_path = r'./save_model/epoch_{}.pth'.format(str(e).zfill(2))
        print("save model :", save_path)
        torch.save(model.state_dict(), save_path)
        exp_lr_scheduler.step()

        # input = Variable(torch.randn(3, 4))
        # target = Variable(torch.FloatTensor(3, 4).random_(2))


if __name__ == '__main__':
    train()
