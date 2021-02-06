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
import data
from data import cv_imread, make_target, cardDataset, collate_fn, Normalize
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import drn


def train():
    transform = transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), transforms.ToTensor()])

    card_dataset = cardDataset(transform)
    train_dataloader = DataLoader(card_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    # model = models.resnet50(pretrained=True)
    # model = drn.drn_a_50(pretrained=True)
    model = models.resnet50(pretrained=True)
    model.fc = models.conv1x1(512 * 4, 6)
    model.cuda()
    model.train()

    # optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    criterion = CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    # loss_fn = torch.nn.BCELoss(reduce=True, size_average=False)
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.L1Loss(size_average=True)
    # loss_fn = torch.nn.CrossEntropyLoss()
    for e in range(config.epoch):
        for i, (img, target) in enumerate(train_dataloader):

            # one_t = target[0]
            # print("1", one_t.shape)
            # # one_t = torch.Tensor.permute(one_t, (2,1,0))
            # # print("2", one_t.shape)
            # one_t = one_t.numpy()
            # one_t = np.max(one_t, axis=0)
            # plt.imshow(one_t)
            # plt.show()
            # exit(0)

            # img = Variable(img.cuda())
            # gt_mask = Variable(target.cuda()).float()

            img = img.to(device='cuda:0', dtype=torch.float32)
            target = target.to(device='cuda:0', dtype=torch.float32)
            out = model(img)
            # target = target.type(torch.long)
            # out = torch.sigmoid(out).type(torch.float32)
            loss = loss_fn(out, target)
            if i % 20 == 0:
                data.show_six_feature(target.cpu().numpy(), figure_num=1, title='tatget')
                plt_save_feature_name = './save_feature/epoch_{}_iter_{}_target.jpg'.format(e, i)
                plt.savefig(plt_save_feature_name)
                plt.close()
                data.show_six_feature(torch.sigmoid(out).cpu().detach().numpy(), figure_num=2, title='outfeat')
                plt_save_feature_name = './save_feature/epoch_{}_iter_{}_outfeat.jpg'.format(e, i)
                plt.savefig(plt_save_feature_name)
                plt.close()

            # loss = (out - target).abs().sum()
            # loss = loss.sum()
            loss.backward()
            optimizer.step()
            print("======> epoch:{}, iter:{}/{}, loss:{}, lr:{}".format(e, i, card_dataset.__len__()//config.batch_size,
                                                                      round(loss.item(), 7), exp_lr_scheduler.get_last_lr()[0]))
            # plt.show()    # 这句不能与下面的同用
            # plt.savefig("./save_feature/filename.png")
            # plt.ion()       # 打开交互模式
            # plt.pause(3)    # 显示秒数
            # # plt.ioff()      # 显示完后一定要配合使用plt.ioff()关闭交互模式，否则可能出奇怪的问题
            # plt.clf()       # 清空图片
            # plt.close()     # 清空窗口
        save_path = r'./save_model/epoch_{}.pth'.format(str(e).zfill(2))
        print("save model :", save_path)
        torch.save(model.state_dict(), save_path)
        exp_lr_scheduler.step()



if __name__ == '__main__':
    train()
