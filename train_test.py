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
    transform = transforms.Compose([transforms.ToTensor()])     # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    card_dataset = cardDataset(transform)
    train_dataloader = DataLoader(card_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    model = models.resnet50(pretrained=True)
    # model = drn.drn_d_38(pretrained=True)
    model.fc = None
    # model = models.resnet50(pretrained=True)
    # model.load_state_dict(torch.load(r'./drn_d_38_cityscapes.pth'))
    # model.load_state_dict(torch.load('./new_drn_d_38_seg_pretrain.pth'))
    model.fc = torch.nn.Conv2d(512 * 4, 6, kernel_size=1, stride=1, bias=True)
    # model.fc.bias.fill(-1)
    # model.fc = models.conv1x1(512, 6)

    model.cuda()
    model.train()

    # optimizer = optim.SGD(model.parameters(), lr=4e-5, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    criterion = CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    # loss_fn = torch.nn.BCELoss(reduce=True, size_average=False)
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.L1Loss(reduction='mean')
    # loss_fn = torch.nn.CrossEntropyLoss()
    for e in range(config.epoch):
        for i, (img, target) in enumerate(train_dataloader):

            # one_t = target[0]
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
            # print("1", out.shape, target.shape)
            # target = target.type(torch.long)
            # out = torch.sigmoid(out).type(torch.float32)
            loss = loss_fn(torch.sigmoid(out), target)
            # loss = -1 * loss/2
            # loss = ((out - target)**2).sum()

            # p = torch.sigmoid(out) + 1e-3
            # loss = (target * torch.log(p) + (1-target) * torch.log(1-p)) * -1

            # loss = loss.sum()
            if i % 30 == 0:
                data.show_six_feature(torch.sigmoid(out).cpu().detach().numpy(), figure_num=2, title='outfeat', img=img)
                # data.show_six_feature(out.cpu().detach().numpy(), figure_num=2, title='outfeat')
                plt_save_feature_name = './save_feature_test/epoch_{}_iter_{}_outfeat.jpg'.format(e, i)
                plt.savefig(plt_save_feature_name)
                plt.close()

                # plt.figure()
                # plt.imshow(img[0][0].cpu().numpy())
                # plt.show()

                data.show_six_feature(target.cpu().numpy(), figure_num=1, title='tatget')
                plt_save_feature_name = './save_feature_test/epoch_{}_iter_{}_target.jpg'.format(e, i)
                plt.savefig(plt_save_feature_name)

                plt.close()

            # loss = (out - target).abs().sum()
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
    save_feat_path = r'F:\1_sheng\card_keypoint\save_feature_test'
    if os.path.exists(save_feat_path):
        img_list = os.listdir(save_feat_path)
        for l in img_list:
            os.remove(os.path.join(save_feat_path, l))
        print("success to remove all old feat")
    train()
