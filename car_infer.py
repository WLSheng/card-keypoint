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
from data import cv_imread, make_target, cardDataset, collate_fn, Normalize, show_six_feature
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F


def card_keypoints_init(model_path):

    transform = transforms.Compose([transforms.ToTensor()])
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Conv2d(512 * 4, 6, kernel_size=1, stride=1, bias=True)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    return transform, model


def card_infer(transform, model, test_img):
    t2 = time.time()
    test_img = cv2.resize(test_img, (512, 512))
    draw_img = test_img.copy()
    test_img = transform(test_img)
    test_img = torch.unsqueeze(test_img, 0).cuda()
    out = model(test_img)
    out = torch.sigmoid(out)
    out = out.cpu().detach().numpy()
    print("infer time: ", time.time() - t2)
    show_six_feature(out, figure_num=1, img=test_img)
    plt.figure(2)
    plt.subplot(1, 2, 1)
    show_out = np.max(out[0], axis=0)
    plt.imshow(show_out)
    point_result_dict = {}
    for i in range(out.shape[1]):
        one_layer = out[0][i]
        max_point_value = np.max(one_layer)
        if max_point_value > 0.6:
            coor = np.where(one_layer == max_point_value)
            print(i, coor[0][0], coor[1][0], "max: ", np.max(one_layer))
            x = coor[1][0] * 8 * 2
            y = coor[0][0] * 8 * 2
            draw_img = cv2.putText(draw_img, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255], 2)
            draw_img = cv2.circle(draw_img, (x, y), 3, [0, 255, 0], 2)
            point_result_dict[str(i)] = [x, y]
    plt.subplot(1, 2, 2)
    plt.imshow(draw_img)
    return point_result_dict


if __name__ == '__main__':
    model_path = r'./save_model/epoch_06.pth'
    t1 = time.time()
    transform, model = card_keypoints_init(model_path)
    print("init time: ", time.time()-t1)
    train_content = open(config.train_txt, 'r').readlines()
    for i, one_content in enumerate(train_content):
        content_list = one_content.split(" ")
        img = cv_imread(content_list[0])
        point_result = card_infer(transform, model, img)
        print("point_result: ", point_result)
        plt.show()
    print("$_$")
