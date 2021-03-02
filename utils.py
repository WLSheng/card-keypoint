import base64
import shutil
import cv2
import torch
import random
import json
import numpy as np
import os
import glob
import codecs


def cv_imread(file_path):
    # 读取中文路径下的图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def resize_512():
    cv2.namedWindow("test", 0)
    cv2.resizeWindow("test", 512, 512)
    num_dic = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}
    org_path = r'F:\1_sheng\答题卡拍照'
    ner_path = r'F:\1_sheng\card_512'
    all_index = 0
    txt_handle = open('./train.txt', 'w')

    for folder in range(1, 10, 1):
        second_path = os.path.join(org_path, str(folder))
        all_json_list = glob.glob(os.path.join(second_path, "*.json"))
        for i, one_json_path in enumerate(all_json_list):
            with codecs.open(one_json_path, 'r', encoding='utf-8', errors='ignore') as f:
                jsondict = json.load(f)
                imagePath = one_json_path.replace(".json", ".jpg")
                if not os.path.exists(imagePath):
                    imagePath = one_json_path.replace(".json", ".png")
                    print(imagePath, "no exit, png")
                elif not os.path.exists(imagePath):
                    imagePath = one_json_path.replace(".json", ".jpeg")
                    print(imagePath, "no exit, jpeg")
                elif not os.path.exists(imagePath):
                    print(imagePath, "no exit, 注意后缀名---------------")
                    break
                pic = jsondict['imageData']
                pic = base64.b64decode(pic)
                # print(pic)
                pic = np.fromstring(pic, np.uint8)
                img = cv2.imdecode(pic, cv2.COLOR_BGR2RGB)
                # img = cv_imread(imagePath)
                height = img.shape[0]
                width = img.shape[1]
                resize_img = cv2.resize(img, (512, 512))
                gt_content = ""
                for one_point in jsondict['shapes']:
                    label = one_point['label']
                    # print(label, "one_point['points']: ", one_point['points'])
                    x = int(one_point['points'][0][0])
                    y = int(one_point['points'][0][1])
                    label_index = num_dic[label]
                    # print(x/width, y/height)
                    x = int((x/width)*512)
                    y = int((y/height)*512)
                    # resize_img = cv2.circle(resize_img, (x, y), 3, [0, 255, 0], 5)
                    gt_content = gt_content + ' {},{},{}'.format(str(label_index), str(x), str(y))

                img_write_path = os.path.join(ner_path, str(folder), os.path.split(imagePath)[1])
                cv2.imencode('.jpg', resize_img)[1].tofile(img_write_path)
                write_content = img_write_path + gt_content
                txt_handle.write(write_content + '\n')
                print(all_index, write_content)
                # cv2.imshow("test", resize_img)
                # cv2.waitKey(0)
            all_index += 1
    txt_handle.close()


def draw_img():
    txt_handle = open('./train.txt', 'r')
    all_content = txt_handle.readlines()
    for one_content in all_content:
        # one_content = r'F:\1_sheng\card_512\1\QQ图片20210112171437.jpg 0,91,74 1,382,67 2,77,148 3,393,142 4,37,428 5,429,428'
        con_list = one_content.split(" ")
        img = cv_imread(con_list[0])
        for i in range(1, len(con_list), 1):
            label_list = con_list[i].split(",")
            label = label_list[0]
            x = int(label_list[1])
            y = int(label_list[2])
            # img = cv2.circle(img, (x, y), 2, [0, 255, 0], 2)
            # img = cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 1)
            img = cv2.resize(img, (32, 32))
        cv2.imshow("test", img)
        cv2.waitKey(0)


def seg_pth_to_my_pth():
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # state_dict = torch.load('./drn_d_38_cityscapes.pth')
    # for k, v in state_dict.items():
    #     name = k.replace("base.", "layer")
    #     name = name.replace("seg", "fc")
    #     if 'up' in name:
    #         print("pass a up")
    #         continue
    #     print(k, name)
    #     new_state_dict[name] = v
    # torch.save(new_state_dict, './new_drn_d_38_seg_pretrain.pth')
    import drn, models
    model = drn.drn_d_38()
    model.fc = None
    for k, v in model.state_dict().items():
        print("org model key:", k)

    model.load_state_dict(torch.load('./new_drn_d_38_seg_pretrain.pth'))
    img = torch.randn((1, 3, 512, 512))
    model.fc = models.conv1x1(512, 6)
    out = model(img)
    print(model)
    print(out.shape)
    # return


def rename_all_name_accord_txt():
    txt_list = open("./train.txt", 'r').readlines()
    save_folder = r'F:\1_sheng\card_512_ubuntu\\'
    new_txt = open("new_train.txt", 'w')
    for i, one_txt in enumerate(txt_list):
        one_txt_list = one_txt.replace("\n", "").split(" ")
        org_img_path = one_txt_list[0]
        # folder = os.path.split(org_img_path)
        folder = org_img_path.split("\\")[-2]
        new_img_path = save_folder + folder + "\\" + folder + "_{}.jpg".format(str(i).zfill(3))
        # print(org_img_path, new_img_path)
        shutil.copy(org_img_path, new_img_path)
        one_txt_list[0] = new_img_path
        write_ont_txt = " ".join(one_txt_list) + "\n"
        print(write_ont_txt)
        new_txt.write(write_ont_txt)
        # exit(0)
    new_txt.close()


if __name__ == '__main__':
    # resize_512()
    # draw_img()
    # seg_pth_to_my_pth()
    rename_all_name_accord_txt()
    pass
