#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   辅助函数
@File    :   utils.py
@Time    :   2020/12/31 16:26:28
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2020/12/31 16:26:28
'''


from pathlib import Path
from PIL import Image
import numpy as np
import torch.nn as nn
import torch

def save_cave_data():
    # 生成{train val test}.npy 文件

    path = Path('/home/hefeng/data1/HSI-SR/DataSet/CAVE')
    data = np.zeros((32,31,512,512))

    for i,p in enumerate(path.iterdir()) :
        for j in range(31):
            
            img_path = p.joinpath(p.parts[-1],p.parts[-1]+'_{:0>2d}.png'.format(j+1))
            print(img_path)
            img = Image.open(img_path)
            img = np.array(img)

            # 有特殊shape的图片
            if len(img.shape) != 2:
                data[i][j] = img[:,:,0]
            
            if len(img.shape) == 2:
                data[i][j] = img
            print((i,j))

    print(data[:20].shape,data[20:26].shape,data[26:].shape)
    np.save('train.npy',data[:20])
    np.save('val.npy',data[20:26])
    np.save('test.npy',data[26:])

def PSNR_GPU(img1, img2):
    mpsnr = 0
    for l in range(img1.size[1]):

        mpsnr += 10. * torch.log10(torch.max(img1)**2 / torch.mean((img1 - img2) ** 2))

    return mpsnr / img1.size[1]

def SAM_GPU(im_true, im_fake):
    loss = 0

    for i in range(im_true.size[3]):
            for j in range(im_true.size[3]):
                fz = (x[:,:,i,j] * y[:,:,i,j]).sum()
                fm = torch.pow((x[:,:,i,j]*x[:,:,i,j]).sum(),0.5) + torch.pow((y[:,:,i,j]*y[:,:,i,j]).sum(),0.5)
                loss += torch.acos(fz/(fm + 1e-12))

    return loss / (im_true.size[3]**2)

    # C = im_true.size()[1]
    # H = im_true.size()[2]
    # W = im_true.size()[3]
    # esp = 1e-12
    # Itrue = im_true.clone()#.resize_(C, H*W)
    # Ifake = im_fake.clone()#.resize_(C, H*W)
    # nom = torch.mul(Itrue, Ifake).sum(dim=0)#.resize_(H*W)
    # denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
    #               Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
    # denominator = denominator.squeeze()
    # sam = torch.div(nom, denominator).acos()
    # sam[sam != sam] = 0
    # sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
    # return sam_sum / 8