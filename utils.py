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
from matplotlib import pyplot as plt
from G import OUT_DIR
import scipy.io as io

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
    for l in range(img1.size()[1]):

        mpsnr += 10. * torch.log10((torch.max(img1[:,l,:,:])**2) / torch.mean((img1[:,l,:,:] - img2[:,l,:,:]) ** 2))

    return mpsnr / img1.size()[1]

    # return 10. * torch.log10((torch.max(img1)**2) / torch.mean((img1 - img2) ** 2))

def SAM_GPU(y, x,shape=144):
    loss = 0

    for i in range(shape):
            for j in range(shape):
                fz = (x[:,:,i,j] * y[:,:,i,j]).sum()
                fm = torch.pow((x[:,:,i,j]*x[:,:,i,j]).sum(),0.5) * torch.pow((y[:,:,i,j]*y[:,:,i,j]).sum(),0.5)
                loss += torch.acos(fz/(fm + 1e-12))

    return (loss / (shape**2 * x.size()[0] )) * 180


def plot():

    path = '/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/train.log'
    sam_1 = []
    psnr_1 = []
    sam_2 = []
    psnr_2 = []

    with open(path,'r') as f:
   
        for line in f.readlines():

            line = line.strip()
            
            if 'psnr' in line and 'step : 1' in line :

                psnr_1.append(float(line.split(' ')[-5]))
                sam_1.append(float(line.split(' ')[-1]))

            if 'psnr' in line and 'step : 2' in line :

                psnr_2.append(float(line.split(' ')[-5]))
                sam_2.append(float(line.split(' ')[-1]))

    psnr = (np.array(psnr_1) + np.array(psnr_2)) / 2
    sam = (np.array(sam_1) + np.array(sam_2)) / 2

    epochs = [i for i in range(len(psnr))]

    fib_size = (5,4)
    fon_size = 12

    plt.figure(figsize=fib_size)
    plt.title('sam of every epoch',fontsize=fon_size)
    plt.xlabel('epoch',fontsize=fon_size)
    plt.ylabel('sam', fontsize=fon_size)
    plt.plot(epochs, sam, 'k.')
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1.1")
    plt.savefig(OUT_DIR.joinpath('sam.png'))


    plt.figure(figsize=fib_size)
    plt.title('psnr of every epoch',fontsize=fon_size)
    plt.xlabel('epoch',fontsize=fon_size)
    plt.ylabel('psnr', fontsize=fon_size)
    plt.plot(epochs, psnr, 'k.')
    plt.grid(True, linestyle = "-.", color = "k", linewidth = "1.1")
    plt.savefig(OUT_DIR.joinpath('psnr.png'))

def save_mat(l=31,w=144,h=144):

    path = '/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/weight/test_fake_hr.pth'
    base_path = Path('/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/data')
    data = torch.load(path)
    for i in range(1,7):
        img = torch.zeros([l,w*3,h*3])

        img[:,:w,:w] = data[3*i-3,:,:,:]
        img[:,w:w*2,w:w*2] = data[3*i-2,:,:,:]
        img[:,w*2:w*3,w*2:w*3] = data[3*i-1,:,:,:]

        img = img.numpy()
        img *= 2**16 - 1
        img = img.astype(np.int16)

        im = Image.fromarray(img[27])
        im.save(base_path.joinpath('img{}.png'.format(i)))

        io.savemat(base_path.joinpath('img{}.mat'.format(i)),{'data':img})

plot()


