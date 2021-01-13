#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   迭代产生训练数据
@File    :   data.py
@Time    :   2020/12/31 18:08:52
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2020/12/31 18:08:52
'''

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torch.nn.functional import interpolate


class LoadData(Dataset):

    def __init__(self,path,s=4):
        # num 31 512 512 
        self.data = np.load(path)
        self.data = torch.from_numpy(self.data)
        self.data /= 2**16 - 1
        # print(torch.max(self.data))

        #TODO: 先边缘裁剪 以获取HR
        shape = self.data.shape
        self.data = self.data[:,:,(s+6):shape[2]-(s+6),(s+6):shape[3]-(s+6)]
        
        # 取三张
        #32*3 31 144 144
        self.HR = torch.zeros((shape[0]*3,31,144,144))

        self.HR[:shape[0]] = self.data[:,:,:144,:144]
        self.HR[shape[0]:shape[0]*2] = self.data[:,:,144:144*2,144:144*2]
        self.HR[shape[0]*2:shape[0]*3] = self.data[:,:,144*2:144*3,144*2:144*3]

        # 得到LR图像 32*3 31 36 36 
        self.LR = self.down_sample(self.HR)

    
    def down_sample(self, data, s=4):
        #TODO: 添加高斯噪声(0.01) 并降采样 
        # data = data + 0.0000001*torch.randn(*(data.shape))

        data = interpolate(
            data,
            scale_factor=1/s,
            mode='bicubic'
        )

        return data


    def __len__(self):
        return self.HR.shape[0]

    
    def __getitem__(self,index):
        return self.LR[index], self.HR[index]

