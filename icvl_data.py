#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

#  要 改count
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
import h5py

class LoadData(Dataset):

    def __init__(self,path,label,s=4,channels=31,fis=144):
        # num 31 512 512 
        if label == 'train':
            num = 2187
        elif label == 'val':
            num = 558
        else:
            num = 576
        self.HR = torch.zeros([num, channels, fis, fis])

        count = 0
        for i in range(len(path)):

            img = h5py.File(path[i], 'r')['rad']
            img = np.array(img)
            img /= 4095.0
            img = torch.tensor(img)

            print(img.size()[1],img.size()[2])
            for x in range((s+6), img.size()[1] -(s+6)-fis, fis):
                for y in range((s+6), img.size()[2] -(s+6)-fis, fis):
                    
                    self.HR[count] = img[:,x:x+fis,y:y+fis]
                    count += 1

        print('safasfasfsdfds:{}',format(count))
        self.LR = self.down_sample(self.HR)
    
    def down_sample(self, data, s=4):
        #TODO: 添加高斯噪声(0.01) 并降采样 
        # data = data + 0.0000001*torch.randn(*(data.shape))

        data = interpolate(
            data,
            scale_factor=1/s,
            mode='bicubic',
            align_corners=True
        )

        return data


    def __len__(self):
        return self.HR.shape[0]

    
    def __getitem__(self,index):
        return self.LR[index], self.HR[index]

