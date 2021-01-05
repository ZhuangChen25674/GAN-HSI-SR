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

save_cave_data()