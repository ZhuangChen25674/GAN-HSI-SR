#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   训练GAN网络
@File    :   train.py
@Time    :   2021/01/05 21:17:38
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2021/01/05 21:17:38
'''

import torch
import torch.nn as nn
import torch.optim as optim
from net import Generator, Discriminator,Spe_loss,TVLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
from G import *
from data import LoadData
from utils import SAM_GPU, PSNR_GPU
from pathlib import Path


EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device is {}'.format(device))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    g_model = Generator(BATCH_SIZE).to(device)
    d_model = Discriminator(BATCH_SIZE).to(device)

    #专门儿编写的生成器损失函数
    # g_criterion = Loss()
    d_criterion = nn.BCELoss()
    criterion = {
        'l1' : nn.L1Loss(),
        'ltv' : TVLoss(),
        'ls' : Spe_loss(),
        'la' : nn.BCELoss(),
    }


    g_optimizer = optim.Adam(
        g_model.parameters(),
        lr = LR
    )
    d_optimizer = optim.Adam(
        d_model.parameters(),
        lr = LR
    )

    # best_weight = {
    #     'g_weight': copy.deepcopy(g_model.state_dict()),
    #     'd_weight': copy.deepcopy(d_model.state_dict())
    # }

    sorce = {
        'd_loss':0.0,
        'g_loss':0.0,
        'real_sorce':0.0,
        'fake_sorce':0.0
    }


    best_sorce = {
        'psnr'  : 0.0,
        'sam'   : 180.0,
        'epoch' : 0,
    }

    for epoch in range(EPOCHS):

        train_data = DataLoader(
            LoadData(TRAIN_DATA_PATH),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers= 2, 
            pin_memory= True,
            drop_last= True,
        )

        val_data = DataLoader(
            LoadData(VAL_DATA_PATH),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers= 2, 
            pin_memory= True,
            drop_last= True,
        )

        count = 0
        for lr, hr in train_data:
            # bs 31 36 36  / bs 31 144 144

            lr = lr.reshape((lr.shape[0],1,lr.shape[1],lr.shape[2],lr.shape[3]))
            lr = lr.to(device)
            hr = hr.reshape((hr.shape[0],1,hr.shape[1],hr.shape[2],hr.shape[3]))
            hr = hr.to(device)

            real_labels = torch.ones(BATCH_SIZE).to(device)
            fake_labels = torch.zeros(BATCH_SIZE).to(device)

            # ================================================ #
            #                训练判别器部分                     #
            # ================================================ #
            
            #计算real标签 也就是hr的损失
            output = d_model(hr)
            d_loss_real = criterion['l1'](torch.squeeze(output),real_labels)
            real_sorce = output
            sorce['real_sorce'] = real_sorce.mean().item()

            #计算fake标签  也就是lr的损失
            fake_hr = g_model(lr)
            output = d_model(fake_hr)
            d_loss_fake = criterion['l1'](torch.squeeze(output),fake_labels)
            # print(torch.squeeze(output))
            fake_sorce = output
            sorce['fake_sorce'] = fake_sorce.mean().item()

            # 反向传播 参数更新部分
            d_loss = d_loss_real + d_loss_fake
            sorce['d_loss'] = d_loss.item()
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()


            # ================================================ #
            #                训练生成器部分                     #
            # ================================================ #
            
            fake_hr = g_model(lr)
            output = d_model(fake_hr)
            
            #损失计算
            # print(fake_hr.shape,hr.shape)
            fake_hr = torch.squeeze(fake_hr)
            hr = torch.squeeze(hr)
            g_loss = criterion['l1'](fake_hr,hr) + \
                + 1e-2 * criterion['ls'](fake_hr,hr)
            # print(criterion['l1'](fake_hr,hr),criterion['ltv'](fake_hr),criterion['ls'](fake_hr,hr))
            sorce['g_loss'] = g_loss


            #反向传播 优化
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            print('EPOCH : {} step : {} \
d_loss : {:.4f} g_loss : {:.4f} \
real_sorce {:.4f} fake_sorce {:.4f}'.format(
                    epoch,count+1,
                    sorce['d_loss'],sorce['g_loss'], 
                    sorce['real_sorce'],sorce['fake_sorce']
                ))
            count += 1
            #  训练完成  开始验证


        g_model.eval()
        d_model.eval()
        val_count = 0
        val_psnr = 0
        val_sam = 0
        for lr,hr in val_data:

            lr = lr.reshape((lr.shape[0],1,lr.shape[1],lr.shape[2],lr.shape[3]))
            lr = lr.to(device)
            hr = hr.reshape((hr.shape[0],1,hr.shape[1],hr.shape[2],hr.shape[3]))
            hr = hr.to(device)

            with torch.no_grad():

                fake_hr = g_model(lr)
                fake_hr = torch.squeeze(fake_hr)
                hr = torch.squeeze(hr)

                psnr = PSNR_GPU(hr.cpu(),fake_hr.cpu())
                val_psnr += psnr
                sam = SAM_GPU(hr,fake_hr)
                val_sam += sam

                print('val epoch : {} step : {} psnr : {:.4f}  sam : {:.4f}'.format(
                    epoch,val_count+1,psnr,sam
                ))

                val_count += 1

        print('val averagr psnr : {:.4f} sam : {:.4f}'.format(
            val_psnr/(val_count),
            val_sam/(val_count))
            )

        if val_psnr/(val_count+1) > best_sorce['psnr']:
            #以psnr为主  找到最好的 保存下来
            best_sorce['psnr'] = val_psnr/(val_count)

            torch.save(copy.deepcopy(g_model.state_dict()),OUT_DIR.joinpath('g_model.pth'))
            torch.save(copy.deepcopy(d_model.state_dict()),OUT_DIR.joinpath('d_model.pth'))


    