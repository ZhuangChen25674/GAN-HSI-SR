#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   网络架构的复现
@File    :   net.py
@Time    :   2020/12/25 14:05:09
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2020/12/25 14:05:09
'''

import torch
import torch.nn as nn

KERNEL_SIZE = (5,3,3)
PAD_SIZE = (2,1,1)

class Attention(nn.Module):

    # band attention 复现
    def __init__(self,bs, c, l=31, h=36, w=36):
        super(Attention,self).__init__()
        # SET Padding=0 Stride=1
        # TODO: INPUT_SHAPE (B_S, 32 , 31 , H, W)
        self.shape = [bs,c,l,h,w]

        self.conv_1 = nn.Sequential(
            nn.Conv3d(32, 32, KERNEL_SIZE, 1, (2,1,1)),
            nn.BatchNorm3d(32),
            nn.PReLU(),

            nn.Conv3d(32, 32, KERNEL_SIZE,1,(2,1,1)),
            nn.BatchNorm3d(32),
            nn.PReLU(),
        )
        # TODO: 3D均值池化 
        self.avg_poll =  nn.AvgPool3d(
            (1, h, w), 1
        )


        # TODO: 调整shape = (BS, 32, 31)
        self.conv_2 = nn.Sequential(
            nn.Conv1d(32, 32, 24),
            nn.PReLU(),

        #   TODO: 反卷积 增大缩小的特征图
            nn.ConvTranspose1d(32,32,5,4,1),
            nn.Sigmoid()
        )


    def forward(self,x):
        # TODO: INPUT_SHAPE (B_S, 32 , 31 , H, W)

        x1 = self.conv_1(x) #bs 32 31 h w
        x2 = self.avg_poll(x1) #bs 32 31 1 1
        
        x2_1 = x2.resize(self.shape[0], self.shape[1], self.shape[2]) # bs 32 31

        x3 = self.conv_2(x2_1) # bs 32 31
        x3_1 = x3.resize(self.shape[0], self.shape[1], self.shape[2], 1, 1) # bs 32 31 1 1
        
        x4 = x1 * x3_1

        y = x + x4

        return y



class Generator(nn.Module):

    # 生成器网络复现

    def __init__(self,bs, c=1, l=31, h=36, w=36):
        super(Generator,self).__init__()
        #TODO: input_shape (bs, 1, 31, h, w)
        self.shape = [bs,c,l,h,w]

        self.conv_1 = nn.Sequential(
            nn.Conv3d(1, 32, KERNEL_SIZE, 1, (2,1,1)),
            nn.PReLU()
        )

        self.attn_1 = Attention(bs, 32, l, h, w)

        self.conv_2 = nn.Sequential(
            nn.Conv3d(32, 32, KERNEL_SIZE, 1, (2,1,1)),
            nn.PReLU()
        )
        # TODO: 反卷积大小计算!
        self.conv_3 = nn.Sequential(
            nn.ConvTranspose3d(32,32,(3,6,6),(1,2,2),(1,2,2)),
            nn.PReLU(),

            nn.ConvTranspose3d(32,32,(3,6,6),(1,2,2),(1,2,2)),
            nn.Tanh(),

            nn.Conv3d(32,1,KERNEL_SIZE,1,(2,1,1)),
        )

    def forward(self,x):
        
        x1 = self.conv_1(x)

        x2 = self.attn_1(x1)

        x2_1 = self.conv_2(x2)

        x3 = x2_1 + x1

        y = self.conv_3(x3)

        return y


class Discriminator(nn.Module):

    #对抗器复现
    def __init__(self,bs, c=1, l=31, h=144, w=144):
        super(Discriminator,self).__init__()
        #TODO:通过单边填充 实现恰好减半
        self.conv = nn.Sequential(
            #1
            nn.Conv3d(1,32,KERNEL_SIZE,1,(2,1,1)),
            nn.PReLU(),

            #TODO: 2通过单边pad完成减半的目的
            nn.ConstantPad3d((1,0,1,0,1,2),0),
            nn.Conv3d(32,32,KERNEL_SIZE,2),
            nn.BatchNorm3d(32),
            nn.PReLU(),
            #3
            nn.Conv3d(32,64,KERNEL_SIZE,1,(2,1,1)),
            nn.BatchNorm3d(64),
            nn.PReLU(),
            #4
            nn.ConstantPad3d((1,0,1,0,1,2),0),
            nn.Conv3d(64,64,KERNEL_SIZE,2),
            nn.BatchNorm3d(64),
            nn.PReLU(),
            #5
            nn.Conv3d(64,128,KERNEL_SIZE,1,(2,1,1)),
            nn.BatchNorm3d(128),
            nn.PReLU(),
            #6
            nn.Conv3d(128,128,KERNEL_SIZE,1,(2,1,1)),
            nn.BatchNorm3d(128),
            nn.PReLU(),
            #7  TODO: 4倍缩小 l h w = 8 36 36
            #  --> bs 128 1 1 1
            nn.AvgPool3d((7,36,36),1),
            nn.Conv3d(128,256,(1,1,1),1),
            nn.PReLU(),

            nn.Conv3d(256,1,(1,1,1),1),
            nn.Sigmoid(),
        )


    def forward(self,x):

        y = self.conv(x)


        # 注意 返回的y的shape 是5维的!!!!!
        return y


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class Spe_loss(nn.Module):
    def __init__(self):
        super(Spe_loss,self).__init__()

    def forward(self,x,y,shape=144):

        loss = 0
        for i in range(shape):
            for j in range(shape):
                fz = (x[:,:,i,j] * y[:,:,i,j]).sum()
                fm = torch.pow((x[:,:,i,j]*x[:,:,i,j]).sum(),0.5) * torch.pow((y[:,:,i,j]*y[:,:,i,j]).sum(),0.5)
                loss += torch.acos(fz/fm)

        return loss / (shape**2)





class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()

    def forward(self,x,y,l1=1,l2=1e-2,l3=1e-3):

        l1 = nn.L1Loss()
        l1 = l1(x,y)

        ltv = TVLoss()
        ltv = TVLoss(x)
        print(ltv)
        #1
        ls = l1 + 1e-6*(ltv.data)
        #2
        le = Spe_loss()
        le = le(x,y)
        # #3
        # la = nn.BCELoss()
        
        # if labels:
        #     l = torch.ones(bs,1)
        #     la = la(y,l)

        # else:
        #     l = torch.zeros(bs,1)
        #     la =  la(x,l)

        return l1*ls + l2*le 