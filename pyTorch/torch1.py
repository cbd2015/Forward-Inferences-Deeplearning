#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------
# @Time    : 2019/6/5 23:31
# @Author  : chenbd
# @File    : torch1
# @Software: PyCharm Community Edition
# @license : Copyright(C), Your Company
# @Contact : 543447223@qq.com
# @Version : V1.1.0
# --------------------------------------------
# 中文显示乱码
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['font.family']='sans-serif'
# 功能     ：
#
#
# 算法链接：
#
#
# 实验结果：
#
#
#
#
# 执行时间：
# --------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class simpleconv3(nn.Module):
    def __init__(self):
        super(simpleconv3,self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, 2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 3, 2)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48 * 5 * 5 , 1200)
        self.fc2 = nn.Linear(1200 , 128)
        self.fc3 = nn.Linear(128 , 2)
    def forward(self , x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 48 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


mynet = simpleconv3(1, 10 ,1)  # （输入通道n_channels或者n_features, 隐藏层数，输出维度）
print(mynet)
# 随机梯度下降
optimizer = torch.optim.SGD( mynet.parameters(), lr=0.1 )
# MSE 损失函数优化  (a-b)^2
loss_func = torch.nn.MSELoss()
torch.nn.CrossEntropyLoss()

#------------Training----------------#
for epoch in range(2000):
    # 梯度下降之前要先将优化器设置为0
    optimizer.zero_grad()

    # forward + backward + optimizer
    pred = mynet(x) # 魔术方法
    loss = loss_func(pred, y)
    loss.backward()
    optimizer.step() # 开始更新参数

    if(epoch%100 ==0):
        print(loss.data)

#------------Testing----------------#
test_data = torch.tensor([-1.0])
pred = mynet(test_data)
print(test_data, pred.data)
print(pred)

