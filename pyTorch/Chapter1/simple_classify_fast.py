#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------
# @Time    : 2019/5/28 21:51
# @Author  : chenbd
# @File    : simple_classify_fast
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
from torch.nn import functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1) # x[n,d]构建矩阵
print("x",x)
print(x.shape,type(x))
y = x.pow(2) # y = x^2函数关系
## 非线性回归函数
# plt.scatter(x.data, y.data, s=10, cmap='autumn')
# plt.show()
"""
    构建一个回归的神经网络，包含一个隐藏层
"""


mynet = torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2)
)
print(mynet)
optimizer = torch.optim.SGD(mynet.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(1000):
    out = mynet(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%10 ==0:
        acc = get_acc(y, out)



















