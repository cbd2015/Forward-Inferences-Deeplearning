#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------
# @Time    : 2019/5/28 20:25
# @Author  : chenbd
# @File    : simple_regression
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
class SimpleNet(torch.nn.Module):
    def __init__(self,n_features, n_hidden, n_output):
        super(SimpleNet, self).__init__()
        ## 神经网络层与层之间的关系
        """
            1、定义网络之间的主干模块，共性模块的定义
               但是___init__这个函数内部没有把网络之间串起来
            2、需要通过forward将网络串起来
        """
        # from input layer -> hidden layer
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        # from hidden layer -> output layer
        self.predict  = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        """
            将网络之间的关系、神经网络模块之间串接起来
            复用__init__函数中的共性模块，将网络拼接搭建起来
            forward() 是控制网络前向传播实施的方法函数
                1) 清零
                2）forward
                3）backward
                4）optimize
        :param x:
        :type x:
        :return:
        :rtype:
        """
        hidden_resule = self.hidden(x)
        x = F.relu(hidden_resule)
        x = self.predict(x)
        return  x

mynet = SimpleNet(1, 10 ,1)  # （输入通道n_channels或者n_features, 隐藏层数，输出维度）
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



















