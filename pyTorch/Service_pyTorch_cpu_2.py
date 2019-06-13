import base64
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入预先训练的模型
# import torchvision.models as models


from PIL import Image
from flask import Flask, request

from torch.autograd import Variable
from torchvision import  transforms
from collections import OrderedDict
# app.debug = True
app = Flask (__name__)
ALLOWED_EXTENSIONS = set (['jpg', 'png', 'jpeg'])

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
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


def run_inference_on_image(imagePath):
    """
    图像尺寸变换
    图像归一化
    前向传播网络,预测图像类别
    :param image:
    :return:
    """
    # data_transforms = transforms.Compose ([
    #     transforms.Resize (224),
    #     transforms.ToTensor (),
    #     # transforms.Normalize ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])

    net = MyNet () # 卷积神经网络模型
    net.eval()
    modelpath = "./model.pth"
    # net.load_state_dict (torch.load (modelpath, map_location=lambda storage, loc:storage.cuda).eval())
    pretrain = torch.load(modelpath)


    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.50, 0.50, 0.50], std=[1, 1, 1])
    ])
    # new_state_dict = OrderedDict ()
    # for k, v in pretrain.items ():
    #     if k == 'state_dict':
    #         state_dict = OrderedDict ()
    #         for keys in v:
    #             name = keys[7:]  # remove `module.`
    #             state_dict[name] = v[keys]
    #         new_state_dict[k] = state_dict
    #     else:
    #         new_state_dict[k] = v
    # net.load_state_dict (new_state_dict['state_dict'])

    # imagepath = imagePath # torch参数模型的路径
    image = Image.open (imagePath)
    imgblob = data_transforms (image) #.unsqueeze (0)
    print(type(imgblob))
    # imgblob = Variable (imgblob)
    torch.no_grad ()
    predict = F.softmax (net (imgblob))
    print (predict)


@app.route ('/', methods=['POST', 'GET'])
def hello_world():
    resultString = "test application function"
    print(resultString)
    return "test application return success"


@app.route ('/demo', methods=['POST', 'GET'])
def imageRcognized():
    # 网络请求数据获取和转换
    b64datas = request.form.get ('b64data')
    imgdata = base64.b64decode (b64datas)
    print ( imgdata )
    # ricename = request.form.get("ricename")
    # imgdata2 = base64.b64decode (ricename)
    # print(ricename)

    filename = str(int (round (time.time () * 100000)))
    file = open(filename + ".jpeg", 'wb')
    file.write(imgdata)
    file.close( )

    # forward inference预测识别结果输出
    results = run_inference_on_image (filename + ".jpeg")

    # os.remove(filename+".jpeg")

    return results


if __name__ == '__main__':
    """
        host service application
    """
    app.run (threaded=True, host='127.0.0.0', port=9012)
