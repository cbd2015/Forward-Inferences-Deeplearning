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

# app.debug = True
app = Flask (__name__)
ALLOWED_EXTENSIONS = set (['jpg', 'png', 'jpeg'])


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


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
    data_transforms = transforms.Compose ([
        transforms.Resize (224),
        transforms.ToTensor (),
        # transforms.Normalize ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    net = MobileNetV2 () # 卷积神经网络模型
    modelpath = "./mobilenet_v2-b0353104.pth"

    # net.load_state_dict (torch.load (modelpath, map_location=lambda storage, loc:storage.cuda).eval())
    # pretrain = torch.load(modelpath, map_location=lambda storage,loc:storage)
    # from collections import OrderedDict
    # new_state_dict = OrderedDict ()
    # for k, v in pretrain.items ():
    #     if  k == 'state_dict':
    #         state_dict = OrderedDict ()
    #         for keys in v:
    #             name = keys[7:]  # remove `module.`
    #             state_dict[name] = v[keys]
    #         new_state_dict[k] = state_dict
    #     else:
    #         new_state_dict[k] = v
    # net.load_state_dict (new_state_dict['state_dict'])
    net.load_state_dict(torch.load(modelpath))
    net.eval()
    # imagepath = imagePath # torch参数模型的路径
    image = Image.open (imagePath)
    imgblob = data_transforms (image).unsqueeze (0)
    imgblob = Variable (imgblob)
    torch.no_grad ()
    predict = F.softmax (net(imgblob),dim=1)
    temp = (predict.detach().numpy()[0])
    tensorList = temp.tolist()
    return str(tensorList.index(max(tensorList))+1) + ":" + str( max(tensorList) )
    # return str(max(predict.detach().numpy()[0]))+":"+ str(index(max(predict.detach().numpy()[0])))

@app.route ('/', methods=['POST', 'GET'])
def hello_world():
    resultString = "test application function"
    print(resultString)
    return "test application return success"


@app.route ('/demo', methods=['POST', 'GET'])
def imageRcognized():
    b64datas = request.form.get ('b64data')
    # use this form function to add other parameters
    imgdata = base64.b64decode (b64datas)
    print ( imgdata )

    ricename = request.form.get("ricename")
    print(ricename)

    filename = str(int (round (time.time () * 100000)))
    file = open(filename + ".jpeg", 'wb')
    file.write(imgdata)
    file.close( )

    # forward inference
    results = run_inference_on_image (filename + ".jpeg")

    os.remove(filename+".jpeg")

    return results


if __name__ == '__main__':
    """
        host service application
    """
    app.run (threaded=True, host='127.0.0.1', port=9013)
