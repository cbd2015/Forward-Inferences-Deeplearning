#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------
# @Time    : 2019/3/6 21:42
# @Author  : chenbd
# @File    : post
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
# 算法链接：
#
# 实验结果：
## POST请求：
# post请求提交用户信息到服务器
# 执行时间：
# --------------------------------------------
import os
import requests
import base64
import time
from flask import Flask
import requests


app = Flask(__name__)

def jpg2base64(picName):
    """
    : jpg图片转base64，
    : url携带base数据请求
    :param picName:
    :return:
    """
    with open(picName, 'rb') as f:
        image_data = f.read()
        # 编码转换为字符串
        base64_data = str(base64.b64encode(image_data), "utf-8")

        return base64_data


def postImage(strBase64):
    """
       CGI(Common Gateway Interface)是HTTP服务器运行的程序
       通过Internet把用户请求送到服务器
       服务器接收用户请求并交给CGI程序处理
       CGI程序把处理结果传送给服务器
     """
    # 食材，测试
    url = "http://127.0.0.1:9013/demo"
    # url = "http://119.29.2.159:9012/demo"
    # Post参数
    # strBase64 ="asdfasdfas2342"
    # print(strBase64)
    body = {'b64data':strBase64,'ricename':"chenbd"}
    headers = {'content-type': "application/x-www-form-urlencoded; charset=utf-8"}

    """
        request.args.get('key')可以获取到单个的值，
        requestValues = request.args可以获取get请求的所有参数返回值是ImmutableMultiDict类型,
        requestValues.to_dict()将获得的参数转换为字典
    """
    response = requests.post (url,data=body,headers=headers)
    print(response)
    # 暴露请求状态
    response.raise_for_status ()
    # 请求不成功不是200，则引发HTTPError异常
    response.encoding = response.apparent_encoding
    # 返回报文信息
    print(response.text)
    return response.text

def hi_person():
    imagePath = "./images/simiao/toilettissue.jpg"
    strBase64 = jpg2base64 (imagePath)
    responseText = postImage (strBase64)
    return responseText
    # requests.post(url_for("http://127.0.0.1:5000/demo", _external=True), data=form)

if __name__ == "__main__":
    # t1 = time.time()
    # for i in  range(10):
    hi_person()
    # t2 = time.time()
    # print((t2-t1)/10)
    """
        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        import requests
        import json
        url = 'http://official-account/app/messages/group'
        body = {"type": "text", "content": "测试文本", "tag_id": "20717"}
        headers = {'content-type': "application/json", 'Authorization': 'APP appid = 4abf1a,token = 9480295ab2e2eddb8'}
        #print type(body)
        #print type(json.dumps(body))
        # 这里有个细节，如果body需要json形式的话，需要做处理
        # 可以是data = json.dumps(body)
        response = requests.post(url, data = json.dumps(body), headers = headers)
        # 也可以直接将data字段换成json字段，2.4.3版本之后支持
        # response  = requests.post(url, json = body, headers = headers)
        # 返回信息
        print response.text
        # 返回响应头
        print response.status_code
    """
