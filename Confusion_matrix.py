# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:24:41 2021

@author: wby
"""
from torch.nn import init
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import Image

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

import AlexNetmodel
#import ResNet18model
#import VGG16model
import GoogleNet
#import tcmodel
#import ResNetmodel
#import SENet
#import SKNet
#import cliquenet
#import ResNeXt
import MyNet
import MyNet_VGG
import MyNet_Google

import MyNet_resnet
import MyNet_Google_Resnet
import MyNet_RseSe_bloock_net
import MyNet_Google_RseSe_bloock_net

#import modle.Google_Resnet_X1

#import Net.AlexNetmodel
#import Net.cliquenet
#import Net.GoogleNet
import ResNetmodel
#import Net.VGGmodel

#import MyNet_RseSe_bloock_net2

num_classes = 3#标签的种类数
num_epochs = 2 #训练的总循环周期
batch_size = 32  #一个撮（批次）的大小，64张图片
data_dir = 'data'

image_size = 224

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                    transforms.Compose([
                                  #       transforms.Grayscale(num_output_channels=1),    
                                        transforms.Resize(256),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
                                    )
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 10, shuffle = True, num_workers=0)  #设成20 LOSS 曲线收敛
#val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 10, shuffle = True, num_workers=0)  
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 10, shuffle = True, num_workers=0)
# 读取得出数据中的分类类别数
num_classes = len(test_dataset.classes)
print(len(test_dataset))
use_cuda = torch.cuda.is_available()
# 当可用GPU的时候，将新建立的张量自动加载到GPU中
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor

def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))   
    #由于在读入图像的时候所有图像的色彩都标准化了，因此我们需要先调回去
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  
    #将图像绘制出来
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 暂停一会是为了能够将图像显示出来。
#获取第一个图像batch和标签
images, labels = next(iter(test_loader))
# 将这个batch中的图像制成表格绘制出来
out = torchvision.utils.make_grid(images)
imshow(out, title=[test_dataset.classes[x] for x in labels])

#----------------------------混淆矩阵---------------------------------
def plot_confusion_matrix(cm, labels_name, title):
  #  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)  
   # plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像，设置颜色
   # plt.title(title)    # 图像标题
  #  figsize = 8,8
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 15,
             }

 
 
  #  figure, ax = plt.subplots(figsize=figsize)
    
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90, fontsize=13)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontsize=13)    # 将标签印在y轴坐标上
    plt.ylabel('True label',font1)    
    plt.xlabel('Predicted label',font1)
    for first_index in range(len(cm)):    #第几行
        for second_index in range(len(cm[first_index])):    #第几列
            plt.text(first_index, second_index, cm[first_index][second_index],
                     fontsize=11, va='center', ha='center')
    plt.tight_layout()   
    plt.savefig('cm.png', format='png')
    plt.colorbar()
    plt.show()

    
#-----------------------------------这里放model------------------------
    
def rightness(predictions, labels):
#    计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素
'''
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix
'''
#net = SENet.Senet()
#net = SKNet.SKNet50()
#net = ResNeXt.Resnext(2)
net = GoogleNet.GoogLeNet()
#net = AlexNetmodel.AlexNet(num_classes=3)
#net = ResNetmodel.ResNet50()
#net = tcmodel.Resnet()
#net = VGG16model.VGG16()
#net = MyNet.MyNet(num_classes = num_classes)
#net = MyNet_VGG.VGG18()
#net = MyNet_Google.My_GoogLeNet(num_classes = num_classes)
#net = MyNet_Google_RseSe_bloock_net.My_Google_Resnet(num_classes = num_classes)

#net = MyNet_RseSe_bloock_net.Resnet(num_classes = num_classes)
#net = MyNet_resnet.Resnet(num_classes = num_classes)

#net = MyNet_Google_Resnet.My_Google_Resnet(num_classes = num_classes)
#net = Net.AlexNetmodel.AlexNet(num_classes = num_classes)
#net = Net.VGGmodel.VGG19()

#net = Net.GoogleNet.GoogLeNet()
#net = modle.Google_Resnet_X2.My_Google_Resnet(num_classes = num_classes)

#net.load_state_dict(torch.load('NetWeight_zz/GoogleNet.pth'))
#net.load_state_dict(torch.load('NetWeight/MyNet.pth'))
#net.load_state_dict(torch.load('NetWeight/MyVGGNet.pth'))
#net.load_state_dict(torch.load('NetWeight/MyGoogle_ResNet.pth'))

#net.load_state_dict(torch.load('NetWeight/Mynet_resnet_7_n.pth'))
#net.load_state_dict(torch.load('NetWeight/MyResSeNet_37.pth'))

#net.load_state_dict(torch.load('NetWeight/Google_Resnet_X2.pth'))
#net.load_state_dict(torch.load('NetWeight_5z/MyNet_RseSe_bloock_Net_1.pth'))
#net.load_state_dict(torch.load('NetWeight_zz/MyNet_RseSe_bloock_Net.pth'))
#net.load_state_dict(torch.load('NetWeight_zz/MyNet_ResSeGoogle_bloock_Net2.pth'))
#net.load_state_dict(torch.load('NetWeight_zz/MyNet_RseSe_bloock_Net2.pth'))

net.load_state_dict(torch.load('NetWeight_pap2/GoogleNet.pth'))
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    net = net.cuda()
#--------------------------------------分界线-----------------------
vals = [] #记录准确率所用列表
pred = []
labels = []
scores=[]


#对测试数据集进行循环
for data, target in test_loader:
    data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    output = net(data) #将特征数据喂入网络，得到分类的输出
    val = rightness(output, target) #获得正确样本数以及总样本数
 #   print(val[0])
    vals.append(val) #记录结果    
    #为画混淆矩阵添加的
 #   print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    prediction = torch.max(output, 1)[1]
 #   print(prediction)
 #   print(val[1])
    pred1 = prediction.tolist()
    label1 = target.tolist()
    for i in range(len(pred1)):
        pred.append(pred1[i])
        
    for j in range(len(label1)):
        labels.append(label1[j])
    
    '''
    pred1 = val[2].tolist()
    label1 = target.tolist()
    for i in range(len(pred1)):
        pred.append(pred1[i])
        
    for j in range(len(label1)):
        labels.append(label1[j])
        

'''
cm = confusion_matrix(labels, pred)
#labels_name=['Atelectasis','Normal']
#labels_name=['Nodule','Atelectasis','Normal','Infection']
labels_name=['Cardiomegaly','Emphysema','Normal']
plot_confusion_matrix(cm, labels_name, "Confusion Matrix")

rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 100. * rights[0].cpu().numpy() / rights[1]
print("the test right rate is : ",right_rate,"%")

#plot_ROC_AUC(labels,scores)
