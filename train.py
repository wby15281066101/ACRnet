# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 09:02:07 2021

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
#import VGGmodel
import GoogleNet
#import tcmodel
import ResNetmodel
import SENet
#import SKNet
#import cliquenet
#import ResNeXt
import MyNet
import MyNet_VGG
import MyNet_Google
import untitled1
import MyNet_resnet
import MyNet_Google_Resnet
import MyNet_RseSe_bloock_net
import MyNet_Google_RseSe_bloock_net

import MyNet_Resnet2
import MyNet_Google_Resnet2
import MyNet_Google_Resnet3
import MyNet_RseSe_Plus_bloock_net

import MyResNet
#torch.backends.cudnn.enabled = False
#from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
#image_size = 28  #图像的总尺寸28*28
num_classes = 2#标签的种类数
num_epochs = 5 #训练的总循环周期
batch_size = 32  #一个撮（批次）的大小，64张图片

data_dir = 'data'

image_size = 224

# 从data_dir/train加载文件
# 加载的过程将会对图像自动作如下的图像增强操作：
# 1. 随机从原始图像中切下来一块224*224大小的区域
# 2. 随机水平翻转图像
# 3. 将图像的色彩数值标准化
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                    transforms.Compose([
  #                                      transforms.RandomResizedCrop(image_size),
  #                                      transforms.RandomHorizontalFlip(),
  #两行换成惠文改的代码            
                                     #   transforms.Grayscale(num_output_channels=1),  
                                        transforms.Resize(256),
                                        transforms.CenterCrop(image_size),
                                      #  transforms.Resize(((224,224))),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                     #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                  
                                    ])
                                    )
#np.random.seed(200)
#np.random.shuffle(train_dataset) 
# 加载校验数据集，对每个加载的数据进行如下处理：
# 1. 放大到256*256像素
# 2. 从中心区域切割下224*224大小的图像区域
# 3. 将图像的色彩数值标准化
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                    transforms.Compose([
                                   #     transforms.Grayscale(num_output_channels=1),    
                                        transforms.Resize(256),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
                                    )
#np.random.seed(200)
#np.random.shuffle(val_dataset)

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

# 创建相应的数据加载器  True   False   shuffle = True


#np.random.shuffle(train_dataset) 
#print(len(val_dataset))
#np.random.seed(200)
#np.random.shuffle(train_dataset) 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 10, shuffle = True, num_workers=0)  #设成20 LOSS 曲线收敛
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 10, shuffle = True, num_workers=0)  
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 10, shuffle = True, num_workers=0)
# 读取得出数据中的分类类别数
num_classes = len(train_dataset.classes)
#print(val_dataset)
#print(val_loader)
# 检测本机器是否安装GPU，将检测结果记录在布尔变量use_cuda中
use_cuda = torch.cuda.is_available()
# 当可用GPU的时候，将新建立的张量自动加载到GPU中
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor


def imshow(inp, title=None):
    # 将一张图打印显示出来，inp为一个张量，title为显示在图像上的文字
    
    #一般的张量格式为：channels*image_width*image_height
    #而一般的图像为image_width*image_height*channels所以，需要将channels转换到最后一个维度
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
images, labels = next(iter(train_loader))
# 将这个batch中的图像制成表格绘制出来
out = torchvision.utils.make_grid(images)
imshow(out, title=[train_dataset.classes[x] for x in labels])


#-------------------------------模型-----------------------------------
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
      #  init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()
        
#-----------------------------------这里放model------------------------
    
def rightness(predictions, labels):
#    计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
   # print(pred)
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels),pred #返回正确的数量和这一次一共比较了多少元素

#net = SENet.Senet()
#net = SKNet.SKNet50()
#net = ResNeXt.Resnext(2)
#net = GoogleNet.GoogLeNet()
#net = AlexNetmodel.AlexNet(num_classes=2) 
#net = MyNet.MyNet(num_classes = num_classes)
#net = MyNet_VGG.VGG18()
#net = MyNet_Resnet2.Resnet(num_classes = num_classes) 
  
    
#net = MyNet_Google.My_GoogLeNet(num_classes = num_classes)
net = MyNet_resnet.Resnet(num_classes = num_classes)   
#net = MyNet_Google_RseSe_bloock_net.My_Google_Resnet(num_classes = num_classes)
#net = MyNet_RseSe_bloock_net.Resnet(num_classes = num_classes)
#net = MyNet_Google_Resnet.My_Google_Resnet(num_classes = num_classes)
#net = MyNet_RseSe_Plus_bloock_net.Resnet(num_classes = num_classes)


#net = MyNet_Google_Resnet2.My_Google_Resnet(num_classes = num_classes)
#net = MyNet_Google_Resnet3.My_Google_Resnet(num_classes = num_classes)
#net = untitled1.My_GoogLeNet(num_classes = num_classes)
#net = ResNetmodel.ResNet50()
#net = tcmodel.Resnet()
#net = VGGmodel.VGG16()
    
#net = MyResNet.resnet50()
#新建一个卷积神经网络的实例，此时ConvNet的__init__函数就会被自动调用
net.apply(weigth_init)
net = net.cuda() if use_cuda else net
#print(net)
criterion = nn.CrossEntropyLoss() #Loss函数的定义，交叉熵

#optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9) #定义优化器，普通的随机梯度下降算法
optimizer = optim.Adam(net.parameters(), lr=0.0001)  #原本是这个
#optimizer = torch.optim.Adadelta(net.parameters(), lr=0.0005)
record = [] #记录准确率等数值的容器  去掉
weights = [] #每若干步就记录一次卷积核
a = []  #装  训练准确率
b = []    #  校验准去率
c = []  #损失函数
d = []
print("OK11111111111111")
#开始训练循环

for epoch in range(num_epochs):
    
    train_rights = [] #记录训练数据集准确率的容器    
    train_losses = []       
    for batch_idx, (data, target) in enumerate(train_loader):  #针对容器中的每一个批进行循环
        data, target = data.clone().requires_grad_(True), target.clone().detach()  #data为一批图像，target为一批标签
        net.train() # 给网络模型做标记，标志说模型正在训练集上训练，
                    #这种区分主要是为了打开关闭net的training标志，从而决定是否运行dropout
        if use_cuda:
            data, target = data.cuda(), target.cuda()    
        output = net(data) #神经网络完成一次前馈的计算过程，得到预测输出output
        loss = criterion(output, target) #将output与标签target比较，计算误差
        optimizer.zero_grad() #清空梯度
        loss.backward() #反向传播
        optimizer.step() #一步随机梯度下降算法
        right = rightness(output, target) #计算准确率所需数值，返回数值为（正确样例数，总样本数）
        train_rights.append(right) #将计算结果装到列表容器train_rights中

        #因为所有计算都在GPU中，打印的数据再加载到CPU中
        loss = loss.cpu() if use_cuda else loss
        train_losses.append(loss.data.numpy())
   
        if batch_idx % 200 == 0: #每间隔100个batch执行一次打印等操作  
            net.eval() # 给网络模型做标记，标志说模型在训练集上训练
            val_rights = [] #记录校验数据集准确率的容器
            
          
            for (data, target) in val_loader:
                data, target = data.clone().requires_grad_(True), target.clone().detach()
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                
                output = net(data) #完成一次前馈计算过程，得到目前训练得到的模型net在校验数据集上的表现
             #+++
                loss1 = criterion(output, target)   
                right = rightness(output, target) #计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                val_rights.append(right)
                #---------AUC-------------------------1句---
               # scores = torch.softmax(output, dim=1).cpu().detach().numpy() # out = model(data)
            # 分别计算在目前已经计算过的测试数据集，以及全部校验集上模型的表现：分类准确率
            #train_r为一个二元组，分别记录目前已经经历过的所有训练集中分类正确的数量和该集合中总的样本数，
            #train_r[0]/train_r[1]就是训练集的分类准确度，同样，val_r[0]/val_r[1]就是校验集上的分类准确度
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            #val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
            #打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
           # print(val_r)
            print('训练周期: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t训练正确率: {:.2f}%\t校验正确率: {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.data, 
                100. * train_r[0].cpu().numpy() / train_r[1], 
                100. * val_r[0].cpu().numpy() / val_r[1]))
            
            #将准确率和权重等数值加载到容器中，以方便后续处理
            record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))#不要
            a.append(train_r[0].cpu().numpy() / train_r[1]) #   a.append(100. * train_r[0].cpu().numpy() / train_r[1])
            b.append(val_r[0].cpu().numpy() / val_r[1]) #    b.append(100. * val_r[0].cpu().numpy() / val_r[1])
            c.append(loss.data)
            d.append(loss1.data)
            B = 100. * val_r[0].cpu().numpy() / val_r[1]


torch.save(net.state_dict(), 'NetWeight/MyNet_resnet.pth')
#torch.save(net.state_dict(), 'NetWeight_zz/MyNet_3_7.pth')
plt.figure(figsize = (10, 7))
plt.plot(a, 'r',label = 'Training Accuracy')
plt.plot(b,'b',label = 'Validation Accuracy')
plt.plot(c,'g',label = 'Training Loss')
#plt.title('Training accuracy and loss rate')
plt.title('Training and Validation Result')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('tvl.png', format='png')
plt.show()  

'''
plt.figure(figsize = (10, 7))
plt.plot(b,'b',label = 'Validation  Accuracy')
plt.plot(d,'g',label = 'Validation Loss')
plt.title('Validation Result')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('vacc.png', format='png')
plt.show()  
'''

'''
#---------------------------------------测试--------------------------
#在测试集上分批运行，并计算总的正确率
net.eval() #标志模型当前为运行阶段
vals = [] #记录准确率所用列表

#对测试数据集进行循环
for data, target in test_loader:
    data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    output = net(data) #将特征数据喂入网络，得到分类的输出
    val = rightness(output, target) #获得正确样本数以及总样本数
    vals.append(val) #记录结果

#计算准确率
rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 100. * rights[0].cpu().numpy() / rights[1]
print("the test right rate is : ",right_rate,"%")
'''