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
import untitled1
import MyNet_resnet
import MyNet_Google_Resnet
import MyNet_RseSe_bloock_net
import MyNet_Google_RseSe_bloock_net

#torch.backends.cudnn.enabled = False
#from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
#image_size = 28  #图像的总尺寸28*28
num_classes = 4#标签的种类数
num_epochs = 2 #训练的总循环周期
batch_size = 32  #一个撮（批次）的大小，64张图片
# 从硬盘文件夹中加载图像数据集


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


#----------------------------混淆矩阵---------------------------------
def plot_confusion_matrix(cm, labels_name, title):
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)    # 在特定的窗口上显示图像，设置颜色
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    for first_index in range(len(cm)):    #第几行
        for second_index in range(len(cm[first_index])):    #第几列
            plt.text(first_index, second_index, cm[first_index][second_index])
    plt.savefig('cm.png', format='png')
    plt.show()

#----------------------------ROC_AUC---------------------------------
#disease_class = ['肿瘤/结节','肺不张','正常','感染'] 
disease_class = ['Nodule','Atelectasis','No Finding','Pneumonia'] 

   
def plot_ROC_AUC(labels,scores):
    binary_label = label_binarize(labels, classes=list(range(4))) # num_classes=10
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(binary_label[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
   
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(binary_label.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    plt.figure(figsize=(8, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (auc = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

  #  plt.plot(fpr["macro"], tpr["macro"],
    #         label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
    #         color='navy', linestyle=':', linewidth=4)

    #  10
  #  for i in range(len(disease_class)):
   #     plt.plot(fpr[i], tpr[i], lw=2,label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        #lw=2
        
          
    plt.plot(fpr[0], tpr[0], lw=2,
                 label='ROC curve of Nodule (auc = {1:0.2f})'.format(0, roc_auc[0]))    
    plt.plot(fpr[1], tpr[1], lw=2,
                 label='ROC curve of Atelectasis (auc = {1:0.2f})'.format(1, roc_auc[1]))    
    plt.plot(fpr[2], tpr[2], lw=2,
                 label='ROC curve of No Finding (auc = {1:0.2f})'.format(2, roc_auc[2]))      
    plt.plot(fpr[3], tpr[3], lw=2,
                 label='ROC curve of Pneumonia (auc = {1:0.2f})'.format(3, roc_auc[3]))
  #  plt.plot(fpr[4], tpr[4], lw=2,
   #              label='ROC curve of cat (auc = {1:0.2f})'.format(4, roc_auc[4]))  
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.savefig('Multi-class ROC.jpg', bbox_inches='tight')
    plt.show()
        
#-----------------------------------这里放model------------------------
    
def rightness(predictions, labels):
#    计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素
'''
#-------cliquenet----------------
import argparse
parser = argparse.ArgumentParser(description='CliqueNet ImageNet Training')
parser.set_defaults(attention=True)
best_prec1 = 0
global args
args = parser.parse_args()
net = cliquenet.build_cliquenet(input_channels=2, list_channels=[2, 4, 8, 8], 
                                list_layer_num=[6, 6, 6, 6], if_att=args.attention)
#-------cliquenet--------
'''
#net = SENet.Senet()
#net = SKNet.SKNet50()
#net = ResNeXt.Resnext(2)
#net = GoogleNet.GoogLeNet()
#net = AlexNetmodel.AlexNet(num_classes=5)
#net = ResNetmodel.ResNet50()
#net = tcmodel.Resnet()

#net = VGG16model.VGG16()
#net = MyNet.MyNet(num_classes = num_classes)
#net = MyNet_VGG.VGG18()

#net = MyNet_Google.My_GoogLeNet(num_classes = num_classes)
net = MyNet_resnet.Resnet(num_classes = num_classes)

#net = MyNet_Google_RseSe_bloock_net.My_Google_Resnet(num_classes = num_classes)
#net = MyNet_RseSe_bloock_net.Resnet(num_classes = num_classes)
#net = MyNet_Google_Resnet.My_Google_Resnet(num_classes = num_classes)

#net.load_state_dict(torch.load('NetWeight_zz/MyNet.pth'))
#net.load_state_dict(torch.load('NetWeight/MyResSeNet_37.pth'))
#net.load_state_dict(torch.load('NetWeight_zz/MyNet_Google.pth'))
net.load_state_dict(torch.load('NetWeight_zz/MyNet_resnet.pth'))
#net.load_state_dict(torch.load('NetWeight_zz/MyNet.pth'))


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
    vals.append(val) #记录结果
 #   pred1 = val[2].tolist()
    label1 = target.tolist()
 #   for i in range(len(pred1)):
  #      pred.append(pred1[i])
    for j in range(len(label1)):
        labels.append(label1[j])
    scores1 = torch.softmax(output, dim=1).cpu().detach().numpy() # out = model(data)
    for k in scores1:
        scores.append(k)
scores=np.array(scores)


rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 100. * rights[0].cpu().numpy() / rights[1]
print("the test right rate is : ",right_rate,"%")

plot_ROC_AUC(labels,scores)
