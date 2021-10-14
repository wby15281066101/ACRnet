# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 06:59:39 2021
  
IS 越高，生成图像质量越好
因此，IS的取值越高，表明生成模型的效果越理想
@author: wby
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
from torchvision import datasets, models, transforms
import MyNet_resnet
# we should use same mean and std for inception v3 model in training and testing process
# reference web page: https://pytorch.org/hub/pytorch_vision_inception_v3/
mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]

num_classes = 4
def inception_score(imgs, batch_size=64, resize=True, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    batch_size -- batch size for feeding into Inception v3
    resize -- if image size is smaller than 229, then resize it to 229
    splits -- number of splits, if splits are different, the inception score could be changing even using same data
    """
    # Set up dtype
    device = torch.device("cuda:0")  # you can change the index of cuda

    N = len(imgs)
    print('40行:',N)
    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    print('Creating data loader')
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    #+++++++++++++
    inception_model = MyNet_resnet.Resnet(num_classes = num_classes)
    inception_model.load_state_dict(torch.load('NetWeightFace/MyNet_resnet.pth'))
    use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
    if use_gpu:
        inception_model = inception_model.cuda()
    #++++++++++++++++++++++++
    
  #  inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
 #   inception_model.eval()
    up = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False).to(device)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions using pre-trained inception_v3 model
    print('Computing predictions using inception v3 model')
    preds = np.zeros((N, 4))

    for i, batch in enumerate(dataloader, 0):
        batch = batch[0].to(device)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean KL Divergence
    print('Computing KL Divergence')
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :] # split the whole data into several parts
        py = np.mean(part, axis=0)  # marginal probability
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]  # conditional probability
            scores.append(entropy(pyx, py))  # compute divergence
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


#------------------- main function -------------------#
    
image_size = 224
data_dir = 'imgs'
cifar = datasets.ImageFolder(os.path.join(data_dir, 'test'),
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
#cifar = torch.utils.data.DataLoader(test_dataset, batch_size = 10, shuffle = True, num_workers=0)
#print(len(test_dataset))
# example of torch dataset, you can produce your own dataset
'''
cifar = dset.CIFAR10(root='imgs/', download=True,
                     transform=transforms.Compose([transforms.Resize(32),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean_inception, std_inception)
                                                   ])
                     )
   ''' 
mean, std = inception_score(cifar, splits=3)
print('IS is %.4f' % mean)
print('The std is %.4f' % std)