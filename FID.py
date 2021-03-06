# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 07:47:27 2021


FID：  FID越小，则图像多样性越好，质量也越好
@author: wby
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import numpy as np
from scipy.stats import entropy
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
#import dataloader_own
from scipy import linalg

from torchvision.models.inception import inception_v3

import os
from torchvision import datasets, models, transforms
import MyNet_resnet
# we should use same mean and std for inception v3 model in training and testing process
# reference web page: https://pytorch.org/hub/pytorch_vision_inception_v3/
mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]
num_classes = 4
def compute_FID(img1, img2, batch_size=1):
    device = torch.device("cuda:0")  # you can change the index of cuda

    N1 = len(img1)
    N2 = len(img2)
    n_act = 4  # the number of final layer's dimension   #2048

    # Set up dataloaders
    dataloader1 = torch.utils.data.DataLoader(img1, batch_size=batch_size)
    dataloader2 = torch.utils.data.DataLoader(img2, batch_size=batch_size)

    # Load inception model
    # inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    #++++++++++++++++++++++
    
    inception_model = MyNet_resnet.Resnet(num_classes = num_classes)
    inception_model.load_state_dict(torch.load('NetWeightFace/MyNet_resnet.pth'))
    use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
    if use_gpu:
        inception_model = inception_model.cuda()
    
    #++++++++++++++++++
    '''
    block_idx = inception_v3.BLOCK_INDEX_BY_DIM[n_act]
    inception_model = inception_v3([block_idx]).to(device)
    inception_model.eval()
    '''
    # get the activations
    def get_activations(x):
        x = inception_model(x)[0]
        return x.cpu().data.numpy().reshape(batch_size, -1)

    act1 = np.zeros((N1, n_act))
    act2 = np.zeros((N2, n_act))

    data = [dataloader1, dataloader2]
    act = [act1, act2]
    for n, loader in enumerate(data):
        for i, batch in enumerate(loader, 0):
            batch = batch[0].to(device)
            batch_size_i = batch.size()[0]
            activation = get_activations(batch)

            act[n][i * batch_size:i * batch_size + batch_size_i] = activation

    # compute the activation's statistics: mean and std
    def compute_act_mean_std(act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    mu_act1, sigma_act1 = compute_act_mean_std(act1)
    mu_act2, sigma_act2 = compute_act_mean_std(act2)

    # compute FID
    def _compute_FID(mu1, mu2, sigma1, sigma2,eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        FID = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        return FID

    FID = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)
    return FID


# main function to compuate FID
'''
data_root = os.path.join('/PATH/TO/YOUR/IMAGE1')
my_dataset_fakeB = dataloader_own.image_loader(data_root, batch_size=64, img_size=299, resize=True, rotation=False, normalize=[mean_inception, std_inception])
data_root = os.path.join('/PATH/TO/YOUR/IMAGE2')
my_dataset_realB = dataloader_own.image_loader(data_root, batch_size=64, img_size=299, resize=True, rotation=False, normalize=[mean_inception, std_inception])
'''
data_dir1 = 'imgs'  #fake imges
data_dir = 'dataface' #real imges
#data_dir = '7_n'
image_size = 224

my_dataset_fakeB = datasets.ImageFolder(os.path.join(data_dir, 'test2'),
                                    transforms.Compose([
                                  #       transforms.Grayscale(num_output_channels=1),    
                                        transforms.Resize(224),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
                                    )
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 10, shuffle = True, num_workers=0)  #设成20 LOSS 曲线收敛
#my_dataset_fakeB = torch.utils.data.DataLoader(test_dataset, batch_size = 10, shuffle = True, num_workers=0)  
#my_dataset_fakeB = dataloader_own.image_loader(test_dataset, batch_size=64, img_size=224, resize=True, rotation=False, normalize=[mean_inception, std_inception])

my_dataset_realB = datasets.ImageFolder(os.path.join(data_dir1, 'test2'),
                                    transforms.Compose([
                                  #       transforms.Grayscale(num_output_channels=1),    
                                        transforms.Resize(224),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
                                    )
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 10, shuffle = True, num_workers=0)  #设成20 LOSS 曲线收敛
#my_dataset_realB = torch.utils.data.DataLoader(test_dataset1, batch_size = 10, shuffle = True, num_workers=0)  
#my_dataset_realB = dataloader_own.image_loader(test_dataset1, batch_size=64, img_size=224, resize=True, rotation=False, normalize=[mean_inception, std_inception])


FID = compute_FID(my_dataset_fakeB, my_dataset_realB)
print('the FID is :',FID)