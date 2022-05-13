# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:34:17 2022

@author: louis
"""

import torch 
# import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio: denoised and ground ̇truth have range [0, 1]
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10**-8)

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, SIZE, train = True):
        'Initialization'
        if train: 
            x, y = torch.load("train_data.pkl")
            print("Training data : \n noisy_imgs_1 : ", x.shape, "\n noisy_imgs_2 : ", y.shape)
        else : 
            x, y = torch.load("val_data.pkl")
            print("Test data : \n noisy_imgs : ", x.shape, "\n clean_imgs : ", y.shape)
        x, y = x[:SIZE], y[:SIZE]
        print("Data reduced : \n noisy_imgs_1_reduced : ", x.shape, "\n noisy_imgs_2_reduced : ", y.shape)
        print("Type : ", x.dtype)
        self.x = x.float()
        self.y = y.float()

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        # get label
        X = self.x[index]
        Y = self.y[index]
        return X, Y
    
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 'same', padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 'same', padding_mode='reflect')
        
        self.deconv1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 'same', padding_mode='reflect')
        self.deconv2 = nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, padding = 'same', padding_mode='reflect')
        self.deconv3 = nn.Conv2d(in_channels = 131, out_channels = 64, kernel_size = 3, padding = 'same', padding_mode='reflect')
        self.deconv4 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 'same', padding_mode='reflect')
        self.deconv5 = nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = 3, padding = 'same', padding_mode='reflect')
        
        self.l_relu = nn.LeakyReLU(negative_slope = 0.1)
        self.upsample = nn.Upsample(scale_factor = (2, 2))
        #self.dropout = nn.Dropout(0.5)
        
        
  
    def forward(self, x):
        # encode
        
        x1 = self.l_relu(self.conv1(x))
        x2 = self.pool(self.l_relu(self.conv2(x1)))
        #print(x2.size())
        x3 = self.pool(self.l_relu(self.conv2(x2)))
        #print(x3.size())
        x4 = self.pool(self.l_relu(self.conv2(x3)))
        #print(x4.size())
        x5 = self.pool(self.l_relu(self.conv2(x4)))
        #print(x5.size())
        x6 = self.l_relu(self.conv2(x5)) #self.pool
        #print(x6.size())
        x7 = self.pool(self.l_relu(self.conv2(x6)))
        x8 = self.upsample(x7)
        x9 = torch.cat((x8, x5), dim = 1)
        #x9 = self.dropout(x9)
        
        # decode
        y1 = self.l_relu(self.deconv1(x9))
        y2 = self.l_relu(self.deconv1(y1))
        y3 = self.l_relu(self.upsample(y2))
        y4 = torch.cat((y3, x4), dim = 1) # 192 channels
        y5 = self.l_relu(self.deconv2(y4))
        y6 = self.l_relu(self.deconv1(y5))
        y7 = self.upsample(y6)
        y8 = torch.cat((y7, x3), dim = 1) # 192 channels
        y9 = self.l_relu(self.deconv2(y8)) # 128 channels
        #print(y4.size())
        y10 = self.l_relu(self.deconv1(y9))
        y11 = self.upsample(y10)
        y12 = torch.cat((y11, x2), dim = 1) # 192 channels
        y13 = self.l_relu(self.deconv2(y12))
        y14 = self.l_relu(self.deconv1(y13))
        y15 = self.upsample(y14)
        y16 = torch.cat((y15, x), dim = 1) # 131 channels
        y17 = self.l_relu(self.deconv3(y16))
        y18 = self.l_relu(self.deconv4(y17))
        y = self.deconv5(y18)
        #y = self.dropout(y)
        return y

def plot_3imgs(denoised, ground_truth, noisy_imgs, add_title = ''): #values of the images are in between [0, 255].
    plt.subplot(1, 3, 1)
    print(noisy_imgs.shape)
    plt.imshow(torch.squeeze(noisy_imgs).permute(1, 2, 0).int()) #int since the data has been changed to float for the NN.
    plt.title("Noisy imgs")
    plt.subplot(1, 3, 2)
    plt.imshow(torch.squeeze(ground_truth).permute(1, 2, 0).int())
    plt.title("Groundtruth")
    plt.subplot(1,3,3)
    plt.imshow(torch.squeeze(denoised).permute(1, 2, 0).int())
    plt.title("Denoised")
    plt.savefig('./validation_test/' + add_title + '.png', dpi = 300)
    plt.show()
    
