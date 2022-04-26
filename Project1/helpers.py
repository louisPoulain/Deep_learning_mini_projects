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
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 4, kernel_size = 3, padding = 'same')
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3, padding = 'same')
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 'same')
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 'same')
        self.conv5 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 'same')
        
        self.deconv1 = nn.ConvTranspose1d(in_channels = 64, out_channels = 32, kernel_size = (7, 7))
        self.deconv2 = nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = (7, 7))
        self.deconv3 = nn.ConvTranspose2d(in_channels = 16, out_channels = 8, kernel_size = (7, 7))
        self.deconv4 = nn.ConvTranspose2d(in_channels = 8, out_channels = 4, kernel_size = (7, 7))
        self.deconv5 = nn.ConvTranspose2d(in_channels = 4, out_channels = 3, kernel_size = (7, 7))
        
        self.l_relu = nn.LeakyReLU(negative_slope = 0.3)
        
        
  
    def forward(self, x):
        x_encoded1 = self.l_relu(self.conv1(x))
        x_encoded2 = self.pool(self.l_relu(self.conv2(x_encoded1)))
        x_encoded3 = self.pool(self.l_relu(self.conv3(x_encoded2)))
        x_encoded4 = self.pool(self.l_relu(self.conv4(x_encoded3)))
        x_encoded5 = self.pool(self.l_relu(self.conv5(x_encoded4)))
        
        x_decoded1 = self.l_relu(self.deconv1(x_encoded5))
        x_decoded2 = self.l_relu(self.deconv2(x_decoded1))
        x_decoded3 = self.l_relu(self.deconv3(x_decoded2))
        x_decoded4 = self.l_relu(self.deconv4(x_decoded3))
        x_decoded5 = self.l_relu(self.deconv5(x_decoded4))
        return x_decoded5

def plot_3imgs(denoised, ground_truth, noisy_imgs): #values of the images are in between [0, 255].
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
    plt.savefig('./validation_test/first.png', dpi = 300)
    plt.show()
    
