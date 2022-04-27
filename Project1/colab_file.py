# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:43:57 2022

@author: louis
"""
"file for colab"
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



import torch
from datetime import datetime
import time


"""
noisy_imgs_1, noisy_imgs_2 = torch.load("train_data.pkl")
print("Training data : \n noisiy_imgs_1 : ", noisy_imgs_1.shape, "\n noisy_imgs_2 : ", noisy_imgs_2.shape)
noisy_imgs, clean_imgs = torch.load("val_data.pkl")
print("Test data : \n noisiy_imgs : ", noisy_imgs.shape, "\n clean_imgs : ", clean_imgs.shape)
noisy_imgs_1_reduced, noisy_imgs_2_reduced = noisy_imgs_1[:SIZE], noisy_imgs_2[:SIZE]
print("Training data reduced : \n noisiy_imgs_1_reduced : ", noisy_imgs_1_reduced.shape, "\n noisy_imgs_2_reduced : ", noisy_imgs_2_reduced.shape)
all_noisy_imgs = torch.cat((noisy_imgs_1_reduced, noisy_imgs_2_reduced), dim = 0)
print("Concatenated training data (reduced) : \n all_noisy_imgs : ", all_noisy_imgs.shape)
"""

SIZE = 50000
BATCH_SIZE = 100
train_set = Dataset(SIZE)

"""N = 20
plt.figure()
for i in range(N):
    plt.subplot(2, N, 2*i+1)
    plt.imshow(train_set.x[i].permute(1, 2, 0).int())
    plt.subplot(2, N, 2*i+2)
    plt.imshow(train_set.y[i].permute(1, 2, 0).int())
plt.show()"""


# Model Initialization
model = AE()
  
# Validation using MSE Loss function
loss_function = nn.L1Loss()
  
# Using an Adam Optimizer with lr = 0.001
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-3, betas=(0.9, 0.99))

# DataLoader is used to load the dataset 
# for training
loader_1 = torch.utils.data.DataLoader(dataset = train_set,
                                     batch_size = BATCH_SIZE,
                                     shuffle = True)


#OPTIMIZATION
epochs = 30
outputs = []
losses = []
for epoch in range(epochs):
    print("epoch : ", epoch + 1)
    start = time.time()
    for noisy_imgs_1, noisy_imgs_2 in loader_1:
        #print(noisy_imgs_1.shape)
        #print(noisy_imgs_2.shape)

        #noisy_imgs_1 = noisy_imgs_1.reshape(-1, 32 * 32)
        #noisy_imgs_2 = noisy_imgs_2.reshape(-1, 32 * 32)    
        # Output of Autoencoder
        #print("type : ", noisy_imgs_1.dtype)
        reconstructed = model(noisy_imgs_1)
            
        # Calculating the loss function
        loss = loss_function(reconstructed, noisy_imgs_2)
            
        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Storing the losses in a list for plotting
        losses.append(loss.detach().numpy())
    print('The epoch took {}s to complete\n'.format(time.time() - start))
    outputs.append((epochs, noisy_imgs_2, reconstructed))
  
# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
  
# Plotting the last 100 values
plt.plot(losses[-100:])
plt.show()


#time = datetime.now().strftime('%m_%d_%Hh_%Mm_%Ss')

#PATH = "./test1/project1_1_" + time + ".pth" # so that we don't overwrite files
#torch.save(model.state_dict(), PATH)

print('Finished training')
print('\n\n\n')
print('------------------------------------------------')
print("validation")
print('\n')


#model = AE()
#time = '04_27_13h_14m_41s' # to be filled according to the job we want to load
#PATH = "./test1/project1_1_" + time + ".pth"
#model.load_state_dict(torch.load(PATH))

SIZE = 1000
BATCH_SIZE = 1
test_set = Dataset(SIZE, train = False)

"""plt.subplot(2, 1, 1)
plt.imshow(test_set.x[-1].permute(1, 2, 0).int())
plt.subplot(2, 1, 2)
plt.imshow(test_set.y[-1].permute(1, 2, 0).int())
plt.show()"""

loader_2 = torch.utils.data.DataLoader(dataset = test_set,
                                     batch_size = BATCH_SIZE,
                                     shuffle = False)

PSNR = torch.empty(size = (1, SIZE))
i = 0
for noisy_imgs, ground_truth in loader_2:
    denoised = model(noisy_imgs)
    Psnr = psnr(denoised / 255, ground_truth / 255)
    PSNR[0, i] = Psnr
    if Psnr > 32:
        plot_3imgs(denoised, ground_truth, noisy_imgs, add_title = 'good' + str(i))
    if Psnr < 20:
        plot_3imgs(denoised, ground_truth, noisy_imgs, add_title = 'bad' + str(i))
    i += 1



print("PSNR mean : ", torch.mean(PSNR).item(), " dB")
plt.style.use('fivethirtyeight')
plt.ylabel('PSNR')
plt.plot(PSNR[0,:].detach().numpy())
plt.show()



