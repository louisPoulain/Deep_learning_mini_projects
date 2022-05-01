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
            x, y = torch.load("drive/MyDrive/Colab_Notebooks/DeepL_miniProj/train_data.pkl")
            print("Training data : \n noisy_imgs_1 : ", x.shape, "\n noisy_imgs_2 : ", y.shape)
            if SIZE > 50000:
                print("You enter a size too big, using size = 50000")
                SIZE = 50000
        else : 
            x, y = torch.load("drive/MyDrive/Colab_Notebooks/DeepL_miniProj/val_data.pkl")
            print("Test data : \n noisy_imgs : ", x.shape, "\n clean_imgs : ", y.shape)
            if SIZE > 50000:
                print("You enter a size too big, using size = 1000")
                SIZE = 1000
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
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 'same', padding_mode = 'reflect') # out 32 x 32
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 'same', padding_mode = 'reflect', groups = 1) # out 16 x 16 (max pool)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 'same', padding_mode = 'reflect', groups = 1) # out 8 x 8
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 'same', padding_mode = 'reflect', groups = 1) # out 4 x 4
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 'same', padding_mode = 'reflect', groups = 1) # out 2 x 2
        self.conv6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 'same', padding_mode = 'reflect', groups = 1) # out 1 x 1
        self.conv7 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 'same', padding_mode = 'reflect', groups = 1) # out 1 x 1
        self.conv8 = nn.Conv2d(in_channels = 192, out_channels = 192, kernel_size = 3, padding = 'same', padding_mode = 'reflect', groups = 1) # out 1 x 1
        self.conv9 = nn.Conv2d(in_channels = 320, out_channels = 192, kernel_size = 3, padding = 'same', padding_mode = 'reflect', groups = 1) # out 1 x 1
        self.conv10 = nn.Conv2d(in_channels = 192, out_channels = 256, kernel_size = 3, padding = 'same', padding_mode = 'reflect', groups = 1) # out 1 x 1
        self.conv11 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 'same', padding_mode = 'reflect', groups = 1) # out 1 x 1
        
        
        self.deconv = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 2, padding_mode = 'reflect', groups = 1) # 2 x 2 to 1 x 1
        self.deconv0 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 2, padding_mode = 'reflect', groups = 1) # 2 x 2 to 1 x 1
        self.deconv1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 2) # 1x1 to 2 x 2
        self.deconv2 = nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 2) # 2x2 to 3 x 3
        self.deconv3 = nn.ConvTranspose2d(in_channels = 16, out_channels = 8, kernel_size = 2) # 6 x 6 to 7 x 7
        self.deconv4 = nn.ConvTranspose2d(in_channels = 8, out_channels = 4, kernel_size = 2) # 14 x 14 to 15 x 15
        self.deconv5 = nn.ConvTranspose2d(in_channels = 4, out_channels = 3, kernel_size = 2) # 30 x 30 to 31 x 31
        self.deconv6 = nn.ConvTranspose2d(in_channels = 6, out_channels = 3, kernel_size = 2) # 31 x 31 to 32 x 32
        self.deconv7 = nn.Conv2d(in_channels = 14, out_channels = 3, kernel_size = 3, padding = 'same', padding_mode = 'reflect', groups = 1) # 32 x 32
        self.deconv8 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 2, stride = 2, padding_mode = 'reflect', groups = 1) # 64 x 64 to 32 x 32
        self.deconv9 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3)
        
        self.l_relu = nn.LeakyReLU(negative_slope = 0.1)
        self.upsample = nn.Upsample(scale_factor = (2, 2))
        self.linear = nn.Linear(32, 32)
        #self.dropout = nn.Dropout(0.5)
        
        
  
    def forward(self, x):
        # encode
        
        x1 = self.l_relu(self.conv1(x)) # 8 x 32 x 32
        #print(x1.size())
        x2 = self.pool(self.l_relu(self.conv2(x1))) # 16 x 16 x 16
        #print(x2.size())
        x3 = self.pool(self.l_relu(self.conv3(x2))) # 32 x 8 x 8
        #print(x3.size())
        x4 = self.pool(self.l_relu(self.conv4(x3))) # 64 x 4 x 4
        #print(x4.size())
        x5 = self.pool(self.l_relu(self.conv5(x4))) # 64 x 2 x 2
        #print(x5.size())
        x6 = self.pool(self.l_relu(self.conv6(x5))) # 64 x 1 x 1
        x6 = self.upsample(x6) # 64 x 2 x 2
        x6 = self.pool(self.l_relu(self.conv6(x6))) # 64 x 1 x 1
        x6 = self.upsample(x6) # 64 x 2 x 2
        x6 = self.pool(self.l_relu(self.conv6(x6))) # 64 x 1 x 1
        #print(x6.size())
        x7 = self.upsample(x6) # 64 x 2 x 2
        x8 = self.pool(self.l_relu(self.conv7(x7))) # 128 x 1 x 1
        x9 = torch.cat((x6, x8), dim = 1) # 192 x 1 x 1
        x10 = self.upsample(x9) # 192 x 2 x 2
        x11 = self.pool(self.l_relu(self.conv8(x10))) # 192 x 1 x 1
        x12 = torch.cat((x8, x11), dim = 1) # 320 x 1 x 1
        x13 = self.upsample(x12) # 320 x 2 x 2
        x14 = self.pool(self.l_relu(self.conv9(x13))) # 192 x 1 x 1
        x15 = self.upsample(x14) # 192 x 2 x 2
        x16 = self.pool(self.l_relu(self.conv10(x15)))
        x17 = self.upsample(x16)
        x18 = self.l_relu(self.conv11(x17))

        
        # decode
        y = self.l_relu(self.deconv(x18)) # 128 x 1 x 1
        #print(y.size())
        y = self.l_relu(self.upsample(y)) # 128 x 2 x 2
        y0 = self.l_relu(self.deconv0(y)) # 64 x 1 x 1
        #print(y0.size())
        y1 = self.l_relu(self.deconv1(y0)) # 32 x 2 x 2
        #print(y1.size())
        y2 = self.l_relu(self.deconv2(y1)) # 16 x 3 x 3
        #print(y2.size())
        y3 = self.l_relu(self.upsample(y2)) # 16 x 6 x 6
        #print(y3.size())
        #y4 = torch.cat((y3, x4), dim = 1) # 192 channels
        y4 = self.l_relu(self.deconv3(y3)) # 8 x 7 x 7
        #print(y4.size())
        y5 = self.l_relu(self.upsample(y4)) # 8 x 14 x 14
        #print(y5.size())
        y6 = self.l_relu(self.deconv4(y5)) # 4 x 15 x 15
        #print(y6.size())
        y7 = self.l_relu(self.upsample(y6)) # 4 x 30 x 30
        #print(y7.size())
        #y8 = torch.cat((y7, x3), dim = 1) # 192 channels
        y8 = self.l_relu(self.deconv5(y7)) # 3 x 31 x 31
        #print(y8.size())
        y9 = torch.cat((y8, y8), dim = 1) # 6 x 31 x 31
        #print(y9.size())
        y9 = self.l_relu(self.deconv6(y9)) # 3 x 32 x 32
        #print(y9.size(), x1.size())
        y10 = torch.cat((y9, x1), dim = 1) # 11 x 32 x 32
        y11 = torch.cat((y10, x), dim = 1) # 14 x 32 x 32
        y12 = self.l_relu(self.deconv7(y11)) # 3 x 32 x 32
        y13 = self.upsample(y12) # 3 x 64 x 64
        y14 = self.l_relu(self.deconv8(y13)) # 3 x 32 x 32
        y15 = self.linear(y14) # 3 x 32 x 32
        
        #print(y5.size())
        #print(y4.size())
        #y10 = self.l_relu(self.deconv1(y9))
        #y11 = self.upsample(y10)
        #y12 = torch.cat((y11, x2), dim = 1) # 192 channels
        #y13 = self.l_relu(self.deconv2(y12))
        #y14 = self.l_relu(self.deconv1(y13))
        #y15 = self.upsample(y14)
        #y16 = torch.cat((y15, x), dim = 1) # 131 channels
        #y17 = self.l_relu(self.deconv3(y16))
        #y18 = self.l_relu(self.deconv4(y17))
        #y = self.deconv5(y18)
        #y = self.dropout(y)
        return y15

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
    plt.savefig('drive/MyDrive/Colab_Notebooks/DeepL_miniProj/test1/img_end' + add_title + '.png', dpi = 300)
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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
model = AE().to(device)
  
# Validation using MSE Loss function
loss_function = nn.MSELoss().to(device)
  
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
start_ = time.time()
for epoch in range(epochs):
    print("epoch : ", epoch + 1)
    start = time.time()
    Loss = 0
    for noisy_imgs_1, noisy_imgs_2 in loader_1:
        #print(noisy_imgs_1.shape)
        #print(noisy_imgs_2.shape)

        #noisy_imgs_1 = noisy_imgs_1.reshape(-1, 32 * 32)
        #noisy_imgs_2 = noisy_imgs_2.reshape(-1, 32 * 32)    
        # Output of Autoencoder
        #print("type : ", noisy_imgs_1.dtype)
        noisy_imgs_1 = noisy_imgs_1.to(device)
        noisy_imgs_2 = noisy_imgs_2.to(device)
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
        Loss += loss.detach().cpu().numpy()
    losses.append(Loss)
    print('The epoch took {}s to complete\n'.format(time.time() - start))
    outputs.append((epochs, noisy_imgs_2, reconstructed))
  
# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Epochs')
plt.ylabel('Loss')
  
# Plotting the last 100 values
plt.plot(losses)
plt.show()


Time = datetime.now().strftime('%m_%d_%Hh_%Mm_%Ss')

#PATH = "drive/MyDrive/Colab_Notebooks/DeepL_miniProj/test1/project1_1_" + time + ".pth" # so that we don't overwrite files
#torch.save(model.state_dict(), PATH)

print('Finished training after {}.'.format(time.time() - start_))
print('\n\n\n')
print('------------------------------------------------')


#model = AE()
#time = '04_27_13h_14m_41s' # to be filled according to the job we want to load
#PATH = "./test1/project1_1_" + time + ".pth"
#model.load_state_dict(torch.load(PATH))


print("validation")
print('\n')
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
    noisy_imgs = noisy_imgs.to(device)
    ground_truth = ground_truth.to(device)
    denoised = model(noisy_imgs)
    Psnr = psnr(denoised.cpu() / 255, ground_truth.cpu() / 255)
    PSNR[0, i] = Psnr
    #if Psnr > 32:
     #   plot_3imgs(denoised, ground_truth, noisy_imgs, add_title = 'good' + str(i))
    #if Psnr < 20:
     #   plot_3imgs(denoised, ground_truth, noisy_imgs, add_title = 'bad' + str(i))
    i += 1

plot_3imgs(denoised.cpu(), ground_truth.cpu(), noisy_imgs.cpu(), add_title = Time)

print("PSNR mean : ", torch.mean(PSNR).item(), " dB")
plt.style.use('fivethirtyeight')
plt.ylabel('PSNR')
plt.plot(PSNR[0,:].detach().numpy())
plt.savefig('drive/MyDrive/Colab_Notebooks/DeepL_miniProj/test1/valid_' + Time + '.png', dpi = 300)
plt.show()