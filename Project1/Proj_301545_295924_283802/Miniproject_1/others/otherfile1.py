import torch 
import torch.nn.functional as F
import torch.nn as nn

#model:
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1) # 32 x 32 
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 24, kernel_size = 3, padding = 1) 
        self.conv3 = nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size = 3, padding = 1) 
        self.conv4 = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 1) 
        self.conv5 = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 1) 
        
        self.deconv1 = nn.Conv2d(in_channels = 96, out_channels = 48, kernel_size = 3, padding = 1) 
        self.deconv2 = nn.Conv2d(in_channels = 144, out_channels = 48, kernel_size = 3, padding = 1) 
        self.deconv3 = nn.Conv2d(in_channels = 120, out_channels = 24, kernel_size = 3, padding = 1) 
        self.deconv4 = nn.Conv2d(in_channels = 64, out_channels = 16, kernel_size = 3, padding = 1) 
        self.deconv5 = nn.Conv2d(in_channels = 35, out_channels = 3, kernel_size = 3, padding = 1) 
        
        self.l_relu = nn.LeakyReLU(negative_slope = 0.1)
        self.upsample = nn.Upsample(scale_factor = (2, 2))
        #self.dropout = nn.Dropout(0.5)
    

    def forward(self, x):
        # encode
        x1 = self.l_relu(self.conv1(x))
        x2 = self.l_relu(self.pool(x1))

        x3 = self.l_relu(self.conv2(x2))
        x4 = self.l_relu(self.pool(x3))

        x5 = self.l_relu(self.conv3(x4))
        x6 = self.l_relu(self.pool(x5))

        x7 = self.l_relu(self.conv4(x6))
        x8 = self.l_relu(self.pool(x7))

        x9 = self.l_relu(self.conv5(x8))
        #print(x5.shape)

        # decode
        y1 = torch.cat((x8, x9), dim = 1)
        y2 = self.l_relu(self.upsample(y1))
        y3 = self.l_relu(self.deconv1(y2))
        #print(y1.shape)

        y4 = torch.cat((y3, x7), dim = 1)
        y5 = torch.cat((x6, y4), dim = 1)
        y6 = self.l_relu(self.upsample(y5))
        y7 = self.l_relu(self.deconv2(y6))
        #print(y2.shape)

        y8 = torch.cat((y7, x5), dim = 1)
        y9 = torch.cat((x4, y8), dim = 1)
        y10 = self.l_relu(self.upsample(y9))
        y11 = self.l_relu(self.deconv3(y10))
        #print(y3.shape)

        y12 = torch.cat((y11, x3), dim = 1)
        y13 = torch.cat((x2, y12), dim = 1)
        y14 = self.l_relu(self.upsample(y13))
        y15 = self.l_relu(self.deconv4(y14))
        #print(y4.shape)

        y16 = torch.cat((y15, x1), dim = 1)
        y17 = torch.cat((x, y16), dim = 1)
        #print(y5.shape)
        y18 = self.l_relu(self.deconv5(y17) + x)
        
        return y18

#Dataset
class Dataset_aug(torch.utils.data.Dataset): 
  'Characterizes a dataset for PyTorch'
  def __init__(self, train_input, train_target, transform = None, switch_pixels = None):
        'Initialization'
        x, y = train_input, train_target
        if transform != None :
            print("With data augmentation : transform.")
        if switch_pixels != None :
            print("With data augmentation : switch pixels with n_max = ", switch_pixels[0], " and p = ", switch_pixels[1])
        self.x = x.float()
        self.y = y.float()
        self.transform = transform
        self.switch_pixels = switch_pixels

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        # get label
        X_trans = self.x[index]
        Y_trans= self.y[index]

        seed = torch.randint(2147483647,(1,1)) # make a seed with generator 
        torch.manual_seed(seed.item()) # set the random seed for transforms
        if self.transform is not None:
            X_trans = self.transform(X_trans)

        torch.manual_seed(seed.item()) # set the random seed for transforms
        if self.transform is not None:
            Y_trans = self.transform(Y_trans)  
        
        torch.manual_seed(seed.item())
        if self.switch_pixels is not None:
            n_max, p = self.switch_pixels
            # n_max : maximum number of pixels that might switch. 
            # p : prob that the pixels are switched.
            if torch.rand((1,1)) < p:
                if n_max<50 :
                    n_max = 51
                # n : number of switched pixels, random in between 50 and n_max (not included)
                n = torch.randint(low = 50, high = n_max, size = (1,1))
                # index : random index of the n pixels that will be switched.
                index = torch.randint(low=0, high = X_trans.shape[1], size = (n,2))
                i,j = index[:, 0], index[:, 1]
                v_x = Y_trans[:, i, j]
                v_y = X_trans[:, i, j]

                X_trans[:, i, j] = v_x
                Y_trans[:, i, j] = v_y

        return X_trans, Y_trans

class Dataset(torch.utils.data.Dataset): #A TESTER
  'Characterizes a dataset for PyTorch'
  def __init__(self, train_input, train_target):
        'Initialization'
        x, y = train_input, train_target
        self.x = x.float()
        self.y = y.float()

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        # get label
        X_trans = self.x[index]
        Y_trans= self.y[index]

        return X_trans, Y_trans

