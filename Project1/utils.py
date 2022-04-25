import torch 
import torch.nn.functional as F
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

def plot_3imgs(denoised, ground_truth, noisy_imgs): #values of the images are in between [0, 255].
    plt.subplot(1, 3, 1)
    plt.imshow(torch.squeeze(noisy_imgs).permute(1, 2, 0).int()) #int since the data has been changed to float for the NN.
    plt.title("Noisy imgs")
    plt.subplot(1, 3, 2)
    plt.imshow(torch.squeeze(ground_truth).permute(1, 2, 0).int())
    plt.title("Groundtruth")
    plt.subplot(1,3,3)
    plt.imshow(torch.squeeze(denoised).permute(1, 2, 0).int())
    plt.title("Denoised")
    plt.show()


# ------------------------- Noise2Noise ---------------------------------------------------------------------------
class Noise2Noise(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.enc_conv0 = torch.nn.Conv2d(in_channels = 3, out_channels = 48, kernel_size = 3, padding = 'same')
        self.enc_conv1 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2)
        self.enc_conv2 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        self.pool2 = torch.nn.MaxPool2d(kernel_size = 2)
        self.enc_conv3 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        self.pool3 = torch.nn.MaxPool2d(kernel_size = 2)
        self.enc_conv4 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        self.pool4 = torch.nn.MaxPool2d(kernel_size = 2)
        self.enc_conv5 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        self.pool5 = torch.nn.MaxPool2d(kernel_size = 2)
        self.enc_conv6 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        
        self.upsample5 = torch.nn.Upsample(size=(2,2))
        #check if it is indeed conv transpose
        self.dec_conv5A = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, padding = 'same')
        self.dec_conv5B = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, padding = 'same')
        self.upsample4 = torch.nn.Upsample(size=4)
        self.dec_conv4A = torch.nn.Conv2d(in_channels = 144, out_channels = 96, kernel_size = 3, padding = 'same')
        self.dec_conv4B = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, padding = 'same')
        self.upsample3 = torch.nn.Upsample(size = 8)
        self.dec_conv3A = torch.nn.Conv2d(in_channels = 144, out_channels = 96, kernel_size = 3, padding = 'same')
        self.dec_conv3B = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, padding = 'same')
        self.upsample2 = torch.nn.Upsample(size = 16)
        self.dec_conv2A = torch.nn.Conv2d(in_channels = 144, out_channels = 96, kernel_size = 3, padding = 'same')
        self.dec_conv2B = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, padding = 'same')
        self.upsample1 = torch.nn.Upsample(size = 32)
        self.dec_conv1A = torch.nn.Conv2d(in_channels = 96 + 3, out_channels = 64, kernel_size = 3, padding = 'same')
        self.dec_conv1B = torch.nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 'same')
        self.dec_conv1C = torch.nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = 3, padding = 'same')
  
    def forward(self, x):
        x0 = F.leaky_relu(self.enc_conv0(x), negative_slope = 0.1)
        x1 = self.pool1(F.leaky_relu(self.enc_conv1(x0), negative_slope = 0.1))
        #print("x1 shape : ", x1.shape)
        x2 = self.pool2(F.leaky_relu(self.enc_conv2(x1), negative_slope = 0.1))
        #print("x2 shape : ", x2.shape)
        x3 = self.pool3(F.leaky_relu(self.enc_conv3(x2), negative_slope = 0.1))
        #print("x3 shape : ", x3.shape)
        x4 = self.pool4(F.leaky_relu(self.enc_conv4(x3), negative_slope = 0.1))
        #print("x4 shape : ", x4.shape)
        x5 = self.pool5(F.leaky_relu(self.enc_conv5(x4), negative_slope = 0.1))
        #print("x5 shape : ", x5.shape)
        x6 = F.leaky_relu(self.enc_conv6(x5), negative_slope = 0.1)
        #print("x6 shape : ", x6.shape)


        x7 = self.upsample5(x6)
        #print("upsample")
        #print("x7 shape : ", x7.shape)
        #print("size of concat. : x7 : ", x7.shape, " x4 : ", x4.shape)
        x8 = torch.cat((x7, x4), dim = 1)
        #print("after concat. : x8 : ", x8.shape)
        x9 = F.leaky_relu(self.dec_conv5A(x8), negative_slope = 0.1)
        #print("x9 shape : ", x9.shape)
        x10 = F.leaky_relu(self.dec_conv5B(x9), negative_slope = 0.1)
        #print("x10 shape : ", x10.shape)

        x11 = self.upsample4(x10)
        #print("upsample")
        #print("x11 shape : ", x11.shape)
        #print("size of concat. : x11 : ", x11.shape, " x3 : ", x3.shape)
        x12 = torch.cat((x11, x3), dim = 1)
        #print("after concat. : x12 : ", x12.shape)
        x13 = F.leaky_relu(self.dec_conv4A(x12), negative_slope = 0.1)
        x14 = F.leaky_relu(self.dec_conv4B(x13), negative_slope = 0.1)

        x15 = self.upsample3(x14)
        x16 = torch.cat((x15, x2), dim = 1)
        x17 = F.leaky_relu(self.dec_conv3A(x16), negative_slope = 0.1)
        x18 = F.leaky_relu(self.dec_conv3B(x17), negative_slope = 0.1)

        x19 = self.upsample2(x18)
        x20 = torch.cat((x19, x1), dim = 1)
        x21 = F.leaky_relu(self.dec_conv2A(x20), negative_slope = 0.1)
        x22 = F.leaky_relu(self.dec_conv2B(x21), negative_slope = 0.1)

        x23 = self.upsample1(x22)
        x24 = torch.cat((x23, x), dim = 1)
        x25 = F.leaky_relu(self.dec_conv1A(x24), negative_slope = 0.1)
        x26 = F.leaky_relu(self.dec_conv1B(x25), negative_slope = 0.1)

        x27 = self.dec_conv1C(x26)

        return x27