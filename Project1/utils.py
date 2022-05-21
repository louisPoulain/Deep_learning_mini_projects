import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio: denoised and ground ̇truth have range [0, 1] 
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10**-8)

class SSIM(nn.Module):
    def __init__(self, max_val = 1.):
        super(SSIM, self).__init__()
        self.max_val = max_val
        #window of size 7 x 7 
        window = torch.tensor([[0.0013, 0.0041, 0.0079, 0.0099, 0.0079, 0.0041, 0.0013],
        [0.0041, 0.0124, 0.0241, 0.0301, 0.0241, 0.0124, 0.0041],
        [0.0079, 0.0241, 0.0470, 0.0587, 0.0470, 0.0241, 0.0079],
        [0.0099, 0.0301, 0.0587, 0.0733, 0.0587, 0.0301, 0.0099],
        [0.0079, 0.0241, 0.0470, 0.0587, 0.0470, 0.0241, 0.0079],
        [0.0041, 0.0124, 0.0241, 0.0301, 0.0241, 0.0124, 0.0041],
        [0.0013, 0.0041, 0.0079, 0.0099, 0.0079, 0.0041, 0.0013]])
        self.kernel = window.repeat(3, 1, 1, 1) #repeat the window on the 3 channels

        self.C1 = (0.01 * self.max_val) ** 2
        self.C2 = (0.03 * self.max_val) ** 2

    def forward(self, groundtruth, noisy_img): 
        kernel = self.kernel.to(groundtruth.device).to(noisy_img.dtype)
        mu1 = F.conv2d(groundtruth, kernel, groups = 3)
        mu2 = F.conv2d(noisy_img, kernel, groups = 3)

        mu1_square = mu1.pow(2)
        mu2_square = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_square =  F.conv2d(groundtruth * groundtruth, kernel, groups = 3) - mu1_square
        sigma2_square =  F.conv2d(noisy_img * noisy_img, kernel, groups = 3) - mu2_square
        sigma12 =  F.conv2d(groundtruth * noisy_img, kernel, groups = 3) - mu1_mu2

        ssim = ((2 * mu1_mu2 + self.C1)*(2 * sigma12 + self.C2)) / ((mu1_square + mu2_square + self.C1) * (sigma1_square + sigma2_square + self.C2))

        loss = 1-ssim
        loss = torch.mean(loss)

        return loss


def ssim(ground_truth, noisy_img, max_val = 1.):
    return SSIM(max_val = max_val)(ground_truth , noisy_img)

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, SIZE, train = True):
        'Initialization'
        if train: 
            if SIZE > 50000: 
                print("SIZE is to big. It is set to SIZE = 50000.")
                SIZE = 50000
            x, y = torch.load("train_data.pkl")
            print("Training data : \n noisy_imgs_1 : ", x.shape, "\n noisy_imgs_2 : ", y.shape)
        else : 
            if SIZE > 1000:
                print("SIZE is to big. It is set to SIZE = 1000.")
                SIZE = 1000
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

def plot_3imgs(denoised, ground_truth, noisy_imgs): 
    #plots the 3 given images. Values of the images are in between [0, 255].
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

def compare_models(PATHS, models, names = None, SIZE = 1000):
    # Given a list PATHS of PATH (str) that leads to a model in the corresponding list models,
    # display a noisy image, its ground truth and the denoised images and compute the PSNR over SIZE images of the test set.
    BATCH_SIZE = 1
    print("size : ", SIZE)
    test_set = Dataset(SIZE, train = False)
    loader_1 = torch.utils.data.DataLoader(dataset = test_set,
                                     batch_size = BATCH_SIZE,
                                     shuffle = False)
    #number of images to plot
    N = (len(PATHS) + 2)//2
    for j, PATH in enumerate(PATHS):
        model = models[j]
        model.load_state_dict(torch.load(PATH))
        PSNR = torch.empty(size = (1, SIZE))
        SSIMs = torch.empty(size = (1, SIZE))
        i = 0
        for noisy_imgs, ground_truth in loader_1:
            denoised = model(noisy_imgs)
            PSNR[0, i] = psnr(denoised/255, ground_truth/255)
            SSIMs[0, i] = 1-ssim(ground_truth/255, denoised/255)
            i += 1
        print("Model ", j, " PSNR:", torch.mean(PSNR).item()) #display the mean of PSNR over the test set.
        print("Model ", j, " SSIM:", torch.mean(SSIMs).item())
        if j%2 == 0:
            plt.subplot(2, N, j//2 + 2)
        else :
            plt.subplot(2, N, j//2 + N + 2)
        plt.imshow(torch.squeeze(denoised).permute(1, 2, 0).int())
        if names != None:
            plt.title(names[j])
        else:
            plt.title("model " + str(j))
    
    plt.subplot(2, N, 1)
    plt.imshow(torch.squeeze(noisy_imgs).permute(1, 2, 0).int()) #int since the data has been changed to float for the NN.
    plt.title("Noisy image")
    plt.subplot(2, N, N + 1)
    plt.imshow(torch.squeeze(ground_truth).permute(1, 2, 0).int())
    plt.title("Groundtruth")
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

class Noise2Noise_2(torch.nn.Module): #we replace the upsampling by conv transpose
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
        
        #we replace an upsampling by a conv transpose
        self.upsample5 = torch.nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1) #torch.nn.Upsample(size=(2,2))
        self.dec_conv5A = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, padding = 'same')
        self.dec_conv5B = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, padding = 'same')
        self.upsample4 = torch.nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1) #torch.nn.Upsample(size=4)
        self.dec_conv4A = torch.nn.Conv2d(in_channels = 144, out_channels = 96, kernel_size = 3, padding = 'same')
        self.dec_conv4B = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, padding = 'same')
        self.upsample3 = torch.nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1) #torch.nn.Upsample(size = 8)
        self.dec_conv3A = torch.nn.Conv2d(in_channels = 144, out_channels = 96, kernel_size = 3, padding = 'same')
        self.dec_conv3B = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, padding = 'same')
        self.upsample2 = torch.nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1) #torch.nn.Upsample(size = 16)
        self.dec_conv2A = torch.nn.Conv2d(in_channels = 144, out_channels = 96, kernel_size = 3, padding = 'same')
        self.dec_conv2B = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, padding = 'same')
        self.upsample1 = torch.nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1) #torch.nn.Upsample(size = 32)
        self.dec_conv1A = torch.nn.Conv2d(in_channels = 96 + 3, out_channels = 64, kernel_size = 3, padding = 'same')
        self.dec_conv1B = torch.nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 'same')
        self.dec_conv1C = torch.nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = 3, padding = 'same')
  
    def forward(self, x):
        """     ENCODER      """
        """Block 1"""
        x0 = F.leaky_relu(self.enc_conv0(x), negative_slope = 0.1)
        x1 = self.pool1(F.leaky_relu(self.enc_conv1(x0), negative_slope = 0.1))
        #print("x1 shape : ", x1.shape)
        
        """Block 2"""
        x2 = self.pool2(F.leaky_relu(self.enc_conv2(x1), negative_slope = 0.1))
        #print("x2 shape : ", x2.shape)
        x3 = self.pool3(F.leaky_relu(self.enc_conv3(x2), negative_slope = 0.1))
        #print("x3 shape : ", x3.shape)
        x4 = self.pool4(F.leaky_relu(self.enc_conv4(x3), negative_slope = 0.1))
        #print("x4 shape : ", x4.shape)
        x5 = self.pool5(F.leaky_relu(self.enc_conv5(x4), negative_slope = 0.1))
        #print("x5 shape : ", x5.shape)

        """     DECODER     """
        """Block 3"""
        x6 = F.leaky_relu(self.enc_conv6(x5), negative_slope = 0.1)
        #print("x6 shape : ", x6.shape)
        x7 = self.upsample5(x6)
        #print("upsample")
        #print("x7 shape : ", x7.shape)
        #print("size of concat. : x7 : ", x7.shape, " x4 : ", x4.shape)
        x8 = torch.cat((x7, x4), dim = 1)
        #print("after concat. : x8 : ", x8.shape)

        """Block 4"""
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

class Noise2Noise_3(torch.nn.Module): 
    #we replace the upsampling by conv transpose and make it shorter.
    def __init__(self):
        super().__init__()

        self.enc_conv0 = torch.nn.Conv2d(in_channels = 3, out_channels = 48, kernel_size = 3, padding = 'same')
        self.enc_conv1 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2)
        self.enc_conv2 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        self.pool2 = torch.nn.MaxPool2d(kernel_size = 2)
        self.enc_conv5 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        self.pool5 = torch.nn.MaxPool2d(kernel_size = 2)
        self.enc_conv6 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        
        #we replace an upsampling by a conv transpose
        self.upsample5 = torch.nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1) #torch.nn.Upsample(size=(2,2))
        self.dec_conv3A = torch.nn.Conv2d(in_channels = 96, out_channels = 48, kernel_size = 3, padding = 'same')
        self.dec_conv3B = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        self.upsample2 = torch.nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1) #torch.nn.Upsample(size = 16)
        self.dec_conv2A = torch.nn.Conv2d(in_channels = 96, out_channels = 48, kernel_size = 3, padding = 'same')
        self.dec_conv2B = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')
        self.upsample1 = torch.nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1) #torch.nn.Upsample(size = 32)
        self.dec_conv1A = torch.nn.Conv2d(in_channels = 48 + 3, out_channels = 32, kernel_size = 3, padding = 'same')
        self.dec_conv1B = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same')
        self.dec_conv1C = torch.nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = 3, padding = 'same')

    def forward(self, x):
        """     ENCODER      """
        """Block 1"""
        x0 = F.leaky_relu(self.enc_conv0(x), negative_slope = 0.1) 
        x1 = self.pool1(F.leaky_relu(self.enc_conv1(x0), negative_slope = 0.1)) #48 x 16 x 16
        #print("x1 shape : ", x1.shape)
        
        """Block 2"""
        x2 = self.pool2(F.leaky_relu(self.enc_conv2(x1), negative_slope = 0.1)) #48 x 8 x 8
        #print("x2 shape : ", x2.shape)
        x5 = self.pool5(F.leaky_relu(self.enc_conv5(x2), negative_slope = 0.1)) #48 x 4 x 4
        #print("x5 shape : ", x5.shape)

        """     DECODER     """
        """Block 3"""
        x6 = F.leaky_relu(self.enc_conv6(x5), negative_slope = 0.1) #48 x 4 x 4
        #print("x6 shape : ", x6.shape)
        x7 = self.upsample5(x6) #48 x 8 x 8
        #print("upsample")
        #print("x7 shape : ", x7.shape)
        

        x16 = torch.cat((x7, x2), dim = 1) #96 x 8 x 8
        #print("size of concat. : x16 : ", x16.shape)
        x17 = F.leaky_relu(self.dec_conv3A(x16), negative_slope = 0.1)
        #print("x17 shape : ", x17.shape)
        x18 = F.leaky_relu(self.dec_conv3B(x17), negative_slope = 0.1)
        #print("x18 shape : ", x18.shape)

        x19 = self.upsample2(x18)
        ##print("x19 shape : ", x19.shape)
        x20 = torch.cat((x19, x1), dim = 1)
        #print("concat : x20 shape : ", x20.shape)
        x21 = F.leaky_relu(self.dec_conv2A(x20), negative_slope = 0.1)
        #print("x21 shape : ", x21.shape)
        x22 = F.leaky_relu(self.dec_conv2B(x21), negative_slope = 0.1)
        #print("x22 shape : ", x22.shape)

        x23 = self.upsample1(x22)
        #print("x23 shape : ", x23.shape)
        x24 = torch.cat((x23, x), dim = 1)
        #print("concat x24 shape : ", x24.shape)
        x25 = F.leaky_relu(self.dec_conv1A(x24), negative_slope = 0.1)
        #print("x25 shape : ", x25.shape)
        x26 = F.leaky_relu(self.dec_conv1B(x25), negative_slope = 0.1)
        #print("x26 shape : ", x26.shape)

        x27 = self.dec_conv1C(x26)

        return x27

class AE_small5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1) # 32 x 32 
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 24, kernel_size = 3, padding = 1) 
        self.conv3 = nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size = 3, padding = 1) 
        self.conv4 = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 1) 
        self.conv5 = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 1) 
        
        self.deconv1 = nn.Conv2d(in_channels = 96, out_channels = 48, kernel_size = 3, padding = 1) 
        self.deconv2 = nn.Conv2d(in_channels = 96, out_channels = 48, kernel_size = 3, padding = 1) 
        self.deconv3 = nn.Conv2d(in_channels = 72, out_channels = 24, kernel_size = 3, padding = 1) 
        self.deconv4 = nn.Conv2d(in_channels = 40, out_channels = 16, kernel_size = 3, padding = 1) 
        self.deconv5 = nn.Conv2d(in_channels = 19, out_channels = 3, kernel_size = 3, padding = 1) 
        
        self.l_relu = nn.LeakyReLU(negative_slope = 0.1)
        self.upsample = nn.Upsample(scale_factor = (2, 2))
        self.linear = nn.Linear(32, 32)
        #self.dropout = nn.Dropout(0.5)
        
        
  
    def forward(self, x):
        # encode
        x1 = self.l_relu(self.conv1(x))
        x1 = self.l_relu(self.pool(x1))

        x2 = self.l_relu(self.conv2(x1))
        x2 = self.l_relu(self.pool(x2))

        x3 = self.l_relu(self.conv3(x2))
        x3 = self.l_relu(self.pool(x3))

        x4 = self.l_relu(self.conv4(x3))
        x4 = self.l_relu(self.pool(x4))

        x5 = self.l_relu(self.conv5(x4))
        #print(x5.shape)

        # decode
        y1 = torch.cat((x5, x4), dim = 1)
        y1 = self.l_relu(self.upsample(y1))
        y1 = self.l_relu(self.deconv1(y1))
        #print(y1.shape)

        y2 = torch.cat((y1, x3), dim = 1)
        y2 = self.l_relu(self.upsample(y2))
        y2 = self.l_relu(self.deconv2(y2))
        #print(y2.shape)

        y3 = torch.cat((y2, x2), dim = 1)
        y3 = self.l_relu(self.upsample(y3))
        y3 = self.l_relu(self.deconv3(y3))
        #print(y3.shape)

        y4 = torch.cat((y3, x1), dim = 1)
        y4 = self.l_relu(self.upsample(y4))
        y4 = self.l_relu(self.deconv4(y4))
        #print(y4.shape)

        y5 = torch.cat((y4, x), dim = 1)
        #print(y5.shape)
        y5 = self.linear(self.deconv5(y5))
        
        return y5