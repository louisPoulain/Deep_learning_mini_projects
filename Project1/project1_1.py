import torch
import matplotlib.pyplot as plt
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
            print("Type : ", x.dtype)
        else : 
            x, y = torch.load("val_data.pkl")
            print("Test data : \n noisy_imgs : ", x.shape, "\n clean_imgs : ", y.shape)
        x, y = x[:SIZE], y[:SIZE]
        print("Data reduced : \n noisy_imgs_1_reduced : ", x.shape, "\n noisy_imgs_2_reduced : ", y.shape)

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

SIZE = 5000
BATCH_SIZE = 4
train_set = Dataset(SIZE)
N = 20
plt.figure()

"""
for i in range(N):
    plt.subplot(2, N, 2*i+1)
    plt.imshow(train_set.x[i].permute(1, 2, 0))
    plt.subplot(2, N, 2*i+2)
    plt.imshow(train_set.y[i].permute(1, 2, 0))
plt.show()
"""

#let's create a simple auto-encoder first:
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope=0.1),
            
            torch.nn.LeakyReLU(negative_slope = 0.1),
            torch.nn.MaxPool2d(kernel_size = 2),

            torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same'),
            torch.nn.LeakyReLU(negative_slope = 0.1),
            torch.nn.MaxPool2d(kernel_size = 2),

            torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same'),
            torch.nn.LeakyReLU(negative_slope = 0.1),
            torch.nn.MaxPool2d(kernel_size = 2),
            
            torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same'),
            torch.nn.LeakyReLU(negative_slope = 0.1),
            torch.nn.MaxPool2d(kernel_size = 2),
            
            torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same'),
            torch.nn.LeakyReLU(negative_slope = 0.1),
            torch.nn.MaxPool2d(kernel_size = 2),
            
            torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same'),
            torch.nn.Upsample(size=(2,2))
        )

        self.enc_conv0 = torch.nn.Conv2d(in_channels = 3, out_channels = 48, kernel_size = 3, padding = 'same'),
        self.enc_conv1 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 'same')

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels = 2, out_channels = 3, kernel_size = (5, 5)),
            torch.nn.Sigmoid()
        )
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Model Initialization
model = AE()
  
# Validation using MSE Loss function
loss_function = torch.nn.L1Loss()
  
# Using an Adam Optimizer with lr = 0.001
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-3, betas=(0.9, 0.99))

# DataLoader is used to load the dataset 
# for training
loader_1 = torch.utils.data.DataLoader(dataset = train_set,
                                     batch_size = BATCH_SIZE,
                                     shuffle = True)


#OPTIMIZATION
epochs = 20
outputs = []
losses = []
for epoch in range(epochs):
    for noisy_imgs_1, noisy_imgs_2 in loader_1:
        #print(noisy_imgs_1.shape)
        #print(noisy_imgs_2.shape)

        #noisy_imgs_1 = noisy_imgs_1.reshape(-1, 32 * 32)
        #noisy_imgs_2 = noisy_imgs_2.reshape(-1, 32 * 32)    
        # Output of Autoencoder
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
    outputs.append((epochs, noisy_imgs_2, reconstructed))
  
# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
  
# Plotting the last 100 values
plt.plot(losses[-100:])
plt.show()
                        