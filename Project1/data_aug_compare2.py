import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from utils import *

"""
Here we explore data augmentation. More precisely we switch some pixels in between image noisy_img_1 and noisy_img_2.
"""
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, SIZE, train = True, transform = None, switch_pixels = None):
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
            n_max, p, show = self.switch_pixels
            if torch.rand((1,1)) < p:
                if show:
                    plt.subplot(2, 2, 1)
                    plt.imshow(torch.squeeze(X_trans).permute(1, 2, 0).int()) #int since the data has been changed to float for the NN.
                    plt.title("X_trans")
                    plt.subplot(2, 2, 2)
                    plt.imshow(torch.squeeze(Y_trans).permute(1, 2, 0).int())
                    plt.title("Y_trans")

                #print("here in switch pixel not None")
                n = torch.randint(low = 50, high = n_max, size = (1,1))
                index = torch.randint(low=0, high = X_trans.shape[1], size = (n,2))
                i,j = index[:, 0], index[:, 1]
                v_x = Y_trans[:, i, j].copy()
                v_y = X_trans[:, i, j].copy()

                X_trans[:, i, j] = v_x
                Y_trans[:, i, j] = v_y

                if show: 
                    plt.subplot(2, 2, 3)
                    plt.imshow(torch.squeeze(X_trans).permute(1, 2, 0).int()) #int since the data has been changed to float for the NN.
                    plt.title("X_trans after")
                    plt.subplot(2, 2, 4)
                    plt.imshow(torch.squeeze(Y_trans).permute(1, 2, 0).int())
                    plt.title("Y_trans after")
                    plt.show()

        return X_trans, Y_trans


class MyRotateTransform(torch.nn.Module):
    def __init__(self, angles, p=0.8):
        self.angles = angles
        self.p = p

    def __call__(self, x):
        #angle = random.choice(self.angles)
        if torch.rand((1,1)) < self.p:
            rand_index = torch.randint(low=0, high = len(self.angles), size = (1,1))
            angle = self.angles[rand_index]
        else : angle = 0
        return TF.rotate(x, angle)



transform2 = transforms.RandomApply(torch.nn.ModuleList([
    # horizontal flip with probability p 
    transforms.RandomHorizontalFlip(p=0.8),
    # vertical flip with probability p 
    transforms.RandomVerticalFlip(p=0.8),
    # rotation of angle in angles with probility p
    MyRotateTransform(angles = [90, 180, 270], p=0.8)]),
    p=0.8) #randomly transform images with probability p


SIZE = 1
BATCH_SIZE = 1
n_max, p, show = 250, 0.8, True
print("n max : ", n_max)
train_set_aug = Dataset(SIZE, switch_pixels = [n_max, p, show])
train_set_non_aug =Dataset(SIZE)
#train_set = Dataset(SIZE)


# Model Initialization
#model = Noise2Noise_3()
#model_aug = Noise2Noise_3()
#model_aug.load_state_dict(model.state_dict()) 
  
# Validation using MSE Loss function
#loss_function = torch.nn.L1Loss()
  
# Using an Adam Optimizer with lr = 0.001
#optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3, betas=(0.9, 0.99))
#optimizer_aug = torch.optim.Adam(model_aug.parameters(), lr = 1e-3, betas=(0.9, 0.99))                            

# DataLoader is used to load the dataset 
# for training
#loader_1 = torch.utils.data.DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)
loader_aug = torch.utils.data.DataLoader(dataset = train_set_aug,
                                     batch_size = BATCH_SIZE,
                                     shuffle = True)                                    

print("Test loader augmentation: ")


for noisy_imgs_1, noisy_imgs_2 in loader_aug:
    plt.subplot(1, 2, 1)
    plt.imshow(torch.squeeze(noisy_imgs_1).permute(1, 2, 0).int()) #int since the data has been changed to float for the NN.
    plt.title("Noisy imgs 1")
    plt.subplot(1, 2, 2)
    plt.imshow(torch.squeeze(noisy_imgs_2).permute(1, 2, 0).int())
    plt.title("Noisy imgs 2")
    
    #print(noisy_imgs_1.shape)
    #print(noisy_imgs_2.shape)

    #noisy_imgs_1 = noisy_imgs_1.reshape(-1, 32 * 32)
    #noisy_imgs_2 = noisy_imgs_2.reshape(-1, 32 * 32)    
    # Output of Autoencoder
    #print("type : ", noisy_imgs_1.dtype)

plt.show()

"""
#OPTIMIZATION
epochs = 10
outputs_aug = []
losses_aug = []
print("Training with augmentation : ")
for epoch in range(epochs):
    print("epoch : ", epoch)
    for noisy_imgs_1, noisy_imgs_2 in loader_aug:
        #print(noisy_imgs_1.shape)
        #print(noisy_imgs_2.shape)

        #noisy_imgs_1 = noisy_imgs_1.reshape(-1, 32 * 32)
        #noisy_imgs_2 = noisy_imgs_2.reshape(-1, 32 * 32)    
        # Output of Autoencoder
        #print("type : ", noisy_imgs_1.dtype)
        reconstructed = model_aug(noisy_imgs_1)
            
        # Calculating the loss function
        loss = loss_function(reconstructed, noisy_imgs_2)
            
        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer_aug.zero_grad()
        loss.backward()
        optimizer_aug.step()
        # Storing the losses in a list for plotting
        losses_aug.append(loss.detach().numpy())
    outputs_aug.append((epochs, noisy_imgs_2, reconstructed))


outputs = []
losses = []
print("Training without augmentation : ")
for epoch in range(epochs):
    print("epoch : ", epoch)
    for noisy_imgs_1, noisy_imgs_2 in loader_1:
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
    outputs.append((epochs, noisy_imgs_2, reconstructed))

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
  
# Plotting the last 100 values
plt.plot(losses[-100:])
plt.plot(losses_aug[-100:])
plt.legend(["Without augmentation", "With augmentation"])
plt.savefig("./Data_aug/1_losses")
plt.show()

PATH = "./Data_aug/1_model.pth"
torch.save(model.state_dict(), PATH)


PATH = "./Data_aug/1_model_aug.pth"
torch.save(model_aug.state_dict(), PATH)

"""