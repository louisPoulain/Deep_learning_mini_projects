import torch
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import *


"""class FunctionWrapperDouble2(Repr):
    "A function wrapper that returns a partial for an input-target pair."

    def __init__(self, function: Callable, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        
        

    def __call__(self, inp: np.ndarray, tar: dict):
        inp,tar = self.function(inp,tar)
        return inp, tar

transforms_training = ComposeDouble([
     FunctionWrapperDouble2(flip_horizontally),
     FunctionWrapperDouble2(flip_vertically),
     FunctionWrapperDouble2(rot90),
     FunctionWrapperDouble(add_gaussian_noise,input=True,target=False),
     FunctionWrapperDouble(resized,input=True,target=False),
     FunctionWrapperDouble(normalize_0,input=True,target=False),
     FunctionWrapperDouble(crop)
])# validation transformations

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long(memory_format=torch.preserve_format)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        #input_ID = self.inputs[index]
        #target_ID = self.targets[index]
        x = self.inputs[index]
        y = self.targets[index]
        x = np.array(x)
        y = np.array(y)
        
        # Load input and target
        #x, y = imread(input_ID), imread(target_ID)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y)#.type(self.targets_dtype)

        return x, y
    




class SegmentationDataSet3(data.Dataset):
    "Image segmentation dataset with caching, pretransforms and multiprocessing."
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            from multiprocessing import Pool
            from itertools import repeat

            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets, repeat(self.pre_transform)))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            x = self.inputs[index]
            y = self.targets[index]
            x = np.array(x,dtype=object)
            y = np.array(y,dtype=object)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)
            x= x.astype(float)
            y = y.astype(float)
           

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).float()#type(self.targets_dtype)#.float()#

        return x, y

    @staticmethod
    def read_images(inp, tar, pre_transform):
        inp = np.array(inp,dtype=object)
        tar = np.array(tar,dtype=object)
        if pre_transform:
            inp, tar = pre_transform(inp, tar)
        return inp, tar

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long(memory_format=torch.preserve_format)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        #input_ID = self.inputs[index]
        #target_ID = self.targets[index]
        x = self.inputs[index]
        y = self.targets[index]
        x = np.array(x)
        y = np.array(y)
        
        # Load input and target
        #x, y = imread(input_ID), imread(target_ID)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y)#.type(self.targets_dtype)

        return x, y

dataset_train = SegmentationDataSet4(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms_training,
                                    use_cache= False,
                                    pre_transform = None
                                    )

    # dataset validation
dataset_valid = SegmentationDataSet4(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms_validation,
                                    use_cache= False,
                                    pre_transform = None
                                    )



# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=batch*N,
                                 shuffle=False)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=batch*N,
                                   shuffle=False)"""

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, SIZE, train = True, transform = None):
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

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        # get label
        X = self.x[index]
        Y = self.y[index]
        X_trans = X
        Y_trans= Y

        seed = torch.randint(2147483647,(1,1)) # make a seed with numpy generator 
        print("seed : ", seed.item())
        #random.seed(seed) # apply this seed to img transforms
        torch.manual_seed(seed.item()) # needed for torchvision 0.7
        print("pb here")
        if self.transform is not None:
            X_trans = self.transform(X)

        #random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed.item()) # needed for torchvision 0.7
        if self.transform is not None:
            Y_trans = self.transform(Y)  

        return X, Y, X_trans, Y_trans

transform = transforms.Compose([
    #transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0),
    # random crop and then resize
    transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)), 
    # horizontal flip with probability p 
    transforms.RandomHorizontalFlip(p=0.5),
    # vertical flip with probability p 
    transforms.RandomVerticalFlip(p=0.5)
])


#we replace the upsampling by conv transpose and shorten the model

SIZE = 4
BATCH_SIZE = 1
train_set = Dataset(SIZE, transform=transform)


# Model Initialization
model = Noise2Noise_3()
  
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
epochs = 2
outputs = []
losses = []
for epoch in range(epochs):
    print("epoch : ", epoch)
    for noisy_imgs_1, noisy_imgs_2, noisy_imgs_1_trans, noisy_imgs_2_trans in loader_1:
        #plots the 4 given images. Values of the images are in between [0, 255].
        plt.subplot(1, 4, 1)
        plt.imshow(torch.squeeze(noisy_imgs_1).permute(1, 2, 0).int()) #int since the data has been changed to float for the NN.
        plt.title("Noisy imgs 1")
        plt.subplot(1, 4, 2)
        plt.imshow(torch.squeeze(noisy_imgs_2).permute(1, 2, 0).int())
        plt.title("Noisy imgs 2")
        plt.subplot(1,4,3)
        plt.imshow(torch.squeeze(noisy_imgs_1_trans).permute(1, 2, 0).int())
        plt.title("transformed 1")
        plt.subplot(1,4,4)
        plt.imshow(torch.squeeze(noisy_imgs_2_trans).permute(1, 2, 0).int())
        plt.title("transformed 2")
        plt.show()
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
    outputs.append((epochs, noisy_imgs_2, reconstructed))
""" 
# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
  
# Plotting the last 100 values
plt.plot(losses[-100:])
plt.show()

PATH = "./Noise2Noise/project1_4.pth"
torch.save(model.state_dict(), PATH)"""