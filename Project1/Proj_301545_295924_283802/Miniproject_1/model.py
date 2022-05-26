import torch
import torch.nn.functional as F
import torch.nn as nn
from .others import otherfile1 as utils
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### For mini-project 1
class Model():
    def  __init__(self) -> None:
    ## instantiate model + optimizer + loss function + any other stuff you need
        self.model = utils.AE().to(device)
        self.BATCH_SIZE = 256
        self.loss_function = nn.L1Loss().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                             lr = 1e-3, betas=(0.9, 0.99))
        
    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel.pth into the model pass
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) #REMOVE MAP LOCATION AT THE END (ONLY FOR MY COMPUTER)
        
    def train(self, train_input, train_target, num_epochs = 10) -> None:
        #:train ̇input: tensor of size (N, C, H, W) containing a noisy version of the images. same images, which only differs from the input by their noise.
        #:train ̇target: tensor of size (N, C, H, W) containing another noisy version of the
        train_set = utils.Dataset(train_input, train_target) 
        loader_1 = torch.utils.data.DataLoader(dataset = train_set,
                                        batch_size = self.BATCH_SIZE,
                                        shuffle = True)
        for epoch in range(num_epochs):
            print("epoch : ", epoch + 1)
            for noisy_imgs_1, noisy_imgs_2 in loader_1:
                noisy_imgs_1 = noisy_imgs_1.to(device)
                noisy_imgs_2 = noisy_imgs_2.to(device)
                reconstructed = self.model(noisy_imgs_1)
                    
                # Calculating the loss function
                loss = self.loss_function(reconstructed, noisy_imgs_2)
                    
                # The gradients are set to zero,
                # the the gradient is computed and stored.
                # .step() performs parameter update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, test_input) -> torch.Tensor:
        #:test ̇input: tensor of size (N1, C, H, W) with values in range 0-255 that has to be denoised by the trained or the loaded network.
        #:returns a tensor of the size (N1, C, H, W) with values in range 0-255.
        reconstructed = self.model(test_input)
        return reconstructed
