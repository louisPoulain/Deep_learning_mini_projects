import torch
import random
from pathlib import Path
from .others.Modules import Module, MSE ,Conv2d ,NearestUpsampling ,ReLU ,Sigmoid ,SGD ,Sequential
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model():
    def  __init__(self):
       
        self.seq = Sequential(Conv2d(3,100,(3,3),(2,2),padding = (1,1)),ReLU(),
                              Conv2d(100,100,(3,3),(2,2),padding=(1,1)),ReLU(),
                              NearestUpsampling(2),Conv2d(100,100,(3,3),stride=(1,1),padding = (1,1)),ReLU(),
                              NearestUpsampling(2), Conv2d(100,3,(3,3),stride=(1,1),padding = (1,1)),Sigmoid())

        self.criterion = MSE()
        self.optimizer = SGD(0.1,self.seq.param())
        

    def load_pretrained_model(self) :
        filename = Path(__file__).parent / "bestmodel.pickle"
        with open(filename, 'rb') as f:
            param = pickle.load(f)
            i = 0
            for j, layer in enumerate(self.seq.layers):
                if isinstance(layer, Conv2d):
                    layer.weight = param[i][0]
                    layer.grad_weight = param[i][1]
                    i = i + 1
                    layer.bias = param[i][0]
                    layer.grad_bias = param[i][1]
                    i = i + 1 

        
    def train(self, x_train,y_train, num_epochs = 1, batch_size = 1):
        N = len(x_train)
        x_train = x_train/255
        y_train = y_train/255

        for e in range(num_epochs):
            print("Epoch:{i} ".format(i=e))
            ind = list(range(N))
            random.shuffle(ind)
            x_train = x_train[ind]
            y_train = y_train[ind]
            
            for i in range(0,N, batch_size):
                
                output = self.seq.forward(x_train.narrow(0,i,batch_size))
                loss = self.criterion.forward(output,y_train.narrow(0,i,batch_size))
                self.seq.zero_grad()
                grad_wrt_output = self.criterion.backward(output,y_train.narrow(0,i,batch_size))

                self.seq.backward(grad_wrt_output)
                self.optimizer.forward(self.seq)
                
                
    def save_model(self):
        filename = Path(__file__).parent / "bestmodel.pickle"
        with open(filename, 'wb') as f:
            param = self.seq.param()
            pickle.dump(param, f)
        
       
    def predict(self, x):
        if x.max() > 1:
            x = x/255.0
            return self.seq.forward(x)*255
        else : 
            return self.seq.forward(x)

    