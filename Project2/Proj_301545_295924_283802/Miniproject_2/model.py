from pickletools import optimize
import os
import numpy as np
import torch
import math
import random
from pathlib import Path
from .others.Modules import Module, MSE ,Conv2d ,NearestUpsampling ,ReLU ,Sigmoid ,SGD ,Sequential
import pickle


class Model():
    def  __init__(self):
       
        self.seq = Sequential(Conv2d(3,64,(3,3),(2,2),padding = (1,1)),ReLU(),
                              Conv2d(64,64,(3,3),(2,2),padding=(1,1)),ReLU(),
                              NearestUpsampling(2),Conv2d(64,64,(3,3),stride=(1,1),padding = (1,1)),ReLU(),
                              NearestUpsampling(2), Conv2d(64,3,(3,3),stride=(1,1),padding = (1,1)),Sigmoid())

        self.criterion = MSE()
        self.optimizer = SGD(0.1,self.seq.param())
        

    def load_pretrained_model(self) :
        filename = Path(__file__).parent / "bestmodel.pikle"
        #filename = 'Miniproject_2/bestmodel.pkl'
        with open(filename, 'rb') as f:
            param = pickle.load(f)
            i = 0
            for j, module in enumerate(self.seq.modules):
                if isinstance(module, Conv2d):
                    module.weight = param[i][0]
                    module.grad_weight = param[i][1]
                    i = i + 1
                    module.bias = param[i][0]
                    module.grad_bias = param[i][1]
                    i = i + 1 

    
    def train(self, x_train,y_train, num_epochs = 10, batch_size = 1):
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
        filename = Path(__file__).parent / "bestmodel.pikle"
        with open(filename) as f:
            param = self.seq.param()
            pickle.dump(param, f)
      
    def predict(self, x):
       x = x/255.0
       return self.seq.forward(x)*255

    class Model_val():
        def  __init__(self):
       
            self.seq = Sequential(Conv2d(3,64,(3,3),(2,2),padding = (1,1)),ReLU(),
                              Conv2d(64,64,(3,3),(2,2),padding=(1,1)),ReLU(),
                              NearestUpsampling(2),Conv2d(64,64,(3,3),stride=(1,1),padding = (1,1)),ReLU(),
                              NearestUpsampling(2), Conv2d(64,3,(3,3),stride=(1,1),padding = (1,1)),Sigmoid())

            self.criterion = MSE()
            self.optimizer = SGD(0.1,self.seq.param())
        

        def load_pretrained_model(self) :
            filename = 'Miniproject_2/bestmodel.pkl'
            mod = pickle.load(open(filename, 'rb')) 
            self.seq = mod.seq
            self.optimizer = mod.optimizer
            self.criterion = mod.criterion

    
        def train(self, x_train,y_train, x_test,y_test,num_epochs = 10, batch_size = 1):
            N = len(x_train)
            x_train = x_train/255
            y_train = y_train/255
            metrics = []
            for e in range(num_epochs):
                print("Epoch:{i} ".format(i=e))
                ind = list(range(N))
                random.shuffle(ind)
                x_train = x_train[ind]
                y_train = y_train[ind]
                met=[]
            
                for i in range(0,N, batch_size):
                
                    output = self.seq.forward(x_train.narrow(0,i,batch_size))
                    loss = self.criterion.forward(output,y_train.narrow(0,i,batch_size))
                
                
                    self.seq.zero_grad()
                    grad_wrt_output = self.criterion.backward(output,y_train.narrow(0,i,batch_size))

                    self.seq.backward(grad_wrt_output)
                    self.optimizer.forward(self.seq)
                met = self.validate(x_test,y_test)
                metrics.append(met)



                
                
        def save_model(self):
            filename = 'bestmodel.pkl'
            pickle.dump(self, open(filename, 'wb'))
      
        def predict(self, x):
            x = x/255.0
            return self.seq.forward(x)*255

        def validate(self,x,y_test):
            pred = self.predict(x)
            mse = torch.mean((pred - y_test) ** 2)
            return -10 * torch.log10(mse + 10**-8)

    