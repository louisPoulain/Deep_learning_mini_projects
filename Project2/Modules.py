import numpy as np
import torch
import math

#########################################################################################################################################

class Module (object) :
    def forward (self, input) :
        raise NotImplementedError
    def backward (self, gradwrtoutput):
        raise NotImplementedError 
    def param (self) :
        return []
    def zero_grad(self):
        return

#########################################################################################################################################


class Conv2d(Module):
      def __init__(self,in_channels,out_channels,kernel,stride=(1,1),padding=(0,0),dilation=(1,1),bias_=True):
             super().__init__()
             self.in_channels = in_channels
             self.out_channels = out_channels
             self.kernel_size = kernel
             self.stride =stride
             self.padding= padding
             self.dilation = dilation
             self.bias_ = bias_
            

             
             #Initialization of the parameters
             p = self.in_channels* self.kernel_size[0]* self.kernel_size[1]
             
             self.weights = torch.empty(size = (out_channels,in_channels,kernel[0],kernel[1])).data.uniform_(-1. /math.sqrt(p), 1. /math.sqrt(p))
             self.grad_weight = torch.empty(self.weights.shape)
             
             if self.bias_:
                self.bias = torch.empty(1,self.out_channels).data.uniform_(-1. / math.sqrt(p), 1. / math.sqrt(p))
                self.grad_bias = torch.empty(self.bias.shape)
             
             
      def forward(self,input):
             
             self.h_in = input.shape[2]
             self.w_in = input.shape[3]
             h_out =  int((self.h_in + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1)/self.stride[0] + 1)
             w_out = int((self.w_in+ 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1] - 1) - 1)/self.stride[1] + 1)
             
             self.unfolded = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding = self.padding, dilation=self.dilation)
             self.w = self.weights.view(self.out_channels, -1)
             self.wxb = self.w @ self.unfolded 
             
             if self.bias_:
                self.wxb+= self.bias.view(1, -1, 1)
             
             self.result = self.wxb.view(input.shape[0], self.out_channels, h_out, w_out)
             
             return self.result

      def backward (self, gradwrtoutput):
           
            
             if self.bias_:
                self.grad_bias = gradwrtoutput.sum(dim=[0, 2, 3])

             a = gradwrtoutput.reshape(self.wxb.shape) #1 x 4 x 961
             
             b = a.transpose(1, 2)
             
             c = b.matmul(self.w) #1x961x12
             #c = self.w.view(self.out_channels,-  1).t() @ a
             

             #d = self.unfolded.matmul(b) #1 x 12 x 4
             d = a @ self.unfolded.transpose(1,2)

             e = d.sum(dim=0) #12x4
             

             f = e.t()
            
             self.grad_weight = e.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]) 
             
             
             h = c.transpose(1, 2)
             
             self.grad_inputX = torch.nn.functional.fold(h, (self.h_in,self.w_in), self.kernel_size,
                                                stride=self.stride,padding = self.padding)

             return self.grad_inputX
      
      def param (self) :
            if self.bias_ :
                return [[self.weights, self.grad_weight],
                    [self.bias, self.grad_bias]]
            else:
                return [[self.weights, self.grad_weight]]

      def zero_grad(self):
        self.grad_weight.zero_()
        if self.bias_:
            self.grad_bias.zero_()
    

#########################################################################################################################################


class NearestUpsampling(Module):
    def __init__(self,scale_factor):
        super().__init__()
        self.b0 = scale_factor
        self.b1 = scale_factor
        

    def forward(self, inputs):
        batch,chan,h,w = inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
        bs,cs,hs,ws = inputs.stride(dim=0),inputs.stride(dim=1),inputs.stride(dim=2),inputs.stride(dim=3)
        
        x = inputs.as_strided( (batch ,chan,h, self.b0, w, self.b1), (bs ,cs, hs, 0, ws, 0)) 
        out = x.reshape(batch,chan,h*self.b0, w*self.b1)
        return out
   
    def backward(self, grad_wrt_output):
        out = grad_wrt_output.reshape(grad_wrt_output.shape[0],grad_wrt_output.shape[1], int(grad_wrt_output.shape[2]/self.b0),self.b0,int(grad_wrt_output.shape[3]/self.b1),self.b1)
        out = torch.sum(out,dim=3)
        out = torch.sum(out,dim=4)
        return out
    
    def param(self): 
      return []

class transpose(Module):
    def __init__(self,scale_factor):
        super().__init__()
        self.b0 = scale_factor
        self.b1 = scale_factor
        

    def forward(self, inputs):
        batch,chan,h,w = inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
        bs,cs,hs,ws = inputs.stride(dim=0),inputs.stride(dim=1),inputs.stride(dim=2),inputs.stride(dim=3)
        x = inputs.as_strided( (batch ,chan,h, self.b0, w, self.b1), (bs ,cs, hs, 0, ws, 0)) 
        out = x.reshape(batch,chan,h*self.b0, w*self.b1)
        return out
   
    def backward(self, grad_wrt_output):
        out = grad_wrt_output.reshape(grad_wrt_output.shape[0],grad_wrt_output.shape[1], int(grad_wrt_output.shape[2]/self.b0),self.b0,int(grad_wrt_output.shape[3]/self.b1),self.b1)
        out = torch.sum(out,dim=3)
        out = torch.sum(out,dim=4)
        return out
    
    def param(self): 
      return []

   
#########################################################################################################################################


class Sigmoid(Module):
    def forward(self,input):
        self.output = 1/(1+torch.exp(-input))
        return self.output
    
    def backward (self, gradwrtoutput):
        self.out = self.output*(1-self.output)*gradwrtoutput
        return self.out

    def param(self):
        return []
    


class Relu(Module):
    def forward(self,input):
        self.input = input
        output = (self.input>0)*self.input
        return output
    
    def backward (self, gradwrtoutput):
        self.out = (self.input>0)*1*gradwrtoutput
        return self.out

    def param(self):
        return []
   
#########################################################################################################################################

class Sequential(Module):
    def __init__(self,*lay):
        super().__init__()
        self.layers = []
        for layer in lay:
            self.layers.append(layer)
        
    def add(self,layer):
        return self.layers.append(layer)

    def forward(self, input):        
        output = input      
        for layer in self.layers:
            output = layer.forward(output)
        return output
      
    def backward(self, grad_wrt_output):
        for layer in reversed(self.layers):
            grad_wrt_output = layer.backward(grad_wrt_output)
        return grad_wrt_output
    
                
    def param(self):
        self.parameters = []
        for layer in self.layers:
            self.parameters.extend(layer.param())
        return self.parameters

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

#########################################################################################################################################

class MSE(Module):
  
    def forward(self,input,target):
        self.output = ((input-target)**2).mean()
        return self.output
    
    def backward (self,input,target):
            out = (2*(input-target))/torch.numel(input)
            return out

    def param(self):
        return []

#########################################################################################################################################

class SGD(Module):
    def __init__(self,lr,parameter):
        self.lr = lr
        self.parameter = parameter
    
    def forward(self,model):
        self.parameter = model.param()
        for param in self.parameter:
            if (param[0]==None or param[1]==None):
                continue
            else:
                param[0]-=self.lr*param[1] 
            
            
    def param(self):
        return []




