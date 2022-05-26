import torch
import math

class Module (object) :
    def forward (self, input) :
        raise NotImplementedError
    def backward (self, gradwrtoutput):
        raise NotImplementedError 
    def param (self) :
        return []
    def zero_grad(self):
        returns

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
             
             self.unfolded = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding = self.padding, dilation=self.dilation) # B x (C_in x k x k) x (H_out x W_out)
             print("unfolded : ", unfolded.shape)
             self.w = self.weights.view(self.out_channels, -1)
             self.wxb = self.w @ self.unfolded 
             
             if self.bias_:
                self.wxb+= self.bias.view(1, -1, 1)
             
             self.result = self.wxb.view(input.shape[0], self.out_channels, h_out, w_out)
             
             return self.result

      def backward (self, gradwrtoutput):
            if self.bias_:
                self.grad_bias = gradwrtoutput.sum(dim=[0, 2, 3])
            print("gradwrtoutput: ", gradwrtoutput.shape)
            a = gradwrtoutput.reshape(self.wxb.shape) #1 x C_out x (H_out x W_out)
            print("a : ", a.shape)
            b = a.transpose(1, 2) # 1 x (H_out x W_out) x C_out
            print("transpose, b : ", b.shape)
            print("weights : ", self.w.shape) #C_out x (C_in x k x k)
            c = b.matmul(self.w) #1 x (H_out x W_out) x (C_in x k x k)
            print("multiply b with weigths, c :", c.shape)
            h = c.transpose(1, 2) #1 x (C_in x k x k) x (H_out x W_out) 
            print("transpose, h :", h.shape)
            self.grad_inputX = torch.nn.functional.fold(h, (self.h_in,self.w_in), self.kernel_size,
                                            stride=self.stride,padding = self.padding) #1 x C_in x H_in x W_in
            print("reshape, grad input : ", self.grad_inputX.shape) 

            print("\nGRAD WEIGHT :")
            d = a @ self.unfolded.transpose(1,2) #B x C_out x (C_in x k x k)
            print("multiply a by unfolded transpose, d :", d.shape)
            e = d.sum(dim=0) # C_out x (C_in x k x k)
            print("sum first dim of d, e : ", e.shape)
            self.grad_weight = e.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]) # C_out x C_in x k x k
            print("grad wrt weight (reshape): ", self.grad_weight.shape)

            

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





inp = torch.ones(1,3,6,6)
inp[0,1,:,:] = 2
print("input : ", inp.shape)
w = torch.ones(2,3,2,2)

unfolded = torch.nn.functional.unfold(inp, (2,2))
print("unfolded :", unfolded.shape)

conv_output = unfolded.transpose(1,2).matmul(w.view(w.size(0), -1).t()).transpose(1,2)

print("\nBACKWARD \n")
w = torch.ones(2,3,2,2)
conv = Conv2d(in_channels = 3, out_channels = 2, kernel = (2,2) ,stride=(1,1))
gradwrtoutput = torch.ones(1,2,5,5)
conv.forward(inp)
grad_inputX = conv.backward( gradwrtoutput)


