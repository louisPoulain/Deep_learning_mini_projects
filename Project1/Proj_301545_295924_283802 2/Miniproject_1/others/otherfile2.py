import torch 
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ~~~~~~ Wavelet transforms ~~~~~~~~

divisor = torch.sqrt(torch.tensor(2))

analysis_lp = (torch.tensor([1, 1, 0], dtype = torch.float32).reshape((1, 1, 1, -1)) / divisor).to(device)
analysis_hp = (torch.tensor([-1, 1, 0], dtype = torch.float32).reshape((1, 1, 1, -1)) / divisor).to(device)
synthesis_lp = (torch.tensor([0, 1, 1], dtype = torch.float32).reshape((1, 1, 1, -1)) / divisor).to(device)
synthesis_hp = (torch.tensor([0, 1, -1], dtype = torch.float32).reshape((1, 1, 1, -1)) / divisor).to(device)

def analysis(img, lp = analysis_lp, hp = analysis_hp, n = 1):
  # return the img transformed as a list of all its wavelet coefficients
    # Get dimensions of the original image
    ny, nx = img.shape[2], img.shape[3]
    # Declare output
    output = torch.clone(img).to(device)
    for i in range(n):
        # Get LL (will select the whole image the first time, and 
        # the LL part of the previous iteration the next times)
        sub = output[:, :, 0:ny, 0:nx]
        
        # Store your result in the variable sub, which is incorporated
        # into output at the end of each iteration
        
        ## Horizontal transform
        # Apply filters to create the two horizontal components
        pad = (1, 1, 1, 1)
        sub_lp = F.conv2d(F.pad(input = sub, pad = pad, mode = 'circular'), weight = lp, stride = (1, 1))
        sub_hp = F.conv2d(F.pad(input = sub, pad = pad, mode = 'circular'), weight = hp, stride = (1, 1))
        # Downsample
        sub_lp = sub_lp[:, :, :, ::2]
        sub_hp = sub_hp[: ,: , :, ::2]
        # Concatenate results to construct horizontal wavelet structure
        sub = torch.cat((sub_lp, sub_hp), axis = -1)
        sub = sub[:, :, 1:-1, :] # remove the artificial padding ...
        
        
        ## Vertical transform
        # Repeat all the steps above for the vertical transform
        sub_lp = F.conv2d(F.pad(input = sub, pad = pad, mode = 'circular'), weight = lp.reshape(1, 1, -1, 1), stride = (1, 1))
        sub_hp = F.conv2d(F.pad(input = sub, pad = pad, mode = 'circular'), weight = hp.reshape(1, 1, -1, 1), stride = (1, 1)) 
        
        sub_lp = sub_lp[:, :, ::2, :]
        sub_hp = sub_hp[:, :, ::2, :]
        
        sub = torch.cat((sub_lp, sub_hp), axis = -2)
        sub = sub[:, :, :, 1:-1] # remove the artificial padding ...
        # Insert sub to replace the old LL coefficients
        output[:, :, 0:ny, 0:nx] = sub
        # Adjust dimensions to represent the new LL coefficient
        nx = nx//2
        ny = ny//2
        #print(output)
    list_of_coeff = []
    ny, nx = img.shape[2], img.shape[3]
    for i in range(1, n+1):
        div = 2 ** i
        lh = output[:, :, ny // div : 2 * ny // div, 0 : nx // div]
        hh = output[:, :, ny // div : 2 * ny // div, nx // div : 2 * nx // div]
        hl = output[:, :, 0 : ny // div, nx // div : 2 * nx // div]
        list_of_coeff.append(lh)
        list_of_coeff.append(hh)
        list_of_coeff.append(hl)
        ll = output[:, :, 0 : ny // div, 0 : nx // div]
        list_of_coeff.append(ll)
    return list_of_coeff, output

def wavelet_reconstitution(coeff, size = 32):
    # from the wavelet coeff, reconstrutct the wavelet transform of an img
    output = torch.zeros((coeff[0].shape[0], coeff[0].shape[1], size, size))
    ll = coeff[-1]
    output[:, :, 0 : ll.shape[2], 0 : ll.shape[3]] = ll
    ny, nx = size, size
    for i in range(0, len(coeff)-1, 4):
        div = 2 ** (i // 4 + 1)
        lh = coeff[i]
        hh = coeff[(i+1)]
        hl = coeff[(i+2)]
        output[:, :, ny // div : 2 * ny // div, 0 : nx // div] = lh
        output[:, :, ny // div : 2 * ny // div, nx // div : 2 * nx // div] = hh
        output[:, :, 0 : ny // div, nx // div : 2 * nx // div] = hl
    return output

def synthesis(coeffs, lp = synthesis_lp, hp = synthesis_hp, n = 1):
    def My_upsample1d(a, factor = 2, axis = -2):
            x, y = a.shape[2], a.shape[3]
            if axis == -2:
                b = torch.zeros((a.shape[0], a.shape[1], int(factor * x), y))
                b[:, :, ::int(factor), :] = a
            else:
                b = torch.zeros((a.shape[0], a.shape[1], x, int(factor * y)))
                b[:, :, :, ::int(factor)] = a
            return b.to(device)
    #print(My_upsample1d(coeffs))
    # Get dimensions of the last wavelet transform
    div = 2**(n-1)
    ny, nx = torch.div(torch.tensor([coeffs.shape[2], coeffs.shape[3]]), div, rounding_mode = 'floor')
    # Declare output
    output = torch.clone(coeffs).to(device)
    for i in range(n):
        # Extract wavelet coefficients (the smallest transform at first, then 
        # incorporating more and more coefficients)
        wt_iter = output[:, :, 0:ny, 0:nx]
        #print(wt_iter.shape)
        
        ## Horizontal inverse transfrom
        # Split wt_iter into its two horizontal components
        wt_horiz_high = wt_iter[:, :, :ny // 2, :]
        wt_horiz_low = wt_iter[:, :, ny // 2:, :]
        #print(wt_horiz_high.shape, wt_horiz_low.shape)
        
        # Upsample each component
        
        wt_horiz_high = My_upsample1d(wt_horiz_high, axis = -2)
        wt_horiz_low = My_upsample1d(wt_horiz_low, axis = -2)
        #print(wt_horiz_high.shape, wt_horiz_low.shape)
        
        # Filter each component with the corresponding filter
        pad = (1, 1, 1, 1)
        wt_filt_high = F.conv2d(F.pad(input = wt_horiz_high, pad = pad, mode = 'circular'), weight = lp.reshape(1, 1, -1, 1), stride = (1, 1))
        wt_filt_low = F.conv2d(F.pad(input = wt_horiz_low, pad = pad, mode = 'circular'), weight = hp.reshape(1, 1, -1, 1), stride = (1, 1))
        #print(wt_filt_high.shape, wt_filt_low.shape)
        # Combine the results into one component
        wt_iter = wt_filt_high[:, :, :, 1:-1] + wt_filt_low[:, :, :, 1:-1]
        
        ## Vertical inverse transform
        # Repeat all the steps above for the vertical components (on the output
        # of the inverse horizontal transform)
        #print(wt_iter.shape)
        wt_vert_left = wt_iter[:, :, :, 0:nx // 2]
        wt_vert_right = wt_iter[:, :, :, nx // 2:]
        
        #print(wt_vert_left.shape, wt_vert_right.shape)
        
        wt_vert_left = My_upsample1d(wt_vert_left, axis = -1)
        wt_vert_right = My_upsample1d(wt_vert_right, axis = -1)
        #print(wt_vert_left.shape, wt_vert_right.shape)
        pad = (1, 1, 1, 1)
        wt_filt_left = F.conv2d(F.pad(input = wt_vert_left, pad = pad, mode = 'circular'), weight = lp, stride = (1, 1))
        wt_filt_right = F.conv2d(F.pad(input = wt_vert_right, pad = pad, mode = 'circular'), weight = hp, stride = (1, 1))
        
        wt_iter = wt_filt_left[:, :, 1:-1, :] + wt_filt_right[:, :, 1:-1, :]
        #print(wt_iter.shape)
        
        # Replace synthesis of corresponding iteration
        output[:, :, 0:ny, 0:nx] = wt_iter
        # Update dimensions for next scale
        nx = nx * 2
        ny = ny * 2  
    return output


# ~~~~~~~ image quality measure ~~~~~~~~

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