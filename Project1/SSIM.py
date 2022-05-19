import torch
import matplotlib.pyplot as plt
from utils import *

model = AE_small5()
PATH = "./Data_aug/AE_small5_model.pth"
model.load_state_dict(torch.load(PATH))


class SSIM(nn.Module):
    def __init__(self, window_size: int = 8, max_val: float = 1.):
        super(SSIM, self).__init__()
        self.max_val = max_val
        window = torch.ones((window_size, window_size))
        self.kernel = window.repeat(3, 1, 1, 1) #repeat the window on the 3 channels

        self.C1 = (0.01 * self.max_val) ** 2
        self.C2 = (0.03 * self.max_val) ** 2

    def forward(self, groundtruth: torch.Tensor, noisy_img: torch.Tensor): 
        kernel = self.kernel.to(groundtruth.device).to(noisy_img.dtype)
        mu1: torch.Tensor = F.conv2d(groundtruth, kernel, groups = 3)
        mu2: torch.Tensor = F.conv2d(noisy_img, kernel, groups = 3)

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


######################
# functional interface
######################


def ssim(ground_truth: torch.Tensor, noisy_img: torch.Tensor,
        window_size = 8, max_val = 1.) -> torch.Tensor:
    r"""Function that measures the Structural Similarity (SSIM) index between
    each element in the input `x` and target `y`.

    See :class:`torchgeometry.losses.SSIM` for details.
    """
    return SSIM(window_size = window_size, max_val = max_val)(ground_truth , noisy_img)


SIZE = 1000
BATCH_SIZE = 1
test_set = Dataset(SIZE, train = False)
loader_1 = torch.utils.data.DataLoader(dataset = test_set,
                                    batch_size = BATCH_SIZE,
                                    shuffle = False)
#number of images to plot


PSNR = torch.empty(size = (1, SIZE))
SSIM_results = torch.empty(size = (1, SIZE))
i = 0
for noisy_imgs, ground_truth in loader_1:
    denoised = model(noisy_imgs)
    PSNR[0, i] = psnr(denoised/255, ground_truth/255)
    SSIM_results[0, i] = 1-ssim(ground_truth/255, denoised/255)
    i += 1
print("Model PSNR:", torch.mean(PSNR).item()) #display the mean of PSNR over the test set.
print("Model SSIM:", torch.mean(SSIM_results).item()) #display the mean of PSNR over the test set.

plt.imshow(torch.squeeze(denoised).permute(1, 2, 0).int())
plt.show()