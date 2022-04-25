import torch
import matplotlib.pyplot as plt
from utils import *

model = Noise2Noise()
PATH = "./Noise2Noise/project1_1.pth"
model.load_state_dict(torch.load(PATH))

SIZE = 500
BATCH_SIZE = 1
test_set = Dataset(SIZE, train = False)

loader_1 = torch.utils.data.DataLoader(dataset = test_set,
                                     batch_size = BATCH_SIZE,
                                     shuffle = True)

PSNR = []
for noisy_imgs, ground_truth in loader_1:
    denoised = model(noisy_imgs)
    PSNR.append(psnr(denoised, ground_truth).detach().numpy())
    
plt.subplot(2, 1, 1)
print(denoised.shape)
print(ground_truth.shape)
plt.imshow(torch.squeeze(denoised).permute(1, 2, 0).detach().numpy())
plt.subplot(2, 1, 2)
plt.imshow(torch.squeeze(ground_truth).permute(1, 2, 0).detach().numpy())
plt.show()

plt.style.use('fivethirtyeight')
plt.ylabel('PSNR')
plt.plot(PSNR)
plt.show()


