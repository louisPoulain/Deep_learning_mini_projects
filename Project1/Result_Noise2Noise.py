import torch
import matplotlib.pyplot as plt
from utils import *

model = Noise2Noise_2()
PATH = "./Noise2Noise/project1_2.pth"
model.load_state_dict(torch.load(PATH))

SIZE = 1000
BATCH_SIZE = 1
test_set = Dataset(SIZE, train = False)

"""plt.subplot(2, 1, 1)
plt.imshow(test_set.x[-1].permute(1, 2, 0).int())
plt.subplot(2, 1, 2)
plt.imshow(test_set.y[-1].permute(1, 2, 0).int())
plt.show()"""

loader_1 = torch.utils.data.DataLoader(dataset = test_set,
                                     batch_size = BATCH_SIZE,
                                     shuffle = False)

PSNR = torch.empty(size = (1, SIZE))
i = 0
for noisy_imgs, ground_truth in loader_1:
    denoised = model(noisy_imgs)
    PSNR[0, i] = psnr(denoised/255, ground_truth/255)
    i += 1
    
plot_3imgs(denoised, ground_truth, noisy_imgs)

print("PSNR mean : ", torch.mean(PSNR).item(), " dB")
plt.style.use('fivethirtyeight')
plt.ylabel('PSNR')
plt.plot(PSNR[0,:].detach().numpy())
plt.show()


