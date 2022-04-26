# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 19:07:51 2022

@author: louis
"""

import torch
import matplotlib.pyplot as plt
from helpers import AE, plot_3imgs, Dataset, psnr

model = AE()
time = '04_25_19h_35m_11s' # to be filled according to the job we want to load
PATH = "./test1/project1_1_" + time + ".pth"
model.load_state_dict(torch.load(PATH))

SIZE = 500
BATCH_SIZE = 1
test_set = Dataset(SIZE, train = False)

"""plt.subplot(2, 1, 1)
plt.imshow(test_set.x[-1].permute(1, 2, 0).int())
plt.subplot(2, 1, 2)
plt.imshow(test_set.y[-1].permute(1, 2, 0).int())
plt.show()"""

loader_1 = torch.utils.data.DataLoader(dataset = test_set,
                                     batch_size = BATCH_SIZE,
                                     shuffle = True)

PSNR = torch.empty(size = (1, SIZE))
i = 0
for noisy_imgs, ground_truth in loader_1:
    denoised = model(noisy_imgs)
    PSNR[0, i] = psnr(denoised / 255, ground_truth / 255)
    i += 1
    
plot_3imgs(denoised, ground_truth, noisy_imgs)

print("PSNR mean : ", torch.mean(PSNR).item(), " dB")
plt.style.use('fivethirtyeight')
plt.ylabel('PSNR')
plt.plot(PSNR[0,:].detach().numpy())
plt.show()
