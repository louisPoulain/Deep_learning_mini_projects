import torch
import matplotlib.pyplot as plt
from utils import *

def compare_models_images(PATHS, models, noisy_imgs, ground_truth):
    "Given a list of PATH that lead to a model in the corresponding list models, display the noisy image, the ground truth and the denoised images."
    #number of images to plot
    N = len(PATHS) + 2 
    plt.subplot(1, N, 1)
    plt.imshow(torch.squeeze(noisy_imgs).permute(1, 2, 0).int()) #int since the data has been changed to float for the NN.
    plt.title("Noisy imgs")
    plt.subplot(1, N, 2)
    plt.imshow(torch.squeeze(ground_truth).permute(1, 2, 0).int())
    plt.title("Groundtruth")
    for j, PATH in enumerate(PATHS):
        model = models[j]
        model.load_state_dict(torch.load(PATH))
        denoised = model(noisy_imgs)
        plt.subplot(1, N, j+3)
        plt.imshow(torch.squeeze(denoised).permute(1, 2, 0).int())
        plt.title("model " + str(j))
    plt.show()


PATHS= ["./Noise2Noise/project1_1.pth", "./Noise2Noise/project1_2.pth"]
models = [Noise2Noise(), Noise2Noise_2()]
"""BATCH_SIZE = 1
test_set = Dataset(SIZE, train = False)
loader_1 = torch.utils.data.DataLoader(dataset = test_set,
                                     batch_size = BATCH_SIZE,
                                     shuffle = True)

noisy_imgs, ground_truth = next(iter(loader_1))

compare_models_images(PATHS, models, noisy_imgs, ground_truth)"""

compare_models(PATHS, models)



