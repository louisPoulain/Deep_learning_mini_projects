import torch
import matplotlib.pyplot as plt
from utils import *

model = Noise2Noise_3()
model_aug = Noise2Noise_3()

PATH = "./Data_aug/1_model.pth"
model.load_state_dict(torch.load(PATH))

PATH_AUG = "./Data_aug/1_model_aug.pth"
model_aug.load_state_dict(torch.load(PATH_AUG))

PATHS = [PATH, PATH_AUG]
models = [model, model_aug]

compare_models(PATHS, models)