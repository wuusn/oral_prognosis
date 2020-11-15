import torch
import glob
import random
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import numpy as np
from multiprocessing import Pool
import girder_client
import numpy as np
from skimage.transform import resize

class TMADataset(Dataset):
    def __init__(self, names, base_dir, phase, augmentations):
        self.names = names
        
    def __len__(self):
        return

    def __getitem__(self, i):
        for augmentation in augmentations:
            faugmentation = getattr(self, augmentation)
            faugmentation()
        return
