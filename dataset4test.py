import torch
import os
import glob 
import cv2
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.images = []
        image_path = glob.glob(data_path+'/*.png')
        for path in image_path:
            self.images.append(path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (320,320))
        image = self.transform(image)
        return image, image_path.split('/')[-1]
    
    def __len__(self):
        return len(self.images)