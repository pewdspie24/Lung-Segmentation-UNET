import torch
import os
import glob
import cv2
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.images = []
        self.masks = []

        image_path = glob.glob(data_path+'/train/*.png')
        mask_path = glob.glob(data_path+'/masks/*.png')
        
        for path in image_path:
            self.images.append(path)
        for path in mask_path:
            self.masks.append(path)

        random_apply = [transforms.RandomAffine(degrees=(-45,+45), scale=(1,2))]
        self.transformI = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply(random_apply, p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transformM = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply(random_apply, p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0],[0.5]) 
        ])

    def __getitem__(self, idx):
        torch.manual_seed(24)
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = cv2.imread(image_path)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (320,320))
        mask = cv2.resize(mask, (320,320))
        image = self.transformI(image)
        mask = self.transformM(mask)
        mask = mask.numpy()
        for px in mask:
            px[px>0] = 1
        return image, torch.from_numpy(mask)

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    dataset = CustomDataset("./lung-seg") # test thu voi tap train
    # khong anh huong khi import duoi dang 1 lib
    print(dataset.__getitem__(1)[1])
