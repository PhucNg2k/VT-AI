import numpy as np
import open3d as o3d
import cv2 as cv
import matplotlib.pyplot as plt 
import pandas as pd 

from config import *
from file_utils import get_file_list
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.data_csv = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_csv)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.data_csv.iloc[idx, 0])
        image = cv.imread(image_path) # H,W,3
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # H,W
        image = torch.from_numpy(image).permute(1,0) # W,H
        
        labels = self.data_csv.iloc[idx, 1:].values.astype(np.float32)
        labels = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)
        
        return image, labels


if __name__ == "__main__":
    train_csv = r"D:\ViettelAI\Public data task 3\Public data\Public data train\Public_train.csv"
    training_data = CustomDataset(train_csv, RGB_TRAIN)

    train_loader = DataLoader(training_data, shuffle=True)

    image, labels  = next(iter(train_loader))

    print(image.shape)
    print(labels)