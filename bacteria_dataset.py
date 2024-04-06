import os
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class BacteriaDataset(Dataset):
    def __init__(self, transform=None, data_dir='train', label_path='train.csv'):
        self.transform = transform
        self.data_dir = data_dir
        self.label_path = label_path
        # self.data_labels = os.path.join(self.data_dir, "/datasets/train.txt")

        self.df = pd.read_csv(label_path)
        labels = self.df['image_name'].to_list()
        coords = self.df[['x', 'y']].to_numpy()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
                                             transforms.Normalize((0.5,), (0.5,))])

        # val1 = self.transform(torch.from_numpy(Image.open(os.path.join(self.data_dir, labels[0]))).to(DEVICE))
        self.data = [(self.transform(Image.open(os.path.join(self.data_dir, labels[i]))).repeat(3, 1, 1),
                                torch.from_numpy(coords[i]).to(torch.float32)) for i in range(len(labels))]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image and label from the file
        return self.data[idx]