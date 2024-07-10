import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class InterpretableDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
        self.label_paths = glob.glob(os.path.join(label_dir, '*.txt'))

        # Ensure the number of images matches the number of labels
        #assert len(self.img_paths) == len(self.label_paths), "Mismatch between images and labels count."

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(img_path).convert("RGB")
        labels = []

        with open(label_path, 'r') as f:
            for line in f:  
                try:
                    temp = list(map(float, line.strip().split()))  # convert labels to floats
                    labels.append(temp)
                except ValueError as e:
                    print(f"Skipping line due to error: {e}")  # Debug statement
                    continue

        if self.transform:
            image = self.transform(image)

        return image, labels
