import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class InterpretableDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # Get all image and label filenames
        img_filenames = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        label_filenames = sorted(glob.glob(os.path.join(label_dir, '*.txt')))

        # Ensure the number of images matches the number of labels
        assert len(img_filenames) == len(label_filenames), "Mismatch between images and labels count."

        # Match images and labels based on filename
        self.img_label_pairs = []
        for img_path in img_filenames:
            base_name = os.path.basename(img_path).replace('.jpg', '')
            label_path = os.path.join(label_dir, base_name + '.txt')
            if os.path.exists(label_path):
                self.img_label_pairs.append((img_path, label_path))
            else:
                print(f"Warning: No label found for image {img_path}")

    def __len__(self):
        return len(self.img_label_pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.img_label_pairs[idx]

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
