import os
import yaml
import torch
from ultralytics import YOLO
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import numpy as np
from sklearn. decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from interpretable_dataset import InterpretableYOLODataset

# wrapper class for pretrained YOLO model, takes training data and weights as args
class InterpretableYOLOTest(YOLO):
    def __init__(self, data_yaml_path, weights='yolov8m_br35h.pt', target_layer_index=21):
        # Load the pretrained YOLOv8 model to YOLO class constructor, assigns to self.model which is accessible by inheritance
        super().__init__(weights)
        self.model.to(self.device)
        self.data_yaml_path = data_yaml_path
        self.data_yaml = self.load_yaml(data_yaml_path)
        self.train_loader, self.val_loader, self.test_loader = self.load_data()

        num_layers = len(list(self.model.named_modules()))
        self.features = []

        # Register hooks with layer index
        target_layer = list(self.model.model.children())[target_layer_index]
        target_layer.register_forward_hook(self.get_features())


        # Forward pass to collect features
        self.extract_features()
        
        print(f"Collected features for {len(self.features)} samples from layer at index '{target_layer_index}'.")


    def get_features(self):
        def hook(model, input, output):
            self.features.append(output.detach().cpu().numpy())
        return hook

    
    def extract_features(self):
        self.model.eval()
        with torch.no_grad():
            for sample_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                cropped_images = self.crop_to_bounding_box(images, labels)
                cropped_images = cropped_images.to(self.device)
                outputs = self.model(cropped_images)  # Forward pass with cropped image

    
    def crop_to_bounding_box(self, images, bboxes):
        """Crop the image to the bounding box, blacking out everything outside."""
        cropped_images = []
        for img, bbox in zip(images, bboxes):
            class_id, x_center, y_center, width, height = bbox

            # Convert from (x_center, y_center, width, height) to (x1, y1, x2, y2)
            img_width, img_height = img.size(-1), img.size(-2)
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            img_pil = transforms.ToPILImage()(img).convert("RGB")
            draw = ImageDraw.Draw(img_pil)

            # Create a black image
            black_img = Image.new("RGB", img_pil.size, (0, 0, 0))

            # Paste the black image on the original image outside the bounding box
            draw.rectangle([0, 0, img_pil.width, y1], fill=(0, 0, 0))
            draw.rectangle([0, y2, img_pil.width, img_pil.height], fill=(0, 0, 0))
            draw.rectangle([0, y1, x1, y2], fill=(0, 0, 0))
            draw.rectangle([x2, y1, img_pil.width, y2], fill=(0, 0, 0))

            cropped_img = transforms.ToTensor()(img_pil)
            cropped_images.append(cropped_img)

        return torch.stack(cropped_images)
    
    def reduce_dimensionality(self, features):
        pca = PCA(n_components=50)  
        flat_features = [feat.flatten() for feat in features]
        reduced_features = pca.fit_transform(flat_features)
        return reduced_features

    def cluster_features(self, features, n_clusters=50):
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(features)
        return kmeans, clusters

    def nearest_neighbor_search(self, features, kmeans):
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(features)
        distances, indices = nn.kneighbors(kmeans.cluster_centers_)
        return distances, indices

    def evaluate_clustering(self, features, clusters):
        score = silhouette_score(features, clusters)
        return score
    
    def load_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, batch_size=1, img_size=(640, 640)):
        train_img_dir = self.data_yaml['train']
        val_img_dir = self.data_yaml['val']
        test_img_dir = self.data_yaml['test']

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = InterpretableYOLODataset(
            img_dir=train_img_dir,
            label_dir=os.path.join(train_img_dir, '../labels'),
            transform=transform
        )

        val_dataset = InterpretableYOLODataset(
            img_dir=val_img_dir,
            label_dir=os.path.join(val_img_dir, '../labels'),
            transform=transform
        )

        test_dataset = InterpretableYOLODataset(
            img_dir=test_img_dir,
            label_dir=os.path.join(test_img_dir, '../labels'),
            transform=transform
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader


def main():
    data = '/home/alan/Documents/YOLOV8_interpretable/Dataset_1/brain-tumor-detection-dataset/Br35H-Mask-RCNN/data.yaml'
    weights = '/home/alan/Documents/YOLOV8_interpretable/yolov8m_br35h.pt'
    model = InterpretableYOLOTest(data, weights, target_layer_index=21)

    # After feature extraction
    reduced_features = model.reduce_dimensionality(model.features)
    kmeans, clusters = model.cluster_features(reduced_features)
    nn_results = model.nearest_neighbor_search(reduced_features, kmeans)
    clustering_score = model.evaluate_clustering(reduced_features, clusters)

    # Print evaluation results
    print("Clustering score:", clustering_score)

#if __name__ == "__main__":
    #main()