import os
import numpy as np
import yaml
import torch
from torch import nn
from PIL import Image
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from interpretable_dataset import InterpretableDataset
from feature_clusterer import FeatureClusterer
from faiss_indexer import FaissIndexer

# Class to add interpretability to pytorch models
class InterpretableTest(nn.Module):
    def __init__(self, data_yaml_path, model, weights_path=None,  batch_size=1, img_size=(224, 224)):
        super().__init__()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.img_size = img_size
        self.data_yaml_path = data_yaml_path
        self.data_yaml = self.load_yaml(data_yaml_path)
        self.train_loader, self.val_loader, self.test_loader = self.load_data(batch_size)


        self.features = []
        self.labels = []
        self.predictions = []
        self.inference_features = None
        self.training = True  # Flag to control training/inference
        self.train_image_filenames = []  # List to store training image paths

        # Modify the fully connected layer to match the number of classes dataset
        num_ftrs = self.model.fc.in_features
        num_classes = self.data_yaml['nc']
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model.fc.to(self.device)  # Move the new fully connected layer to the correct device

        if weights_path:
            self.load_weights(weights_path)

        # Register hooks with layer index
        all_layers = self.get_all_layers(self.model)
        print(f"Total number of layers: {len(all_layers)}")
        self.model.avgpool.register_forward_hook(self.get_features())

        # Forward pass to collect features
        self.extract_training_features()
        print(f"Collected features for {len(self.features)} samples from layer avgpool.")

        # Convert features and labels to numpy arrays
        self.features = np.concatenate(self.features, axis=0)
        self.labels = np.array(self.labels).reshape(-1)
        self.predictions = np.array(self.predictions)
        
        # Flatten features to a 2D array
        self.features = self.features.reshape(self.features.shape[0], -1)

        # Initialize FAISS indexer
        feature_dim = self.features.shape[1]
        #print(feature_dim)
        self.faiss_indexer = FaissIndexer(feature_dim=feature_dim)

        # Insert features into FAISS vector library
        self.insert_features_into_faiss()


    def load_weights(self, weights_path):
        # Load weights into the model if provided
        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)

    def get_features(self):
        def hook(model, input, output):
            if(self.training):
                self.features.append(output.detach().cpu().numpy())
            self.inference_features = output.detach().cpu().numpy() 
        return hook

    def extract_training_features(self):
        self.model.eval()
        with torch.no_grad():
            for sample_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                outputs = self.model(images)  # Forward pass with images
                self.labels.extend(labels) # Collect labels for class-specific clustering
                self.predictions.extend(outputs.cpu().tolist()) 
                batch_image_filenames = [os.path.basename(path) for path in self.train_loader.dataset.img_paths[sample_idx * self.train_loader.batch_size : (sample_idx + 1) * self.train_loader.batch_size]]
                self.train_image_filenames.extend(batch_image_filenames)




    def single_image_inference(self, image_path, k=5):
        """
        Perform inference on a single image and return the prediction and nearest neighbors.
        
        Parameters:
        - image_path (str): Path to the input image.
        - k (int): Number of nearest neighbors to retrieve from the FAISS index.
        
        Returns:
        - dict: Dictionary containing the model prediction, nearest neighbors, and distances.
        """
        self.training=False
        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Move the image to the specified device
        image = image.to(self.device)

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image)  # Forward pass
            softmax = nn.Softmax(dim=1)
            outputs = softmax(outputs)  # Apply softmax to get probabilities
            predicted_class = np.argmax(outputs.cpu().numpy(), axis=1)

        # Get features from the target layer
        inference_features = self.inference_features  # Assuming the feature extraction is set up correctly
        inference_features = inference_features.reshape(1, -1)  # Reshape if necessary

        # Perform k-nearest neighbor search in FAISS
        distances, indices = self.faiss_indexer.search_k_nearest_neighbors(inference_features, k=k)
        
        # Map FAISS indices to image files
        nearest_neighbor_filenames = [self.train_image_filenames[idx] for idx in indices[0]]
        
        
        # Construct the result
        result = {
            'model_prediction': predicted_class,
            'nearest_neighbors': nearest_neighbor_filenames,
            'distances': distances
        }
        
        return result

    def iterate_directory_inference(self, directory_path, k=5):
        """
        Iterate through all images in a directory and perform inference on each.
        
        Parameters:
        - directory_path (str): Path to the directory containing images.
        - k (int): Number of nearest neighbors to retrieve from the FAISS index.
        
        Returns:
        - list: List of dictionaries containing the model predictions, nearest neighbors, and distances for each image.
        """
        results = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(directory_path, filename)
                result = self.single_image_inference(image_path, k)
                result['filename'] = filename  # Add the image path to the result for reference
                results.append(result)
        return results
        
    def insert_features_into_faiss(self):
        self.faiss_indexer.add_features(self.features)
    
    def load_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def load_data(self, batch_size=1):
        train_img_dir = self.data_yaml['train']
        val_img_dir = self.data_yaml.get('val')  
        test_img_dir = self.data_yaml['test']

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = InterpretableDataset(
            img_dir=train_img_dir,
            label_dir=os.path.join(train_img_dir, '../labels'),
            transform=transform
        )

        val_dataset = None
        if val_img_dir is not None:
            val_dataset = InterpretableDataset(
                img_dir=val_img_dir,
                label_dir=os.path.join(val_img_dir, '../labels'),
                transform=transform
            )

        test_dataset = InterpretableDataset(
            img_dir=test_img_dir,
            label_dir=os.path.join(test_img_dir, '../labels'),
            transform=transform
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader

    def get_all_layers(self, model):
        layers = []
        for name, module in model.named_modules():
            layers.append(module)
        return layers

def main():
    data = '/home/alan/Documents/YOLOV8_interpretable/Dataset_2/data.yaml'
    weights = '/home/alan/Documents/YOLOV8_interpretable/ResNet50_weights/Brain_Tumor_MRI.pth'

    # Initialize a model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # Last Conv layer is 145
    cnn_interpreter = InterpretableTest(data, model, weights, target_layer_index=145)
    # Perform inference on all images in a directory
    directory_path = '/home/alan/Documents/YOLOV8_interpretable/Dataset_2/testing/images'
    results = cnn_interpreter.iterate_directory_inference(directory_path, k=3)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()