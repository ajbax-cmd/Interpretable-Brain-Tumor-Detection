import os
import yaml
import torch
from torch import nn
from ultralytics import YOLO
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from interpretable_dataset import InterpretableDataset
from faiss_indexer import FaissIndexer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class InterpretableYOLOTest(nn.Module):
    def __init__(self, data_yaml_path, model, batch_size=1, img_size=(640, 640), target_layer_index=0):
        super().__init__()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.img_size=img_size[0]
        self.data_yaml_path = data_yaml_path
        self.data_yaml = self.load_yaml(data_yaml_path)
        self.train_loader, self.val_loader, self.test_loader = self.load_data(batch_size, img_size)

        self.features = []
        self.predictions = []
        self.inference_features = None
        self.training = True
        self.train_image_filenames = []

        # Register hooks with layer index
        all_layers = self.get_all_layers(self.model)
        target_layer = all_layers[target_layer_index]
        target_layer.register_forward_hook(self.get_features())

        #for idx, layer in enumerate(all_layers):
            #print(f"Layer {idx}: {layer}")

        # Forward pass to collect features
        self.extract_training_features()
        print(f"Collected features for {len(self.features)} samples from layer at index '{target_layer_index}'.")

        # Convert features and labels to numpy arrays
        self.features = np.concatenate(self.features, axis=0)
        self.predictions = np.array(self.predictions)
        
        # Flatten features to a 2D array
        self.features = self.features.reshape(self.features.shape[0], -1)

        # Initialize FAISS indexer
        feature_dim = self.features.shape[1]
        self.faiss_indexer = FaissIndexer(feature_dim=feature_dim)

        # Insert features into FAISS vector library
        self.insert_features_into_faiss()

        # Calculate Pearson Correlation
        #self.calculate_pearson_correlation()
        print("complete")

    def get_features(self):
        def hook(model, input, output):
            if self.training:
                self.features.append(output.detach().cpu().numpy())
            self.inference_features = output.detach().cpu().numpy()
        return hook

    def extract_training_features(self):
        self.model.eval()
        with torch.no_grad():
            for sample_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                cropped_images = self.crop_to_bounding_box(images, labels)
                cropped_images = cropped_images.to(self.device)
                img_filename, label_filename = self.train_loader.dataset.img_label_pairs[sample_idx]
                outputs = self.model(cropped_images)  # Forward pass with cropped images
                predictions = outputs[0] if isinstance(outputs, tuple) else outputs
                self.predictions.extend(predictions.cpu().tolist())
                self.train_image_filenames.append(os.path.basename(img_filename))

    def crop_to_bounding_box(self, images, bboxes):
        """Crop the image to the bounding box, blacking out everything outside."""
        cropped_images = []
        for img, bbox in zip(images, bboxes):
            if len(bbox) == 5:
                _, x_center, y_center, width, height = [float(coord) for coord in bbox]
            elif len(bbox) == 4:
                x_center, y_center, width, height = [float(coord) for coord in bbox]
            else:
                raise ValueError(f"Unexpected number of elements in bbox: {len(bbox)}")

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

            # Ensure coordinates are within the image bounds
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > img_width: x2 = img_width
            if y2 > img_height: y2 = img_height

            # Create a mask for the bounding box
            mask = torch.zeros_like(img)
            mask[:, y1:y2, x1:x2] = 1

            # Apply the mask to black out everything outside the bounding box
            img_cropped = img * mask

            cropped_images.append(img_cropped)

        return torch.stack(cropped_images)



    def single_image_inference(self, image_path, k=5):
        self.training = False
        # Load and transform the image
        image = Image.open(image_path).convert('L')  # Ensure the image is grayscale

        transform = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize to YOLOv8 input size
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize for grayscale images
        ])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension 

        # Move the image to the specified device
        image = image.to(self.device, non_blocking=True)

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image)  # Forward pass

            # Extract the output with highest confidence score
            prediction = self.process_output(outputs)  # Assuming outputs[0] contains the predictions
            if prediction.size == 0:
                return {
                    'model_prediction': None,
                    'confidence_score': None,
                    'bounding_box': None,
                    'nearest_neighbors': [],
                    'distances': []
                }

            # YOLO output format: [x_center, y_center, width, height, confidence]
            best_confidence_score = prediction[4]
            best_bbox = prediction[:4]
            best_bbox = best_bbox / self.img_size  # Normalize

            # Crop the image based on the bounding box coordinates
            cropped_image = self.crop_to_bounding_box(image, [best_bbox])

            # Convert the cropped image tensor to a numpy array for plotting
            cropped_image_np = cropped_image[0].cpu().numpy().transpose(1, 2, 0)

            # Denormalize the image back to 0-1 range for visualization
            cropped_image_np = cropped_image_np * 0.5+0.5
            cropped_image_np = np.clip(cropped_image_np, 0, 1)

            # Display the cropped image
            plt.figure(figsize=(8, 8))
            plt.title("Inference Image Cropped")
            plt.imshow(cropped_image_np, cmap='gray')  
            plt.axis('off')
            plt.show()

            cropped_image = cropped_image.to(self.device)

            # Forward pass with the cropped image to get features from the target layer
            with torch.no_grad():
               _ = self.model(cropped_image)  # Forward pass with cropped image
            print("inference features size: ",len(self.inference_features))
            # Get features from the target layer
            inference_features = self.inference_features  # Assuming the feature extraction is set up correctly
            inference_features = inference_features.reshape(1, -1)  # Reshape if necessary

            # Perform k-nearest neighbor search in FAISS
            distances, indices = self.faiss_indexer.search_k_nearest_neighbors(inference_features, k=k)

            # Map FAISS indices to image files
            nearest_neighbor_filenames = [self.train_image_filenames[idx] for idx in indices[0]]

            # Construct the result
            result = {
                'model_prediction': 0,  # Assuming class index 0
                'confidence_score': best_confidence_score,
                'bounding_box': best_bbox,
                'nearest_neighbors': nearest_neighbor_filenames,
                'distances': distances
            }

        return result

    def iterate_directory_inference(self, directory_path, k=5):
        results = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(directory_path, filename)
                result = self.single_image_inference(image_path, k)
                result['filename'] = filename  # Add the image path to the result for reference
                results.append(result)
        return results
    
    def process_output(self, output):
        # Assuming output tensor a shape of [1, 5, 8400]
        output = output[0]
        highest_confidence = 0
        hc_index = 0
        result = []
        for i in range(0,8400,1):
            if(output[0][4][i] > highest_confidence):
                highest_confidence = output[0][4][i]
                hc_index = i
        for i in range(0, 5, 1):
            result.append(output[0][i][hc_index].item())
        return np.array(result)

    def insert_features_into_faiss(self):
        self.faiss_indexer.add_features(self.features)

    def load_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def load_data(self, batch_size=1, img_size=(640, 640)):
        train_img_dir = self.data_yaml['train']
        val_img_dir = self.data_yaml['val']
        test_img_dir = self.data_yaml['test']

        transform = transforms.Compose([
            transforms.Resize((640, 640)),  
            transforms.Grayscale(num_output_channels=3),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        train_dataset = InterpretableDataset(
            img_dir=train_img_dir,
            label_dir=os.path.join(train_img_dir, '../labels'),
            transform=transform
        )

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
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader

    def get_all_layers(self, model):
        layers = []
        for name, module in model.named_modules():
            layers.append(module)
        return layers
    
    def calculate_pearson_correlation(self):
        # Ensure features and predictions are 2D arrays
        assert self.features.ndim == 2, "Features should be a 2D array"
        assert self.predictions.ndim == 3, "Predictions should be a 3D array (samples, attributes, bounding boxes)"

        # Select a relevant dimension from predictions, e.g., confidence scores (5th attribute in this case)
        selected_predictions = self.predictions[:, 4, :]  # Shape: (500, 8400)
        selected_predictions = selected_predictions.mean(axis=1)  # Average over bounding boxes, resulting in shape: (500,)
        
        correlations = []
        for i in range(self.features.shape[1]):  # Iterate over each feature dimension
            feature_vector = self.features[:, i]  # Shape: (500,)
            corr, _ = pearsonr(feature_vector, selected_predictions)
            correlations.append(corr)
        
        # Calculate the average of the absolute values of the Pearson correlations
        overall_correlation_score = np.mean(np.abs(correlations))
        
        #print(f"Pearson correlations for the target layer: {correlations}")
        print(f"Pearson correlation score for the target layer: {overall_correlation_score}")

        return overall_correlation_score
    


def main():
    data = '/home/alan/Documents/YOLOV8_interpretable/Dataset_1/brain-tumor-detection-dataset/Br35H-Mask-RCNN/data.yaml'
    weights = '/home/alan/Documents/YOLOV8_interpretable/YOLOv8_weights/yolov8m_br35h.pt'
    model = YOLO(weights)

    model = InterpretableYOLOTest(data, model, target_layer_index=76)
    
    
    # Perform inference on all images in a directory
    directory_path = '/home/alan/Documents/YOLOV8_interpretable/Dataset_1/brain-tumor-detection-dataset/Br35H-Mask-RCNN/test/images'
    #results = model.iterate_directory_inference(directory_path, k=3)
    #for result in results:
        #print(result)

#if __name__ == "__main__":
    #main()
