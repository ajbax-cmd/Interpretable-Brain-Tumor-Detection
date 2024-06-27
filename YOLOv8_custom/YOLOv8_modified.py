import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import SPPF, C2f
from ultralytics.nn.modules.head import Detect



class YOLOv8mModified(nn.Module):
    def __init__(self, nc=80):
        super(YOLOv8mModified, self).__init__()
        width, depth = 0.75, 0.67
        self.backbone = nn.Sequential(
            Conv(3, int(64 * width), 3, 2),  # P1/2
            Conv(int(64 * width), int(128 * width), 3, 2),  # P2/4
            C2f(int(128 * width), int(128 * width), int(3 * depth)),
            Conv(int(128 * width), int(256 * width), 3, 2),  # P3/8
            C2f(int(256 * width), int(256 * width), int(6 * depth)),
            Conv(int(256 * width), int(512 * width), 3, 2),  # P4/16
            C2f(int(512 * width), int(512 * width), int(6 * depth)),
            Conv(int(512 * width), int(1024 * width), 3, 2),  # P5/32
            C2f(int(1024 * width), int(1024 * width), int(3 * depth)),
            SPPF(int(1024 * width), int(1024 * width), 5)
        )
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            Concat(),
            C2f(int(1024 * width) + int(512 * width), int(512 * width), int(3 * depth)),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Concat(),
            C2f(int(512 * width) + int(256 * width), int(256 * width), int(3 * depth)),
            Conv(int(256 * width), int(256 * width), 3, 2),
            Concat(),
            C2f(int(512 * width), int(512 * width), int(3 * depth)),
            Conv(int(512 * width), int(512 * width), 3, 2),
            Concat(),
            C2f(int(1024 * width), int(1024 * width), int(3 * depth)),
            Detect(nc, ch=(256, 512, 1024))
        )

    def forward(self, x):
        features = []
        # Forward pass through the backbone
        for layer in self.backbone:
            x = layer(x)
            features.append(x)
        # Forward pass through the head
        for layer in self.head:
            if isinstance(layer, Concat):
                x = torch.cat([x, features[-1]], dim=1)  
            x = layer(x)
            features.append(x)
        return features
    
    # Function to flatten the spatial dimensions and channels
    def flatten_features(feature_tensor):
        batch_size, num_channels, height, width = feature_tensor.shape
        # Reshape each feature map to collapse spatial dimensions and channels into a single vector per sample
        flattened = feature_tensor.view(batch_size, -1)  # flatten all dimensions except batch dimension, result is bactch_size # of 1d arrays
        return flattened
    
    def flatten_1d(features):
        # Apply the function to each tensor in the list
        flattened_features = [flatten_features(feature) for feature in features]
        # Now, extract each feature from the batch and flatten it
        all_flattened_features = []
        for feature in flattened_features:
            for i in range(feature.shape[0]):  # Iterate over the batch dimension
                flattened_feature = feature[i].reshape(-1)  # Flatten each feature map (one per sample in the batch)
                all_flattened_features.append(np.asarray(flattened_feature))  # Convert to CuPy array and store

        # Stack all flattened features into a single 2D array
        all_features = np.vstack(all_flattened_features)
        return all_features
    
    # used to load pretrained weights into model example: model.load_weights(path/to/weights.pt)
    def load_weights(self, path):
        pretrained_weights_path = path
        pretrained_weights = torch.load(pretrained_weights_path)
        self.load_state_dict(pretrained_weights, strict=False)




def main():
    model = YOLOv8mModified(nc=80)
    model.load_weights('/home/alan/Documents/YOLOv8_custom/yolov8m.pt')
    print(model)
    print("Model loaded with pretrained weights.")
#main()