import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import SPPF, C2f
from ultralytics.nn.modules.head import Detect, Segment



class YOLOv8mModified(nn.Module):
    def __init__(self, nc=80, nm=32, npr=256):
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
            Segment(nc=nc, nm=nm, npr=npr, ch=(256, 512, 1024))
        )

    # x= input tensor, features = 2d array with dimensions [n][k], n = num of layers, k = num of training samples
    def forward_all(self, x, features):
        tensor_outputs = [] #store each layers output for use with Concat()
        concat_layers = [None] * 22 #list to store layers to concat with
        # layers to concatenate with the previous layer per .yaml
        concat_layers[11] = 6
        concat_layers[14] = 4
        concat_layers[17] = 12
        concat_layers[20] = 9   
        i=0  #ith layer
        # Forward pass through the backbone
        for layer in self.backbone:
            x = layer.forward(x)
            tensor_outputs.append(x)
            flat_features = self.flatten_1d(x) #flatten the batch of tensors into 1D numpy arrays
            for flat in flat_features:
                features[i].append(flat)
            i+=1

        # Forward pass through the head
        for layer in self.head[:-1]:
            if isinstance(layer, Concat):
                if(concat_layers[i] != None):
                    concat_inputs =  [tensor_outputs[-1], tensor_outputs[concat_layers[i]]]
                    x = layer.forward(concat_inputs)
                else:
                    print("Error - Concat Layer index does not match expected", i)
            else:
                x = layer.forward(x)
                tensor_outputs.append(x)
                flat_features = self.flatten_1d(x) #flatten the batch of tensors into 1D numpy arrays
                for flat in flat_features:
                    features[i].append(flat)
            i+=1

        x = self.head[-1].forward(x)
        return features, x # x contains model output, classification/segmentation results

    
    # Function to flatten the spatial dimensions and channels
    def flatten_features(self, feature_tensor):
        batch_size, num_channels, height, width = feature_tensor.shape
        # Reshape each feature map to collapse spatial dimensions and channels into a single vector per sample
        flattened = feature_tensor.view(batch_size, -1)  # flatten all dimensions except batch dimension
        return flattened

    def flatten_1d(self, feature_tensor):
        # Flatten the spatial dimensions and channels for each tensor in the batch
        flattened = self.flatten_features(feature_tensor)
        # Convert each flattened feature map to a numpy array
        for i in range(len(flattened)):
            flattened[i] = flattened[i].cpu().detach().numpy()
        return flattened
     
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
    #example initialization of features array to be passed to forward_all method during training
    num_layers = len(model.backbone) + len(model.head[:-1])
    features = [[] for _ in range(num_layers)]
main()