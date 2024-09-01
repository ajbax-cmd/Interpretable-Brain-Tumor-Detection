# Case-Based Interpretability in Medical Imaging: A Hybrid Approach Using Convolutional Neural Networks and Vector Libraries 

## Table of contents
* [Overview](#overview)
* [Requirements](#requirements)
* [Usage](#usage)
* [Experiment Setup](#experiment-setup)
* [Class Files](#Class-Files)

## Overview
This project demonstrates how to add post-hoc interpretability to Convolutional Neural Networks (CNNs) using vector libraries. By mapping the training set to the vector space of the most relevant layer, we can retrieve the k-nearest neighbors for inference images, providing insight into the model's output.
#### Methodology
We developed wrapper classes that take a trained CNN and its training set as inputs. These classes enable us to:
* Map the training set to the vector space of the most relevant layer to the model's output
* Reduce dimensionality using Principal Component Analysis (PCA)
* Retrieve the k-nearest neighbors for inference images
#### Results
We tested our approach on the YOLOv8m model using the BR35H brain tumor dataset. Our results show that layer 210, with PCA dimension reduction to 30, provides the most reliable results for identifying the most influential training images for a given inference prediction.

<p align="center">
  <img src="https://raw.githubusercontent.com/ajbax-cmd/Interpretable-Brain-Tumor-Detection/master/images_and_graphs/Duplicate_Nearest_Neighbors.png" width="45%" style="height: 30vh; object-fit: cover;" />
  <img src="https://raw.githubusercontent.com/ajbax-cmd/Interpretable-Brain-Tumor-Detection/master/images_and_graphs/YOLOv8_Layer_Pearson.png" width="45%" style="height: 30vh; object-fit: cover;" />
</p>


## Requirements
Ensure the necessary dependencies are installed in your python environment.
```
pip install -r requirements.txt
```
## Usage
For the YOLOv8 wrapper, a data.yaml file with following format should be created for the dataset as follows:
```
path: /path/to/dataset
train: /path/to/train/images
val: /path/to/validate/images
test: /path/to/test/images

names:
  0: tumor
```
The weights of a trained YOLOv8m model will need to be already obtained and saved. To instantiate an instance of the InterpretableYOLOTest class:
```python
data_yaml_path = 'path/to/data.yaml'
weights_path = 'path/to/weights.pt'
weights = torch.load(model_path)
weights =  model['model'].float()
weights.eval() 
target_layer_index = 210 # default parameter value for InterpretableYOLOTest
pca_components = 30 # default parameter value for InterpretableYOLOTest

model = InterpretableYOLOTest(data_yaml_path, model=weights, target_layer_index=target_layer_index, pca_components=pca_components)
```
To perform infernce on an image:
```python
image_path = path/to/inference/image
k = 5 # k nearest neighbors
result = model.single_image_inference(image_path, k=k)
print(result)
```
<p align="center">
  <img src="https://raw.githubusercontent.com/ajbax-cmd/Interpretable-Brain-Tumor-Detection/master/images_and_graphs/prediction_results.png" width="100%" style="height: 4vh; object-fit: cover;" />
</p>



## Experiment Setup
## Class Files


