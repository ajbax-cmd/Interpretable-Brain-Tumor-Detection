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
  <img src="https://raw.githubusercontent.com/ajbax-cmd/Interpretable-Brain-Tumor-Detection/master/Images_and_Graphs/Duplicate_Nearest_Neighbors.png" width="45%" style="height: 30vh; object-fit: cover;" />
  <img src="https://raw.githubusercontent.com/ajbax-cmd/Interpretable-Brain-Tumor-Detection/master/Images_and_Graphs/YOLOv8_Layer_Pearson.png" width="45%" style="height: 30vh; object-fit: cover;" />
</p>


## Requirements
## Usage
## Experiment Setup
## Class Files

