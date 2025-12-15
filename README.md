# Fashion-MNIST Classification with Multi-Layer Perceptron (MLP)

A PyTorch implementation of a Multi-Layer Perceptron neural network for classifying Fashion-MNIST images.

## Overview

This project implements a deep learning model to classify images from the Fashion-MNIST dataset, which consists of 70,000 grayscale images of 10 different clothing categories. The model achieves approximately 90% accuracy on the validation set.

## Project Structure

The implementation includes:
- Data loading and preprocessing for Fashion-MNIST
- MLP model architecture with batch normalization and dropout
- Training pipeline with validation monitoring
- Visualization tools for predictions and performance analysis
- Confusion matrix for model evaluation

## Dataset

**Fashion-MNIST Dataset:**
- 60,000 training images
- 10,000 validation images
- 10 clothing categories:
  - 0: T-shirt/top
  - 1: Trouser
  - 2: Pullover
  - 3: Dress
  - 4: Coat
  - 5: Sandal
  - 6: Shirt
  - 7: Sneaker
  - 8: Bag
  - 9: Ankle boot

## Model Architecture

The MLP model consists of:
- **Input Layer**: 784 neurons (28x28 flattened image)
- **Hidden Layer 1**: 512 neurons with BatchNorm and Dropout (p=0.3)
- **Hidden Layer 2**: 256 neurons with BatchNorm
- **Hidden Layer 3**: 128 neurons with BatchNorm
- **Hidden Layer 4**: 64 neurons with BatchNorm
- **Output Layer**: 10 neurons (one per class) with LogSoftmax

**Total Parameters**: 576,970

## Key Features

### 1. Data Preprocessing
- Image normalization: (0.5, 0.5)
- Data augmentation: Random shuffling
- Batch size: 64

### 2. Training Configuration
- **Optimizer**: Adam with learning rate 1e-2
- **Loss Function**: Negative Log Likelihood Loss
- **Epochs**: 40
- **Device**: GPU (if available)

### 3. Model Components
- **Batch Normalization**: Applied after each linear layer for stable training
- **Dropout**: Only in first hidden layer (p=0.3) for regularization
- **Activation**: ReLU for hidden layers, LogSoftmax for output

### 4. Visualization Tools
- Sample image display with labels
- Training/validation loss and accuracy plots
- Confusion matrix for performance evaluation
- Individual prediction probability visualization

## Performance

The model achieves:
- **Training Accuracy**: ~90% (final epoch)
- **Validation Accuracy**: ~90% (final epoch)
- **Training Loss**: ~0.144 (final epoch)
- **Validation Loss**: ~0.299 (final epoch)

## Installation & Setup

### Prerequisites
```bash
pip install torch torchvision torchinfo numpy matplotlib scikit-learn seaborn
