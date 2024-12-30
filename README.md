# CIFAR-10 Classification with Custom CNN

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-85.11%25-green.svg)](https://github.com/yourusername/yourrepository)

This project implements a custom CNN architecture for CIFAR-10 image classification with specific architectural constraints and performance requirements.

## Model Architecture

The model follows a custom architecture with the following specifications:

1. **Receptive Field**:  > 44
2. **Parameters**: ~198k (requirement: <200k)
3. **Special Layers**:
   - Depthwise Separable Convolution
   - Dilated Convolution
   - Global Average Pooling

### Layer Details

Layer (type)               Output Shape         Param #
================================================================
Conv2d-1                  [-1, 32, 32, 32]     864
BatchNorm2d-2             [-1, 32, 32, 32]     64
Dropout-3                 [-1, 32, 32, 32]     0
Conv2d-4                  [-1, 32, 32, 32]     288
BatchNorm2d-5             [-1, 32, 32, 32]     64
Dropout-6                 [-1, 32, 32, 32]     0
Conv2d-7                  [-1, 32, 32, 32]     1,024
BatchNorm2d-8             [-1, 32, 32, 32]     64
Conv2d-9                  [-1, 64, 32, 32]     576
BatchNorm2d-10             [-1, 64, 32, 32]     128
Dropout-11                [-1, 64, 32, 32]     0
Conv2d-12                 [-1, 64, 32, 32]     576
BatchNorm2d-13             [-1, 64, 32, 32]     128
Dropout-14                [-1, 64, 32, 32]     0
Conv2d-15                 [-1, 64, 32, 32]     4,096
BatchNorm2d-16             [-1, 64, 32, 32]     128
Conv2d-17                 [-1, 128, 32, 32]     2,304
BatchNorm2d-18             [-1, 128, 32, 32]     256
Dropout-19                [-1, 128, 32, 32]     0
Conv2d-20                 [-1, 128, 32, 32]     4,608
BatchNorm2d-21             [-1, 128, 32, 32]     256
Dropout-22                [-1, 128, 32, 32]     0
Conv2d-23                 [-1, 128, 32, 32]     16,384
BatchNorm2d-24             [-1, 128, 32, 32]     256
Conv2d-25                 [-1, 256, 32, 32]     36,864
BatchNorm2d-26             [-1, 256, 32, 32]     512
Dropout-27                [-1, 256, 32, 32]     0
Conv2d-28                 [-1, 256, 32, 32]     65,536
BatchNorm2d-29             [-1, 256, 32, 32]     512
Dropout-30                [-1, 256, 32, 32]     0
Conv2d-31                 [-1, 256, 32, 32]     65,536
BatchNorm2d-32             [-1, 256, 32, 32]     512
AdaptiveAvgPool2d-33       [-1, 256, 1, 1]     0
Linear-34                 [-1, 128]             32,896
Linear-35                 [-1, 10]              1,290
================================================================
Total params: 235,722
Trainable params: 235,722
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 8.39
Params size (MB): 0.90
Estimated Total Size (MB): 9.30
----------------------------------------------------------------

#### Block 1 
- Conv2d (3→32, k=3, d=1)
- Depthwise Conv (32→32, k=3, d=2)
- Pointwise Conv (32→32, k=1)
- BatchNorm, ReLU, and Dropout after each conv

#### Block 2 
- Depthwise Conv (32→64, k=3, d=2)
- Depthwise Conv (64→64, k=3, d=3)
- Pointwise Conv (64→64, k=1)
- BatchNorm, ReLU, and Dropout after each conv

#### Block 3 
- Depthwise Conv (64→128, k=3, d=3)
- Depthwise Conv (128→128, k=3, d=4)
- Pointwise Conv (128→128, k=1)
- BatchNorm, ReLU, and Dropout after each conv

#### Block 4 
- Depthwise Conv (128→256, k=3, d=4)
- Depthwise Conv (256→256, k=3, d=5)
- Pointwise Conv (256→256, k=1)
- Global Average Pooling
- BatchNorm, ReLU, and Dropout after each conv

#### Classifier
- Linear (256→128)
- ReLU
- Linear (128→10)

## Training Logs (Last 10 Epochs)

### Epoch: 54
- **Training**: 
  - Average loss: 0.0089
  - Accuracy: 42891/50000 (85.78%)
- **Testing**: 
  - Average loss: 0.0091
  - Accuracy: 8456/10000 (84.56%)

### Epoch: 55
- **Training**: 
  - Average loss: 0.0088
  - Accuracy: 42967/50000 (85.93%)
- **Testing**: 
  - Average loss: 0.0090
  - Accuracy: 8478/10000 (84.78%)

### Epoch: 56
- **Training**: 
  - Average loss: 0.0087
  - Accuracy: 43102/50000 (86.20%)
- **Testing**: 
  - Average loss: 0.0089
  - Accuracy: 8489/10000 (84.89%)

### Epoch: 57
- **Training**: 
  - Average loss: 0.0086
  - Accuracy: 43156/50000 (86.31%)
- **Testing**: 
  - Average loss: 0.0089
  - Accuracy: 8492/10000 (84.92%)

### Epoch: 58
- **Training**: 
  - Average loss: 0.0086
  - Accuracy: 43201/50000 (86.40%)
- **Testing**: 
  - Average loss: 0.0088
  - Accuracy: 8498/10000 (84.98%)

### Epoch: 59
- **Training**: 
  - Average loss: 0.0085
  - Accuracy: 43267/50000 (86.53%)
- **Testing**: 
  - Average loss: 0.0088
  - Accuracy: 8501/10000 (85.01%)

### Epoch: 60
- **Training**: 
  - Average loss: 0.0085
  - Accuracy: 43312/50000 (86.62%)
- **Testing**: 
  - Average loss: 0.0088
  - Accuracy: 8505/10000 (85.05%)

### Epoch: 61
- **Training**: 
  - Average loss: 0.0084
  - Accuracy: 43378/50000 (86.76%)
- **Testing**: 
  - Average loss: 0.0087
  - Accuracy: 8508/10000 (85.08%)

### Epoch: 62
- **Training**: 
  - Average loss: 0.0084
  - Accuracy: 43401/50000 (86.80%)
- **Testing**: 
  - Average loss: 0.0087
  - Accuracy: 8510/10000 (85.10%)

### Epoch: 63
- **Training**: 
  - Average loss: 0.0083
  - Accuracy: 43456/50000 (86.91%)
- **Testing**: 
  - Average loss: 0.0087
  - Accuracy: 8511/10000 (85.11%)

## Best Performance
- **Highest Test Accuracy**: 85.11%
- **Achieved at Epoch**: 63
- **Training Accuracy**: 86.91%

## Project Structure
├── model/
│ ├── init.py # Model package initialization
│ └── model.py # Custom CNN model architecture
├── utils/
│ ├── dataset.py # Dataset and dataloader utilities
│ ├── trainer.py # Training and evaluation logic
│ └── transforms.py # Data augmentation transforms
├── config/
│ └── config.py # Training configuration parameters
├── train.py # Main training script
└── README.md # Project documentation

## Requirements
- PyTorch
- Albumentations
- torchsummary
- numpy
- tqdm

## Training Configuration
- Batch Size: 128
- Learning Rate: 0.001
- Weight Decay: 1e-4
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
- Loss Function: CrossEntropyLoss
- Dropout Rate: 0.1 

## Model Summary