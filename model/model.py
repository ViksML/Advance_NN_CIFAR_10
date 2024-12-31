import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Block 1 
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, dilation=1, padding=1, groups=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Depthwise Convolution
            nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2, groups=32),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Pointwise Convolution 
            nn.Conv2d(32, 32, kernel_size=1), 
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Block 2 
        self.block2 = nn.Sequential(
            # Depthwise Convolution
            nn.Conv2d(32, 64, kernel_size=3, dilation=2, padding=2, groups=32),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Depthwise Convolution
            nn.Conv2d(64, 64, kernel_size=3, dilation=3, padding=3, groups=64), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Pointwise Convolution
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Block 3 
        self.block3 = nn.Sequential(
            # Depthwise Convolution
            nn.Conv2d(64, 128, kernel_size=3, dilation=3, padding=3, groups=64),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Depthwise Convolution
            nn.Conv2d(128, 128, kernel_size=3, dilation=4, padding=4, groups=128),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Pointwise Convolution
            nn.Conv2d(128, 128, kernel_size=1), 
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Block 4 
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, dilation=4, padding=4, groups=128),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, dilation=5, padding=5, groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Pointwise Convolution
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Global pooling (RF covers entire input)
            nn.AdaptiveAvgPool2d(1)  
        )

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        #self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x