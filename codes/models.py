import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import math
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler

from utils import apply_taylor_softmax

NUM_CLASSES = 10

class CifarHENet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CifarHENet, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Sequential(
    
            nn.Linear(128 * 4 * 4, 256),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x*x
        #print(x[0][0])
        x = self.conv2(x)
        x = self.pool(x)
        x = x*x
        #print(x[0][0])
        x = self.conv3(x)
        x = self.pool(x)
        x = x*x
        #print(x[0][0])
        x = x.view(-1, 128 * 4 * 4)
        logit = self.classifier(x)
        #print(x[0])
        out = 1+logit+0.5*logit**2                                   
        out /= torch.sum(out,axis=1).view(-1,1)
        #print(x)
        return out
    def forward_logit(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x*x
        #print(x[0][0])
        x = self.conv2(x)
        x = self.pool(x)
        x = x*x
        #print(x[0][0])
        x = self.conv3(x)
        x = self.pool(x)
        x = x*x
        #print(x[0][0])
        x = x.view(-1, 128 * 4 * 4)
        logit = self.classifier(x)/100.0
        #print(x[0])        
        #print(x)
        return logit
    
class CifarNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CifarNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Sequential(
    
            nn.Linear(128 * 4 * 4, 256),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.classifier(x)
        return x

class CifarHESmNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_conv = 32, hidden_fc = 128, kernel_size = 3):
        super(CifarHESmNet, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(3, hidden_conv, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.classifier = nn.Sequential(
    
            nn.Linear(hidden_conv * 16 * 16, hidden_fc),
            nn.Linear(hidden_fc, 10),
        )
        self.hidden_conv = hidden_conv
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x*x
        x = x.view(-1, self.hidden_conv * 16 * 16)
        logit = self.classifier(x)
        #print(x[0])
        out = 1+logit+0.5*logit**2                                   
        out /= torch.sum(out,axis=1).view(-1,1)
        #print(x)
        return out
    def forward_logit(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x*x
        x = x.view(-1, self.hidden_conv * 16 * 16)
        logit = self.classifier(x)/100.0
        #print(x[0])
        #print(x)
        return logit

class CifarSmNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_conv = 32, hidden_fc = 128, kernel_size = 3):
        super(CifarSmNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(3, hidden_conv, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_conv * 16 * 16, hidden_fc),
            nn.Linear(hidden_fc, 10),
        )
        self.hidden_conv = hidden_conv
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = x.view(-1, self.hidden_conv * 16 * 16)
        x = self.classifier(x)
        return x

class MnistHENet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(MnistHENet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(5, 50, kernel_size=5, stride=2, padding=0)
        self.classifier = nn.Linear(50*4*4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x*x
        x = self.conv2(x)
        x = x*x
        x = x.view(-1, 50 * 4 * 4)
        x = self.classifier(x)
        x = 1+x+0.5*x**2                                   
        x /= torch.sum(x,axis=1).view(-1,1)
        return x
    def forward_logit(self,x):
        x = self.conv1(x)
        x = x*x
        x = self.conv2(x)
        x = x*x
        x = x.view(-1, 50 * 4 * 4)
        x = self.classifier(x)/100.0        
        return x

class MnistNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(5, 50, kernel_size=5, stride=2, padding=0)
        self.classifier = nn.Linear(50*4*4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(-1, 50 * 4 * 4)
        x = self.classifier(x)
        return x  
    
class MnistHESmNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(MnistHESmNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0)
        self.classifier = nn.Linear(5*12*12, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x*x
        x = x.view(-1, 5 * 12 * 12)
        x = self.classifier(x)
        x = 1+x+0.5*x**2                                 
        x /= torch.sum(x,axis=1).view(-1,1)
        return x
    def forward_logit(self,x):
        x = self.conv1(x)
        x = x*x
        x = x.view(-1, 5 * 12 * 12)
        x = self.classifier(x)/100.0        
        return x

class MnistSmNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(MnistSmNet, self).__init__() #28 28
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0) #5 12 12
        self.classifier = nn.Linear(5*12*12, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = x.view(-1, 5 * 12 * 12)
        x = self.classifier(x)
        return x


