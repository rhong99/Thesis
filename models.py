# PyTorch models


import torch
import torch.nn as nn
from torchvision import models


# base ResNet18
class ResNet18(nn.Module):
    def __init__(self, pretrained, output_size):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained)
        if pretrained:
            print('Using Pretrained')
            checkpoint = torch.load('resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'])
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# using all OpenFace data
class ResNet18_openface(nn.Module):
    def __init__(self, pretrained, output_size):
        super(ResNet18_openface, self).__init__()
        resnet = models.resnet18(pretrained)
        if pretrained:
            print('Using Pretrained')
            checkpoint = torch.load('resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'])
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc1 = nn.Linear(512+710, 1000)
        self.fc2 = nn.Linear(1000, output_size)

    def forward(self, x, x1):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = torch.cat((x, x1), 1).float()
        x = self.fc1(x)
        return self.fc2(x)


# using OpenFace AUs
class ResNet18_openface_au(nn.Module):
    def __init__(self, pretrained, output_size):
        super(ResNet18_openface_au, self).__init__()
        resnet = models.resnet18(pretrained)
        if pretrained:
            print('Using Pretrained')
            checkpoint = torch.load('resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'])
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc1 = nn.Linear(512+17, output_size)
        # self.fc2 = nn.Linear(640, output_size)

    def forward(self, x, x1):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = torch.cat((x, x1), 1).float()
        return self.fc1(x)
        # return self.fc2(x)
