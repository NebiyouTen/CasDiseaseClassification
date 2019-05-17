'''

Author: nyismaw
Model for Casava image files as a part of an in-class kaggle competition
https://www.kaggle.com/c/cassava-disease/

Adopted from the official pytorch vision repo
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

'''

import torch.nn as nn


def conv5x5(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, bias=bias, padding =1)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)


    def forward(self, x):
        identity = x
        out = self.conv1(x)

        out = self.relu(self.bn1(out))

        out = self.conv2(out)

        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, in_planes=64, layers =[1,1,1,1], num_classes=5):
        super(ResNet, self).__init__()
        self.inplanes = in_planes
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 128)
        self.layer2 = self._make_layer(BasicBlock, 128, stride=1)
        self.layer3 = self._make_layer(BasicBlock, 256, stride=1)
        self.layer4 = self._make_layer(BasicBlock, 512, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, dilate=False):
        layers = []
        layers.append(conv1x1(self.inplanes, planes, stride=stride));
        layers.append(block(planes, planes, stride))
        self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
