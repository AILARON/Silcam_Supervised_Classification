#################################################################################################################
# The Pytorch neural network architecture module
# Implementation of the net class
# Author: Aya Saad
# email: aya.saad@ntnu.no
#
# Date created: 6 April 2020
#
# Project: AILARON
# Contact
# email: annette.stahl@ntnu.no
# funded by RCN IKTPLUSS program (project number 262701) and supported by NTNU AMOS
# Copyright @NTNU 2020
#######################################

import torch
import torch.nn as nn
from layers_2D import *
from layers import *
from torchsummary import summary
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.main = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride=1,
            #                  padding=0, dilation=1, n_angles = 8, mode=1)
            RotConv(3, 6, [9, 9], 1, 9 // 2, n_angles=17, mode=1),
            VectorMaxPool(2),
            VectorBatchNorm(6),

            RotConv(6, 16, [9, 9], 1, 9 // 2, n_angles=17, mode=2),
            VectorMaxPool(2),
            VectorBatchNorm(16),

            RotConv(16, 32, [9, 9], 1, 1, n_angles=17, mode=2),
            Vector2Magnitude(),

            nn.Conv2d(32, 128, 1),  # FC1
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.7),
            nn.Conv2d(128, 10, 1),  # FC2

        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size()[0], x.size()[1])

        return x

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=False, **kwargs):
        super(Unit, self).__init__()
        self.max_pool = max_pool

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        if (self.max_pool):
            output = self.maxpool(output)

        return output

# a modified version of the COAPNet, added the Batch norm layer after each conv layer
# to improve the network performance
# BN are used to rescale and resample the input for network stability and impove performance
class COAPModNet(nn.Module):
    def __init__(self, num_classes=10):
        super(COAPModNet, self).__init__()
        # The next 6 lines of code were inspired 
        # by Bjarne Kaestad 
        # Environment and New Resources, SINTEF Ocean, Trondheim, Norway 
        self.unit1 = Unit(in_channels=3, out_channels=64, max_pool=True)
        self.unit2 = Unit(in_channels=64, out_channels=128, max_pool=True)
        self.unit3 = Unit(in_channels=128, out_channels=256, max_pool=True)
        self.unit4 = Unit(in_channels=256, out_channels=512, max_pool=True)

        # Add all the units into the Sequential layer in exact order
        self.features = nn.Sequential(self.unit1,
                                      self.unit2,
                                      self.unit3,
                                      self.unit4) # 64: 32-16-8-4

        #Pytorch output shapes from Conv2D
        # Hout = (Hin +2xpadding(0)-dilation(0)x(kernel_size(0)-1)-1)/stride(0) + 1
        # Wout = (Win +2xpadding(1)-dilation(1)x(kernel_size(1)-1)-1)/stride(1) + 1

        # MaxPool2D
        # Hout = (Hin +2xpadding(0)-dilation(0)x(kernel_size(0)-1)-1)/stride(0) + 1
        # Wout = (Win + 2xpadding(1) - dilation(1)x(kernel_size(1) - 1) - 1) / stride(1) + 1
        # -> default padding is 0, default stride = kernel_size dilation=1
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4*4*512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, input):
        output = self.features(input)
        #print('output.shape before flatening', output.shape)
        #output = output.view(output.size(0), -1)
        output = output.view(-1, 4*4*512)
        #print('output.shape after flatening ', output.shape)
        output = self.classifier(output)
        #output = nn.Softmax(dim=1)(output)
        return output

class PlanktonNet(nn.Module):
    # based on the paper
    # @inproceedings{Dai2016,
    #  title={ZooplanktoNet: Deep convolutional network for zooplankton classification},
    #  author={Dai, Jialun and Wang, Ruchen and Zheng, Haiyong and Ji, Guangrong and Qiao, Xiaoyan},
    #  booktitle={OCEANS 2016-Shanghai},
    #  pages={1--6},
    #  year={2016},
    #  organization={IEEE}
    # }
    def __init__(self, num_classes=1000):
        super(PlanktonNet, self).__init__()
        self.features = nn.Sequential(
            # Pytorch output shapes from Conv2D
            # Hout = (Hin +2xpadding(0)-dilation(0)x(kernel_size(0)-1)-1)/stride(0) + 1
            # Wout = (Win +2xpadding(1)-dilation(1)x(kernel_size(1)-1)-1)/stride(1) + 1

            # MaxPool2D
            # Hout = (Hin +2xpadding(0)-dilation(0)x(kernel_size(0)-1)-1)/stride(0) + 1
            # Wout = (Win + 2xpadding(1) - dilation(1)x(kernel_size(1) - 1) - 1) / stride(1) + 1
            # -> default padding is 0, default stride = kernel_size dilation=1
            nn.Conv2d(3, 96, kernel_size=13, stride=1, padding=2),  # -> 56   - stride=4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 27
            #nn.LocalResponseNorm(128),
            nn.Conv2d(96, 256, kernel_size=7, padding=2), # -> 21
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # -> 10
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # -> 10
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # -> 10
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # -> 10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # -> 4

            nn.Conv2d(384, 512, kernel_size=3, padding=1), # -> 4
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # -> 4
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # -> 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # -> 1

        )
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*2*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #print('x.shape before flatten ', x.shape)
        #x = torch.flatten(x, 1)
        x = x.view(-1, 512*2*2)
        #print('x.shape after flatten ', x.shape)
        x = self.classifier(x)
        return x

def load_model(name, num_classes,
               input_height = 64, input_width = 64, num_of_channels = 3):
    if name == 'COAPModNet':
        net = COAPModNet(num_classes=num_classes)
    elif name == 'COAPNet':
        net = COAPNet(num_classes=num_classes)
    elif name == 'SimpleNet':
        net = SimpleNet(num_classes=num_classes)
    elif name == 'AlexNet':
        net = models.AlexNet(num_classes=num_classes)
    elif name == 'PlanktonNet':
        net = PlanktonNet(num_classes=num_classes)
    elif name == 'ResNet18':
        net = models.resnet18(num_of_channels, num_classes)
    elif name == 'ResNet34':
        net = models.resnet34(num_of_channels, num_classes)
    elif name == 'ResNet50':
        net = models.resnet50(num_of_channels, num_classes)
    elif name == 'ResNet101':
        net = models.resnet101(num_of_channels, num_classes)
    elif name == 'ResNet152':
        net = models.resnet152(num_of_channels, num_classes)
    elif name == 'VGGNet11':
        net = models.vgg11(num_classes=num_classes)
    elif name == 'VGGNet13':
        net = models.vgg13(num_classes=num_classes)
    elif name == 'VGGNet16':
        net = models.vgg16(num_classes=num_classes)
    elif name == 'VGGNet19':
        net = models.vgg19(num_classes=num_classes)
    elif name == 'ResNext50':
        net = models.resnext50_32x4d(num_classes=num_classes)
    elif name == 'ResNext101':
        net = models.resnext101_32x8d(num_classes=num_classes)
    elif name == 'GoogLeNet':
        net = models.GoogLeNet(num_classes=num_classes)

    return net

if __name__ == '__main__':
    name = ['COAPModNet', 'COAPNet', 'SimpleNet', 'AlexNet', 'PlanktonNet',
            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
            'VGGNet11', 'VGGNet13', 'VGGNet16', 'VGGNet19',
            'ResNext50', 'ResNext101',
            'GoogLeNet']
    class_list = ['copepod', 'diatom_chain', 'other', 'faecal_pellets', 'bubble', 'oily_gas', 'oil']
    input_height = 64
    input_width = 64
    num_of_channels = 3
    for i in range(len(name)):  # loop over the network names
        print('-- {} ------------ '.format(name[i]))
        net = load_model(name[i], num_classes=len(class_list),
                         input_height=input_height, input_width=input_width,
                         num_of_channels=num_of_channels)
        summary(net, (num_of_channels, input_width, input_height))

    #net = models.AlexNet(num_classes=len(class_list))
    #model = PlanktonNet(6)
