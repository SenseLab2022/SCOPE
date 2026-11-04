'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pdb
import torchvision
from torch.autograd import Variable


__all__ = ['resnet12_1w4a']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, idx=0, layer_idx=0, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shortcut1 = nn.Sequential()
        self.activation1 = nn.Hardtanh()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.act = nn.Hardtanh()
        self.k = torch.tensor([10]).float()
        self.t = torch.tensor([0.1]).float()
        self.layer_idx = layer_idx
        self.idx = idx
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out += self.shortcut(x)
        out = self.act(out)

        x1 = out
        out = self.conv2(x1)
        out = self.bn2(out)
        out += x1
        out = self.act(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, layer_idx=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, layer_idx=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, layer_idx=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.act = nn.Hardtanh()
        self.linear = nn.Linear(64, num_classes, bias=True)

        # self.linear = nn.Linear(64, num_classes, bias=False)
        self.apply(_weights_init)
        self.k = torch.tensor([10]).float()
        self.t = torch.tensor([0.1]).float()

    def _make_layer(self, block, planes, num_blocks, stride, layer_idx):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        idx = 0
        for stride in strides:
            idx = idx + 1
            layers.append(block(self.in_planes, planes, stride, idx, layer_idx))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])

        out = self.bn2(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class ResNet16(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet16, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act = nn.Hardtanh()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, layer_idx=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, layer_idx=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, layer_idx=3)
        
        

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64) #要注销
        self.activation1 = nn.Hardtanh()
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.activation2 = nn.Hardtanh()
        self.conv4 = nn.Conv2d(16, num_classes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_classes)
        self.activation3 = nn.Hardtanh()

        # self.linear = nn.Linear(64, num_classes, bias=True)

        self.apply(_weights_init)
        self.k = torch.tensor([10]).float()
        self.t = torch.tensor([0.1]).float()

    def _make_layer(self, block, planes, num_blocks, stride, layer_idx):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        idx = 0
        for stride in strides:
            idx = idx + 1
            layers.append(block(self.in_planes, planes, stride, idx, layer_idx))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.conv2(out)
        out = self.bn3(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn4(out)
        out = self.act(out)

        out = self.conv4(out)
        out = self.bn5(out)
        out = self.act(out)
        
        # out = out[:,:10,:,:]
        # out = out[:, [0, 1, 2, 4, 5, 6, 8, 9, 12, 13], :, :]
        out = out.view(out.size(0), -1)
        return out



def resnet14():
    return ResNet(BasicBlock, [2, 2, 2])

def resnet16(num_classes=10):
    return ResNet16(BasicBlock, [2, 2, 2], num_classes)

def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params = total_params + np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))

