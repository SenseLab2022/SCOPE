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
import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.autograd import Variable

from .modules import binaryfunction
from .modules import ir_1w8a


__all__ = ['resnet12_1w4a']


def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_1w8a_q(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, idx=0, layer_idx=0, option='A'):
        super(BasicBlock_1w8a_q, self).__init__()

        groups = planes // 16

        self.conv1 = ir_1w8a.IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ir_1w8a.IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
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
                    ir_1w8a.IRConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

            if in_planes == planes:
                self.shortcut = LambdaLayer(lambda x: x[:, :, ::2, ::2])


    def my_quantize_conv(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        # if Tmin >= 0:
        #     activation_q = binaryfunction.qfn2().apply(x_s, prec)
        # else:
        #     activation_q = binaryfunction.qfn2().apply(x_s * 0.5, prec)

        return activation_q

    def my_quantize(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()  # Tmax[0]  # torch.max(input).detach()#
        Tmin = torch.min(input).detach()  # Tmax[1]  # torch.min(input).detach()#
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        activation_q = activation_q * T
        return activation_q

    def my_quantize_activation(self, input, prec, T_a, sw):
        x = input
        x_s = x / T_a
        activation_q = binaryfunction.qfn().apply(x_s, prec, sw)
        activation_q = activation_q * T_a
        return activation_q

    def my_bn1(self, input, sw, input_channel):
        global var, mean
        if self.training:
            y = input
            y = y.permute(1, 0, 2, 3)  # NCHW -> CNHW
            y = y.contiguous().view(y.shape[0], -1)  # CNHW -> C,NHW
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn1.running_mean = self.bn1.momentum * self.bn1.running_mean + (1 - self.bn1.momentum) * mean
            self.bn1.running_var = self.bn1.momentum * self.bn1.running_var + (1 - self.bn1.momentum) * var
        else:  # BN2?2??üD?
            mean = self.bn1.running_mean
            var = self.bn1.running_var
        std = torch.sqrt(var + self.bn1.eps)
        weight = self.bn1.weight / std
        bias = self.bn1.bias - weight * mean
        weight = weight.view(input.shape[1], 1)

        p3d = (0, input.shape[1] - 1)
        weight = F.pad(weight, p3d, 'constant', 0)
        for i in range(input.shape[1]):
            weight[i][i] = weight[i][0]
            if i > 0:
                weight[i][0] = 0
        weight = weight.view(input.shape[1], input.shape[1], 1, 1)
        T_a = 3 * 3 * 64
        bw = self.my_quantize(weight, 3)
        activation_q = self.my_quantize_activation(input, 10, T_a, sw)
        bb = self.my_quantize(bias, 12)
        out = F.conv2d(activation_q, bw, bb, stride=1, padding=0)

        return out

    def my_bn2(self, input, sw2, input_channel):
        global var2, mean2
        if self.training:
            y2 = input
            y2 = y2.permute(1, 0, 2, 3)
            y2 = y2.contiguous().view(y2.shape[0], -1)
            mean2 = y2.mean(1).detach()
            var2 = y2.var(1).detach()
            self.bn2.running_mean = self.bn2.momentum * self.bn2.running_mean + (1 - self.bn2.momentum) * mean2
            self.bn2.running_var = self.bn2.momentum * self.bn2.running_var + (1 - self.bn2.momentum) * var2
        else:  # BN2?2??üD?
            mean2 = self.bn2.running_mean
            var2 = self.bn2.running_var
        std2 = torch.sqrt(var2 + self.bn2.eps)
        weight2 = self.bn2.weight / std2
        bias2 = self.bn2.bias - weight2 * mean2
        weight2 = weight2.view(input.shape[1], 1)
        p3d2 = (0, input.shape[1] - 1)
        weight2 = F.pad(weight2, p3d2, 'constant', 0)
        for i in range(input.shape[1]):
            weight2[i][i] = weight2[i][0]
            if i > 0:
                weight2[i][0] = 0
        weight2 = weight2.view(input.shape[1], input.shape[1], 1, 1)

        T_a2 = 3 * 3 * 64
        bw2 = self.my_quantize(weight2, 3)

        activation_q2 = self.my_quantize_activation(input, 10, T_a2, sw2)
        bb2 = self.my_quantize(bias2, 12)
        out = F.conv2d(activation_q2, bw2, bb2, stride=1, padding=0)

        return out

    def forward(self, x):
        x = self.my_quantize_conv(x, 3)
        out, sw1 = self.conv1(x)
        sw1 = sw1.view(1, sw1.shape[0], 1, 1)
        out = self.my_bn1(out, sw1, self.conv1.in_channels)
        out += self.shortcut(x)
        out = F.hardtanh(out)
        x1 = self.my_quantize_conv(out, 3)
        out, sw2 = self.conv2(x1)
        sw2 = sw2.view(1, sw2.shape[0], 1, 1)
        out = self.my_bn2(out, sw2, self.conv2.in_channels)
        out += x1
        out = F.hardtanh(out)
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

    def my_quantize(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        activation_q = activation_q * T
        return activation_q

    def my_quantize_activation(self, input, prec, T_a, sw):
        x = input
        x_s = x / T_a
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        activation_q = activation_q * T_a
        return activation_q

    def my_quantize_activation2(self, input, prec, T_a):
        x = input
        x_s = x / T_a
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        activation_q = activation_q * T_a
        return activation_q

    def my_quantize_conv1(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        # activation_q = activation_q * T
        return activation_q

    def my_bn(self, input, sw):
        global var, mean
        if self.training:
            y = input
            y = y.permute(1, 0, 2, 3)  # NCHW -> CNHW
            y = y.contiguous().view(y.shape[0], -1)  # CNHW -> C,NHW
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn1.running_mean = self.bn1.momentum * self.bn1.running_mean + (1 - self.bn1.momentum) * mean
            self.bn1.running_var = self.bn1.momentum * self.bn1.running_var + (1 - self.bn1.momentum) * var
        else:  # BN2?2??üD?
            mean = self.bn1.running_mean
            var = self.bn1.running_var
        std = torch.sqrt(var + self.bn1.eps)
        weight = self.bn1.weight / std
        bias = self.bn1.bias - weight * mean
        weight = weight.view(input.shape[1], 1)

        p3d = (0, input.shape[1] - 1)
        weight = F.pad(weight, p3d, 'constant', 0)
        for i in range(input.shape[1]):
            weight[i][i] = weight[i][0]
            if i > 0:
                weight[i][0] = 0
        weight = weight.view(input.shape[1], input.shape[1], 1, 1)
        T_a = 3 * 3 * self.conv1.weight.shape[1]
        bw = self.my_quantize(weight, 3)

        activation_q = self.my_quantize_activation(input, 10, T_a, sw)
        bb = self.my_quantize(bias, 12)
        out = F.conv2d(activation_q, bw, bb, stride=1, padding=0)

        return out

    def my_bn2(self, input):
        global var2, mean2
        if self.training:
            y2 = input
            y2 = y2.permute(1, 0, 2, 3)
            y2 = y2.contiguous().view(y2.shape[0], -1)
            mean2 = y2.mean(1).detach()
            var2 = y2.var(1).detach()
            self.bn2.running_mean = \
                self.bn2.momentum * self.bn2.running_mean + \
                (1 - self.bn2.momentum) * mean2
            self.bn2.running_var = \
                self.bn2.momentum * self.bn2.running_var + \
                (1 - self.bn2.momentum) * var2
        else:
            mean2 = self.bn2.running_mean
            var2 = self.bn2.running_var
        std2 = torch.sqrt(var2 + self.bn2.eps)
        weight2 = self.bn2.weight / std2
        bias2 = self.bn2.bias - weight2 * mean2
        weight2 = weight2.view(input.shape[1], 1)

        p3d2 = (0, input.shape[1] - 1)
        weight2 = F.pad(weight2, p3d2, 'constant', 0)
        for i in range(input.shape[1]):
            weight2[i][i] = weight2[i][0]
            if i > 0:
                weight2[i][0] = 0
        weight2 = weight2.view(input.shape[1], input.shape[1], 1, 1)
        weight2 = self.my_quantize(weight2, 3)
        activation_q2 = self.my_quantize_activation2(input, 10, 1)
        bb2 = self.my_quantize(bias2, 12)
        out = F.conv2d(activation_q2, weight2, bb2, stride=1, padding=0)
        return out

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.hardtanh(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])

        out = self.bn2(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def forward_features(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.hardtanh(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = self.bn2(out)
        feats = out.view(out.size(0), -1)
        out = self.linear(feats)
        return out, feats


class ResNet_16(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_16, self).__init__()
        self.in_planes = 16
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv1 = ir_1w8a.IRConv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, layer_idx=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2, layer_idx=2)
        self.layer3 = self._make_layer(block, 16, num_blocks[2], stride=2, layer_idx=3)
        self.bn2 = nn.BatchNorm2d(16)
        # self.linear = nn.Linear(16, num_classes, bias=True)
        # self.linear = ir_1w8a.IRlinear(16, num_classes, bias=False)

        self.conv2 = ir_1w8a.IRConv2d(16, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv3 = ir_1w8a.IRConv2d(8, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(4)
        self.conv4 = ir_1w8a.IRConv2d(4, 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(2)

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

    def my_quantize(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        activation_q = activation_q * T
        return activation_q
        

    def my_quantize_activation(self, input, prec, T_a):
        x = input
        x_s = x / T_a
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        activation_q = activation_q * T_a
        return activation_q

    def my_quantize_activation2(self, input, prec, T_a):
        x = input
        x_s = x / T_a
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        activation_q = activation_q * T_a
        return activation_q

    def my_quantize_conv1(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        # activation_q = activation_q * T
        return activation_q

    def my_quantize_conv(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        # if Tmin >= 0:
        #     activation_q = binaryfunction.qfn2().apply(x_s, prec)
        # else:
        #     activation_q = binaryfunction.qfn2().apply(x_s * 0.5, prec)

        return activation_q
    
    def my_bn1(self, input, sw, input_channel):
        global var, mean
        if self.training:
            y = input
            y = y.permute(1, 0, 2, 3)  # NCHW -> CNHW
            y = y.contiguous().view(y.shape[0], -1)  # CNHW -> C,NHW
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn1.running_mean = self.bn1.momentum * self.bn1.running_mean + (1 - self.bn1.momentum) * mean
            self.bn1.running_var = self.bn1.momentum * self.bn1.running_var + (1 - self.bn1.momentum) * var
        else:  # BN2?2??üD?
            mean = self.bn1.running_mean
            var = self.bn1.running_var
        std = torch.sqrt(var + self.bn1.eps)
        weight = self.bn1.weight / std
        bias = self.bn1.bias - weight * mean
        weight = weight.view(input.shape[1], 1)

        p3d = (0, input.shape[1] - 1)
        weight = F.pad(weight, p3d, 'constant', 0)
        for i in range(input.shape[1]):
            weight[i][i] = weight[i][0]
            if i > 0:
                weight[i][0] = 0
        weight = weight.view(input.shape[1], input.shape[1], 1, 1)
        T_a = 3 * 3 * 64
        bw = self.my_quantize(weight, 3)
        activation_q = self.my_quantize_activation(input, 10, T_a)
        bb = self.my_quantize(bias, 12)
        out = F.conv2d(activation_q, bw, bb, stride=1, padding=0)

        return out

    def my_bn2(self, input):
        global var2, mean2
        if self.training:
            y2 = input
            y2 = y2.permute(1, 0, 2, 3)
            y2 = y2.contiguous().view(y2.shape[0], -1)
            mean2 = y2.mean(1).detach()
            var2 = y2.var(1).detach()
            self.bn2.running_mean = \
                self.bn2.momentum * self.bn2.running_mean + \
                (1 - self.bn2.momentum) * mean2
            self.bn2.running_var = \
                self.bn2.momentum * self.bn2.running_var + \
                (1 - self.bn2.momentum) * var2
        else:
            mean2 = self.bn2.running_mean
            var2 = self.bn2.running_var
        std2 = torch.sqrt(var2 + self.bn2.eps)
        weight2 = self.bn2.weight / std2
        bias2 = self.bn2.bias - weight2 * mean2
        weight2 = weight2.view(input.shape[1], 1)

        p3d2 = (0, input.shape[1] - 1)
        weight2 = F.pad(weight2, p3d2, 'constant', 0)
        for i in range(input.shape[1]):
            weight2[i][i] = weight2[i][0]
            if i > 0:
                weight2[i][0] = 0
        weight2 = weight2.view(input.shape[1], input.shape[1], 1, 1)
        weight2 = self.my_quantize(weight2, 3)
        activation_q2 = self.my_quantize_activation2(input, 10, 1)
        bb2 = self.my_quantize(bias2, 12)
        out = F.conv2d(activation_q2, weight2, bb2, stride=1, padding=0)
        return out

    def my_bn3(self, input):
        global var3, mean3
        if self.training:
            y3 = input
            y3 = y3.permute(1, 0, 2, 3)
            y3 = y3.contiguous().view(y3.shape[0], -1)
            mean3 = y3.mean(1).detach()
            var3 = y3.var(1).detach()
            self.bn3.running_mean = self.bn3.momentum * self.bn3.running_mean + (1 - self.bn3.momentum) * mean3
            self.bn3.running_var = self.bn3.momentum * self.bn3.running_var + (1 - self.bn3.momentum) * var3
        else:  # BN2?2??üD?
            mean3 = self.bn3.running_mean
            var3 = self.bn3.running_var
        std3 = torch.sqrt(var3 + self.bn3.eps)
        weight3 = self.bn3.weight / std3
        bias3 = self.bn3.bias - weight3 * mean3
        weight3 = weight3.view(input.shape[1], 1)
        p3d3 = (0, input.shape[1] - 1)
        weight3 = F.pad(weight3, p3d3, 'constant', 0)
        for i in range(input.shape[1]):
            weight3[i][i] = weight3[i][0]
            if i > 0:
                weight3[i][0] = 0
        weight3 = weight3.view(input.shape[1], input.shape[1], 1, 1)

        T_a2 = 3 * 3 * 64
        bw3 = self.my_quantize(weight3, 3)

        activation_q3 = self.my_quantize_activation(input, 10, T_a2)
        bb3 = self.my_quantize(bias3, 12)
        out = F.conv2d(activation_q3, bw3, bb3, stride=1, padding=0)

        return out

    def my_bn4(self, input):
        global var4, mean4
        if self.training:
            y4 = input
            y4 = y4.permute(1, 0, 2, 3)
            y4 = y4.contiguous().view(y4.shape[0], -1)
            mean4 = y4.mean(1).detach()
            var4 = y4.var(1).detach()
            self.bn4.running_mean = self.bn4.momentum * self.bn4.running_mean + (1 - self.bn4.momentum) * mean4
            self.bn4.running_var = self.bn4.momentum * self.bn4.running_var + (1 - self.bn4.momentum) * var4
        else:  # BN2?2??üD?
            mean4 = self.bn4.running_mean
            var4 = self.bn4.running_var
        std4 = torch.sqrt(var4 + self.bn4.eps)
        weight4 = self.bn4.weight / std4
        bias4 = self.bn4.bias - weight4 * mean4
        weight4 = weight4.view(input.shape[1], 1)
        p3d4 = (0, input.shape[1] - 1)
        weight4 = F.pad(weight4, p3d4, 'constant', 0)
        for i in range(input.shape[1]):
            weight4[i][i] = weight4[i][0]
            if i > 0:
                weight4[i][0] = 0
        weight4 = weight4.view(input.shape[1], input.shape[1], 1, 1)

        T_a2 = 3 * 3 * 64
        bw4 = self.my_quantize(weight4, 3)

        activation_q4 = self.my_quantize_activation(input, 10, T_a2)
        bb4 = self.my_quantize(bias4, 12)
        out = F.conv2d(activation_q4, bw4, bb4, stride=1, padding=0)

        return out

    def my_bn5(self, input):
        global var5, mean5
        if self.training:
            y5 = input
            y5 = y5.permute(1, 0, 2, 3)
            y5 = y5.contiguous().view(y5.shape[0], -1)
            mean5 = y5.mean(1).detach()
            var5 = y5.var(1).detach()
            self.bn5.running_mean = self.bn5.momentum * self.bn5.running_mean + (1 - self.bn5.momentum) * mean5
            self.bn5.running_var = self.bn5.momentum * self.bn5.running_var + (1 - self.bn5.momentum) * var5
        else:  # BN2?2??üD?
            mean5 = self.bn5.running_mean
            var5 = self.bn5.running_var
        std5 = torch.sqrt(var5 + self.bn5.eps)
        weight5 = self.bn5.weight / std5
        bias5 = self.bn5.bias - weight5 * mean5
        weight5 = weight5.view(input.shape[1], 1)
        p3d5 = (0, input.shape[1] - 1)
        weight5 = F.pad(weight5, p3d5, 'constant', 0)
        for i in range(input.shape[1]):
            weight5[i][i] = weight5[i][0]
            if i > 0:
                weight5[i][0] = 0
        weight5 = weight5.view(input.shape[1], input.shape[1], 1, 1)

        T_a2 = 3 * 3 * 64
        bw5 = self.my_quantize(weight5, 3)

        activation_q5 = self.my_quantize_activation(input, 10, T_a2)
        bb5 = self.my_quantize(bias5, 12)
        out = F.conv2d(activation_q5, bw5, bb5, stride=1, padding=0)

        return out

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.bn1(out)

        x = self.my_quantize_conv(x, 3)
        out, sw1 = self.conv1(x)

        sw1 = sw1.view(1, sw1.shape[0], 1, 1)
        out = self.my_bn1(out, sw1, self.conv1.in_channels)

        out = F.hardtanh(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.hardtanh(out)

        # print(out[0][1])
        # exit()

        # HxW=8x8
        out = self.my_quantize_conv(out, 3)
        out, _ = self.conv2(out)
        out = self.my_bn3(out)
        out = F.hardtanh(out)

        # HxW=4x4
        out = self.my_quantize_conv(out, 3)
        out, _ = self.conv3(out)
        out = self.my_bn4(out)
        out = F.hardtanh(out)

        # HxW=2x2
        out = self.my_quantize_conv(out, 3)
        out, _ = self.conv4(out)
        out = self.my_bn5(out)
        out = F.hardtanh(out)

        out = out.view(out.size(0), -1)
        # exit()
        # print(out.shape)

        return out
    
    def forward_features(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.hardtanh(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = self.bn2(out)
        feats = out.view(out.size(0), -1)
        out = self.linear(feats)
        return out, feats


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params = total_params + np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


class ResNet14_1w4a_shuffle_fq_down(ResNet):
    def __init__(self, num_classes):
        super(ResNet14_1w4a_shuffle_fq_down, self).__init__(BasicBlock_1w8a_q, [2, 2, 2], num_classes)


class ResNet14_16_1w4a_shuffle_fq_down(ResNet_16):
    def __init__(self, num_classes):
        super(ResNet14_16_1w4a_shuffle_fq_down, self).__init__(BasicBlock_1w8a_q, [2, 2, 2], num_classes)

if __name__ == "__main__":
    model =ResNet14_16_1w4a_shuffle_fq_down(num_classes=2)

    from thop import profile

    input_tensor = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(input_tensor,))

    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")

