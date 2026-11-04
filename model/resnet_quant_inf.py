import torch.nn as nn
import torch
import math
import pdb
from general.logger import log_print
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms

import torch.nn.functional as F
import sys
import os
# import img_counter
from .modules import binaryfunction
from .modules import ir_1w8a_inf

# torch.set_log_printoptions(profile="full")
# torch.set_log_printoptions(precision=4, sci_mode=False)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock_1w4a(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, idx=0, layer_idx =0,option='A'):
        super(BasicBlock_1w4a, self).__init__()

        self.conv1 = ir_1w8a_inf.IRConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # self.relu = nn.ReLU(inplace=True)
        self.activation1 = nn.Hardtanh()
        self.conv2 = ir_1w8a_inf.IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.activation2 = nn.Hardtanh()
        self.k = torch.tensor([10]).float()
        self.t = torch.tensor([0.1]).float()
        self.layer_idx = layer_idx
        self.idx = idx
        self.count = 0
        if stride != 1 or inplanes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    ir_1w8a_inf.IRConv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

            if inplanes == planes:
                self.shortcut = LambdaLayer(lambda x: x[:, :, ::2, ::2])

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

    def my_quantize_7(self, input, prec):
        x = input

        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        # T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        n = float(2 ** prec - 1)
        activation_q = torch.round(activation_q * n)
        # activation_q = activation_q * T

        return activation_q, T

    def my_bn1(self, input, sw_conv1, layer_idx):
        global var, mean
        if self.training:
            y = input
            y = y.permute(1, 0, 2, 3)
            y = y.contiguous().view(y.shape[0], -1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn1.running_mean = self.bn1.momentum * self.bn1.running_mean + (1 - self.bn1.momentum) * mean
            self.bn1.running_var = self.bn1.momentum * self.bn1.running_var + (1 - self.bn1.momentum) * var
        else:  
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

        T_a = 7 * 3 * 3 * 64
        T_w = 1
        activation_bit = 10
        activation_k = 2 ** activation_bit - 1

        bw, T_w = self.my_quantize_7(weight, 3)
        activation_q = torch.floor(torch.round(input / 4) * sw_conv1)

        bb = torch.round(self.my_quantize(bias, (activation_bit + 4)) * 7 * activation_k / T_a * 7 / T_w)    
        
        # error = torch.tensor([3, 0, 5, 8, 7, 0, -3, 3, 2, 9, 1, 4, -2, 0, 0, 5])  # layer2
        # for i in range(16):
        #     bb[i] = bw[i, i] * error[i] * sw_conv1[0, i, 0, 0] + bb[i]

        # log_print(error)

        # # # *********** bn parameters ***************
        # bn_save(bw, bb)
        # # # *********** *************** ***************

        out = torch.round(F.conv2d(activation_q, bw, bb, stride=1, padding=0))

        return out, T_a, T_w

    def my_bn2(self, input, sw_conv2, layer_idx):
        global var2, mean2
        if self.training:
            y2 = input
            y2 = y2.permute(1, 0, 2, 3)
            y2 = y2.contiguous().view(y2.shape[0], -1)
            mean2 = y2.mean(1).detach()
            var2 = y2.var(1).detach()
            self.bn2.running_mean = self.bn2.momentum * self.bn2.running_mean + (1 - self.bn2.momentum) * mean2
            self.bn2.running_var = self.bn2.momentum * self.bn2.running_var + (1 - self.bn2.momentum) * var2
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
        T_w2 = 1
        T_a2 = 7 * 3 * 3 * 64
        bw2, T_w2 = self.my_quantize_7(weight2, 3)

        activation_bit = 10
        activation_k = 2 ** activation_bit - 1

        activation_q2 = torch.floor(torch.round(input / 4) * sw_conv2)
        bb2 = torch.round(self.my_quantize(bias2, (activation_bit + 4)) * 7 * activation_k / T_a2 * 7 / T_w2)

        # if layer_idx == 16:
        #     error = torch.tensor([-2, 0, -1, 2, -2, 3, 7, -2, 7, -1, 5, 0, 2, -3, 4, -2])
        # else:  
        #     error = torch.tensor([3, 0, 5, 8, 7, 0, -3, 3, 2, 9, 1, 4, -2, 0, 0, 5])   # layer2
        # log_print(error)
        # for i in range(16):
        #     bb2[i] = bw2[i, i] * error[i] * sw_conv2[0, i, 0, 0] + bb2[i]

        # # # *********** bn parameters generate ***************
        # bn_save(bw2, bb2)
        # # # *********** *************** ***************
        out = torch.round(F.conv2d(activation_q2, bw2, bb2, stride=1, padding=0))

        return out, T_a2, T_w2

    def forward(self, input_feature):
        # log_print("input",input_feature[0].shape)
   
        [x, count], act_out = input_feature
        self.layer_idx = count + self.count + self.idx * 2 - 1
        # ---------------------conv1-----------------------
        x_cim = x
        cim_conv = ir_1w8a_inf.IRConv2d_cim(self.conv1.in_channels, self.conv1.out_channels, kernel_size=(3, 3),
                                        stride=self.conv1.stride, padding=self.conv1.padding, bias=False)
        cim_conv.weight = self.conv1.weight
        out, sw_conv1, act_out1 = cim_conv(x_cim, self.layer_idx)
        sw_conv1 = sw_conv1.view(1, sw_conv1.shape[0], 1, 1)
        # # # *********** sw1 generate ***************
        # sw1 = sw_conv1.clone()
        # sw_save(sw1, self.layer_idx)
        # # # *********** *************** ***************

        # ---------------------bn1-----------------------
        out, T_a, T_w = self.my_bn1(out, sw_conv1, self.layer_idx)
        # ---------------------shortcut1-----------------------
        short_cut = torch.round(1023 / T_a * 7 / T_w)

        out += torch.round(self.shortcut(x * short_cut))
        # ---------------------hardtanh1-----------------------
        T_hardtanh = torch.round(7 * 1023 / T_a * 7 / T_w).float()
        out1 = torch.round(F.hardtanh(out, min_val=-T_hardtanh, max_val=T_hardtanh) / T_hardtanh * 7)
        # # # *********** *************** ***************
        # hardtanh_save(out, T_hardtanh, short_cut)
        # # # *********** *************** ***************
        out = out1

        # if self.layer_idx == 7:
        #     pdb.set_trace()

        x1 = out
        x1_cim = x1

        num =self.count +1
        # self.count += 1
        self.layer_idx = count + num + self.idx * 2 - 1
        # log_print(self.count)
        # ---------------------conv2-----------------------
        cim_conv2 = ir_1w8a_inf.IRConv2d_cim(self.conv2.in_channels, self.conv2.out_channels, kernel_size=(3, 3),
                                         stride=self.conv2.stride, padding=self.conv2.padding, bias=False)
        cim_conv2.weight = self.conv2.weight
        out, sw_conv2, act_out2 = cim_conv2(x1_cim, self.layer_idx)
        sw_conv2 = sw_conv2.view(1, sw_conv2.shape[0], 1, 1)
        # # # *********** sw2 generate ***************
        # sw2 = sw_conv2.clone()
        # sw_save(sw2, self.layer_idx)
        # # # *********** *************** ***************
        # ---------------------bn2-----------------------
        out, T_a2, T_w2 = self.my_bn2(out, sw_conv2, self.layer_idx)
        # ---------------------add2-----------------------
        short_cut = torch.round(1023 / T_a2 * 7 / T_w2)
        # if self.layer_idx == 12:
        #     short_cut = 63
        out = torch.round(out + (x1 * short_cut))
        # ---------------------hardtanh2-----------------------
        T_hardtanh2 = torch.round(7 * 1023 / T_a2 * 7 / T_w2).float()
        out2 = torch.round(F.hardtanh(out, min_val=-T_hardtanh2, max_val=T_hardtanh2) / T_hardtanh2 * 7)
        # # # *********** *************** ***************
        # hardtanh_save(out, T_hardtanh2, short_cut)
        # # # *********** *************** ***************

        out = out2
        # if self.layer_idx == 8:
        # log_print(self.layer_idx)
            # log_print(out.reshape(-1))
        # log_print(out.shape)
            # exit()
        return [out, count],act_out+[act_out1]+[act_out2]


class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=16):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        # self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        # self.P7_1 = nn.ReLU()
        # self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        # P6_x = self.P6(C5)

        # P7_x = self.P7_1(P6_x)
        # P7_x = self.P7_2(P7_x)

        return [P2_x, P3_x, P4_x, P5_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=2, layer_idx=0,feature_size=64):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, num_features_in // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features_in // 2)
        # self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(num_features_in // 2, num_anchors, kernel_size=3, padding=1)
        # self.act2 = nn.ReLU()

        # self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act3 = nn.ReLU()

        # self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act4 = nn.ReLU()

        # self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)
        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False) 

    def my_bn(self, input):
        global var, mean
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

        log_print(weight.shape, weight)
        log_print(bias.shape ,bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # self.my_bn(out)
        out = F.hardtanh(out)

        out = self.conv2(out)

        return out
        # out = F.hardtanh(out)

        # out = self.conv3(out)
        # out = F.hardtanh(out)

        # out = self.conv4(out)
        # out = F.hardtanh(out)

        # out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        # out = out.permute(0, 2, 3, 1)

        # return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = ir_1w8a_inf.IRConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.Hardtanh()
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], stride =1, layer_idx=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, layer_idx=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, layer_idx=3)
        # self.layer4 = self._make_layer(block, 16, layers[3], stride=2, layer_idx=4)

        # if block == BasicBlock_1w4a:
        #     fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels, self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
        #                  self.layer4[layers[3] - 1].conv2.out_channels]
        # elif block == Bottleneck:
        #     fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
        #                  self.layer4[layers[3] - 1].conv3.out_channels]
        # else:
        #     raise ValueError(f"Block type {block} not understood")

        # self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2],fpn_sizes[3],16)

        # self.regressionModel = RegressionModel(16,num_classes)
        # self.classificationModel = ClassificationModel(64, num_classes=num_classes)

        # self.anchors = Anchors()

        # self.regressBoxes = BBoxTransform()

        # self.clipBoxes = ClipBoxes()

        # self.focalLoss = losses.FocalLoss()

        ##initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # prior = 0.01

        # self.classificationModel.output.weight.data.fill_(0)
        # self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        # self.regressionModel.output.weight.data.fill_(0)
        # self.regressionModel.output.bias.data.fill_(0)

        # self.freeze_bn()

    def _make_layer(self, block, planes, num_blocks, stride, layer_idx):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        idx = 0 
        for stride in strides:
            idx = idx + 1
            layers.append(block(self.inplanes, planes, stride, idx, layer_idx))
            self.inplanes = planes * block.expansion
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
    
    def my_quantize_7(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn2().apply(x_s, prec)
        n = float(2 ** prec - 1)
        activation_q = torch.round(activation_q * n)

        return activation_q, T

    def my_bn(self, input, sw_conv1):
        global var, mean
        if self.training:
            y = input
            y = y.permute(1, 0, 2, 3)  # NCHW -> CNHW
            y = y.contiguous().view(y.shape[0], -1)  # CNHW -> C,NHW
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn1.running_mean = self.bn1.momentum * self.bn1.running_mean + (1 - self.bn1.momentum) * mean
            self.bn1.running_var = self.bn1.momentum * self.bn1.running_var + (1 - self.bn1.momentum) * var
        else:
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

        T_a = 7 * 9 * 64

        bw, T_w = self.my_quantize_7(weight, 3)

        activation_q_test = torch.floor(torch.round(input / 4)* sw_conv1 )
        # activation_q_test = torch.round(input * 1023 / T_a)

        bb = torch.round(self.my_quantize(bias, 12) * 7 / T_w * 7 * 1023 / T_a)

        # error = torch.tensor([3, 0, 5, 8, 7, 0, -3, 3, 2, 9, 1, 4, -2, 0, 0, 5])  # layer2
        # for i in range(16):
        #     bb[i] = bw[i, i] * error[i] * sw_conv1[0, i, 0, 0] + bb[i]
        # log_print(error)
        # # # *********** bn parameters ***************
        # bn_save(bw, bb)
        # # # *********** *************** ***************

        out = torch.round(F.conv2d(activation_q_test, bw, bb, stride=1, padding=0))

        return out, T_a, T_w

    # def freeze_bn(self):
    #     '''Freeze BatchNorm layers.'''
    #     for layer in self.modules():
    #         if isinstance(layer, nn.BatchNorm2d):
    #             layer.eval()

    def forward(self, inputs):

        x, T = self.my_quantize_7(inputs, 3)
        # --------------feature generate---------------
        # if img_counter.cnt < 50:
        #     x_3to16 = torch.zeros(1, 16, 32, 32)
        #     x_3to16[:, :3, :, :] = x
            # gen_feature(x_3to16, 'feature1-50', img_counter.cnt)
        #     img_counter.cnt += 1
        # else:
        #     exit()
        # --------------------------------------------
        x_cim = x
        cim_conv = ir_1w8a_inf.IRConv2d_cim(self.conv1.in_channels, self.conv1.out_channels, kernel_size=(3, 3),
                                        stride=self.conv1.stride, padding=self.conv1.padding, bias=False)
        cim_conv.weight = self.conv1.weight

        
        out, sw_conv1, act_out1 = cim_conv(x_cim, 0)

        sw_conv1 = sw_conv1.view(1, -1, 1, 1)

        # # # *********** sw1 generate ***************
        # sw1 = sw_conv1.clone()
        # sw_save(sw1, 0)
        # # # *********** *************** ***************

        out, T_a, T_w = self.my_bn(out, sw_conv1)

        out1 = torch.round(F.hardtanh(out, min_val=-(7 / T_w * 7 * 1023 / T_a), max_val=(7 / T_w * 7 * 1023 / T_a)) / (
                7 / T_w * 7 * 1023 / T_a) * 7)
        T_hardtanh = 7 / T_w * 7 * 1023 / T_a
        # # # *********** *************** ***************
        # hardtanh_save(out, T_hardtanh, 0)
        # # # *********** *************** ***************
        out = out1        

        [x1,_],act_out2 = self.layer1(([out,0],[act_out1]))
        [x2,_],act_out3 = self.layer2(([x1,4],act_out2))
        [x3,_],act_out4 = self.layer3(([x2,8],act_out3))
        # x4 = self.layer4([x3,12])[0]

        # for i in range(len(act_out4)):
        #     log_print(act_out4[i].shape)
        return x3, act_out4


def resnet18(num_classes=2, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock_1w4a, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock_1w4a, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
