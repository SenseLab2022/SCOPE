'''
用于将用户提供的网络（全浮点），在forward更改，做激活值和权重的量化
'''
import sys
import os
# from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import pdb
import torch.fx as fx
from torch.fx import symbolic_trace
from torch.fx import GraphModule


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.modules import ir_1w8a,binaryfunction
from model.resnet import resnet16,BasicBlock
from general.parameters import *

tensor_list = []

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def replace_conv_with_irconv(model, irconv_cls):
    """
    replace all nn.Conv2d to class IRConv2d
    
    """
    for name, module in model.named_children():
        # print(name,module)
        if isinstance(module, nn.Conv2d):
            device = module.weight.device
            new_module = irconv_cls(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None)
            ).to(device)

            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)

            setattr(model, name, new_module)

        else:
            # 递归替换子模块
            replace_conv_with_irconv(module, irconv_cls)

    return model

def replace_forward(model, target_performance=1):
    """
    replace forward function of Conv2d and BatchNorm2d to add quantization function
    """
    for name, module in model.named_modules():
        # print(name, type(module))
        if isinstance(module, nn.Conv2d):
            orig_forward = module.forward
            def quan_act (self,x, ori_forward = orig_forward):
                if INFERENCE:
                    x, T_a = my_quantize_7(x, ACTIVATION_BIT-1)
                else: 
                    x = my_quantize_conv(x, ACTIVATION_BIT-1)
                return ori_forward(x)
            module.forward = types.MethodType(quan_act, module)
            print(f"[INFO] Add Quantization layer: {name}")

        elif isinstance(module, nn.BatchNorm2d) :
            def bn_forward(self, x):
                if INFERENCE:
                    output, sw, activation_list = x
                    if target_performance:
                        global tensor_list
                        tensor_list.append(activation_list)
                    out,T_a,T_w = my_bn_cim(self,output)
                    
                    return out, T_a, T_w
                else:
                    output = x
                    out = my_bn(self, output)
                return out
            
            module.forward = types.MethodType(bn_forward, module)
            print(f"[INFO] Replace BN layer: {name}")

        elif isinstance(module, ResidualAdd):
            orig_forward = module.forward
            if INFERENCE:
                def shortcut_forward(self, out, x, orig_forward= orig_forward):
                    # pdb.set_trace()
                    output,T_a,T_w =  out
                    short_cut = torch.round(1023 / T_a * 7 / T_w)
                    
                                
                    return orig_forward(output,x*short_cut),T_a,T_w

                module.forward = types.MethodType(shortcut_forward, module)
                print(f"[INFO] Replace shortcut: {name}")



        elif isinstance(module, nn.Hardtanh) or module == F.hardtanh:
            if INFERENCE:
                def hardtanh_forward(self, x):
                    # pdb.set_trace()
                    output,T_w,T_a = x
                    out1 = torch.round(F.hardtanh(output, min_val=-(7 / T_w * 7 * 1023 / T_a), max_val=(7 / T_w * 7 * 1023 / T_a)) / (
                    7 / T_w * 7 * 1023 / T_a) * 7)
                
                    return out1

                module.forward = types.MethodType(hardtanh_forward, module)
                print(f"[INFO] Replace hardtanh layer: {name}")

    return model


def replace_forward_function(model: nn.Module) -> GraphModule:
    traced: GraphModule = symbolic_trace(model)
    modules = dict(traced.named_modules())
    hardtanh_count = 0
    residual_count = 0
    planes = 0
    args_save =None
    count = 1

    for node in list(traced.graph.nodes):  
        args_node = node.args
        
        if len(args_node) >= 1 and isinstance(args_node[0], fx.Node) and args_node[0].op == 'call_module' and "act" in str(args_node[0].target):
            args_save = args_node[0]


        elif len(args_node) >= 1 and isinstance(args_node[0], fx.Node) and args_node[0].op == 'call_module' and "conv" in str(args_node[0].target):
            bn_mod = modules.get(node.target, None)
            planes = bn_mod.num_features

        elif node.op == 'call_function' and node.target.__name__ == 'add':
            if (len(args_node) == 2 and isinstance(args_node[1], fx.Node)and args_node[1].op == 'call_function' and 'bn' in str(args_node[0].target)):
                count *= 2
                shortcut_mod = LambdaLayer(lambda x,count=count :F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (16*count) // 4, (16*count) // 4), "constant",0))
                # print("short F.pad")

            elif(len(args_node) == 2 and isinstance(args_node[1], fx.Node)and args_node[1].op == 'call_module' and 'bn' in str(args_node[0].target)):
                shortcut_mod = nn.Sequential()
                args_save = args_node[1]
                # print("short nn.Sequential")
           
            name = f'residual_add_{residual_count}'
            residual_add = ResidualAdd(shortcut_mod)
            setattr(traced, name, residual_add)

            with traced.graph.inserting_after(node):
                # print(args_node[0],args_save)
                new_node = traced.graph.call_module(name, args=(args_node[0], args_save))
                node.replace_all_uses_with(new_node)
                traced.graph.erase_node(node)
            residual_count += 1

    traced.graph.lint()
    traced.recompile()
    
    return traced

def my_quantize_conv(input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn2().apply(x_s, prec)

        return activation_q

def my_quantize_7(input, prec):
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

def my_quantize_activation(input, prec, T_a):
    x = input
    x_s = x / T_a
    activation_q = binaryfunction.qfn2().apply(x_s, prec)
    activation_q = activation_q * T_a
    return activation_q

def my_quantize(input, prec):
    x = input
    Tmax = torch.max(input).detach()
    Tmin = torch.min(input).detach()
    T_abs = torch.max(torch.abs(Tmin), torch.abs(Tmax))
    T_abs = torch.clamp(T_abs, 1e-10, 255.)
    x = torch.clamp(x, 0 - T_abs, T_abs)
    x_s = x / T_abs
    activation_q = binaryfunction.qfn2().apply(x_s, prec)
    activation_q = activation_q * T_abs
    return activation_q

def my_bn(bn_module, input):
    if bn_module.training:
        y = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
        mean = y.mean(1).detach()
        var = y.var(1).detach()
        bn_module.running_mean = bn_module.momentum * bn_module.running_mean + (1 - bn_module.momentum) * mean
        bn_module.running_var = bn_module.momentum * bn_module.running_var + (1 - bn_module.momentum) * var
    else:
        mean = bn_module.running_mean
        var = bn_module.running_var

    std = torch.sqrt(var + bn_module.eps)
    weight = bn_module.weight / std
    bias = bn_module.bias - weight * mean
    weight = weight.view(input.shape[1], 1)

    # pad and construct bw
    p3d = (0, input.shape[1] - 1)
    weight = F.pad(weight, p3d, 'constant', 0)
    for i in range(input.shape[1]):
        weight[i][i] = weight[i][0]
        if i > 0:
            weight[i][0] = 0
    weight = weight.view(input.shape[1], input.shape[1], 1, 1)

    T_a = 9 * 64
    bw = my_quantize(weight, 3)
    bit_input = (ACTIVATION_BIT-1) + ADC_BIT
    activation_q = my_quantize_activation(input, bit_input-1, T_a)
    bb= my_quantize(bias, 12)
    out = F.conv2d(activation_q, bw, bb, stride=1, padding=0)

    return out

def my_bn_cim(bn_module, input):
    if bn_module.training:
        y = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
        mean = y.mean(1).detach()
        var = y.var(1).detach()
        bn_module.running_mean = bn_module.momentum * bn_module.running_mean + (1 - bn_module.momentum) * mean
        bn_module.running_var = bn_module.momentum * bn_module.running_var + (1 - bn_module.momentum) * var
    else:
        mean = bn_module.running_mean
        var = bn_module.running_var

    std = torch.sqrt(var + bn_module.eps)
    weight = bn_module.weight / std
    bias = bn_module.bias - weight * mean
    weight = weight.view(input.shape[1], 1)

    # pad and construct bw
    p3d = (0, input.shape[1] - 1)
    weight = F.pad(weight, p3d, 'constant', 0)
    for i in range(input.shape[1]):
        weight[i][i] = weight[i][0]
        if i > 0:
            weight[i][0] = 0
    weight = weight.view(input.shape[1], input.shape[1], 1, 1)


    T_a = 7 * 9 * 64
    bit_input = (ACTIVATION_BIT-1) + ADC_BIT -1
    bw, T_w = my_quantize_7(weight, 3)
    # activation_q_test = torch.floor(torch.round(input / 4)* sw_conv1 )
    activation_q_test = torch.round(input * (2**bit_input-1) / T_a)

    bb = torch.round(my_quantize(bias, 12) * 7 / T_w * 7 * 1023 / T_a)

    # error = torch.tensor([3, 0, 5, 8, 7, 0, -3, 3, 2, 9, 1, 4, -2, 0, 0, 5])  # layer2
    # for i in range(16):
    #     bb[i] = bw[i, i] * error[i] * sw_conv1[0, i, 0, 0] + bb[i]
    # log_print(error)
    # # # *********** bn parameters ***************
    # bn_save(bw, bb)
    # # # *********** *************** ***************

    out = torch.round(F.conv2d(activation_q_test, bw, bb, stride=1, padding=0))

    return out, T_a, T_w


class ResidualAdd(nn.Module):
    def __init__(self, shortcut_module):
        super().__init__()
        self.shortcut = shortcut_module 

    def forward(self, out, x):
        if isinstance(out, tuple):

            output = out[0]
            T_a,T_w = out[1:]
            short_cut = torch.round(1023 / T_a * 7 / T_w)
            return output+self.shortcut(x*short_cut)
        else:
            output = out      
            return output + self.shortcut(x)

if __name__ == "__main__":

    model = resnet16().to("cuda")
    model.load_state_dict(torch.load("../resnet16_60000.pth",map_location="cuda"))
    model.eval()
    if INFERENCE:
        model_replaced = replace_conv_with_irconv(model, ir_1w8a.IRConv2d)
    else:
        model_replaced = replace_conv_with_irconv(model, ir_1w8a.IRConv2d_cim)
    input_data = torch.randn(1,3,32,32).to("cuda")
    
    
    model_forward = replace_forward(model_replaced)
    output = model_forward(input_data)
    
    
    
    print(output)


